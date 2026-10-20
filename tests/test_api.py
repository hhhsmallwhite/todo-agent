"""FastAPI 层测试：OpenAPI 契约、参数校验、以及"绝不把 500 和堆栈泄漏给调用方"。

两条重点：

* 没配 API_KEY 时 ``POST /chat`` 必须是 **503**（服务端没准备好），不是 500。
* 模型调用抛异常时必须是 **502**，且响应体里不能出现 ``Traceback`` / 原始异常文本。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent import TodoAgent
from app.llm import MockLLMClient, ScriptedToolCall, ScriptedTurn
from app.storage import TodoRepository

LEAK_MARKERS: tuple[str, ...] = ("Traceback", "File \"", "pydantic_core", "openai.APIError")


def assert_no_stack_leak(body: str) -> None:
    for marker in LEAK_MARKERS:
        assert marker not in body, f"响应体泄漏了内部细节 {marker!r}：{body}"


class TestOpenApiContract:
    def test_openapi_lists_chat_and_todos(self, api_client: Any) -> None:
        response = api_client.get("/openapi.json")

        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/chat" in paths
        assert "/todos" in paths
        assert set(paths["/chat"]) == {"post"}
        assert set(paths["/todos"]) == {"get"}

    def test_swagger_ui_is_served(self, api_client: Any) -> None:
        response = api_client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_schema_pins_message_length_limit(self, api_client: Any) -> None:
        """长度上限必须体现在对外发布的 OpenAPI 里，否则调用方（前端/自动化）无从得知。"""
        schema = api_client.get("/openapi.json").json()

        # 先确认 /chat 真的引用这个 schema，否则断言一个没人用的定义毫无意义。
        request_body = schema["paths"]["/chat"]["post"]["requestBody"]
        reference = request_body["content"]["application/json"]["schema"]["$ref"]
        assert reference == "#/components/schemas/ChatRequest"

        message = schema["components"]["schemas"]["ChatRequest"]["properties"]["message"]
        assert message["maxLength"] == 2000
        assert message["minLength"] == 1


class TestRequestValidation:
    @pytest.mark.parametrize(
        "payload",
        [{}, {"message": ""}, {"message": "   "}, {"message": None}, {"message": 123}],
    )
    def test_chat_rejects_bad_body_with_422(self, api_client: Any, payload: dict[str, Any]) -> None:
        response = api_client.post("/chat", json=payload)

        assert response.status_code == 422

    def test_todos_rejects_unknown_status_with_422(self, api_client: Any) -> None:
        response = api_client.get("/todos", params={"status": "bogus"})

        assert response.status_code == 422

    def test_todos_accepts_valid_status_filter(
        self, api_client: Any, api_repo: TodoRepository
    ) -> None:
        api_repo.add("还没做的")
        done = api_repo.add("做完的")
        api_repo.complete(done.id)

        pending = api_client.get("/todos", params={"status": "pending"})
        finished = api_client.get("/todos", params={"status": "done"})

        assert pending.status_code == 200
        assert [todo["title"] for todo in pending.json()] == ["还没做的"]
        assert [todo["title"] for todo in finished.json()] == ["做完的"]


class TestListTodos:
    def test_returns_json_list(self, api_client: Any, api_repo: TodoRepository) -> None:
        api_repo.add("买牛奶", due_at="2026-10-21 09:30")

        response = api_client.get("/todos")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["title"] == "买牛奶"
        assert body[0]["due_at"] == "2026-10-21 09:30"
        assert body[0]["status"] == "pending"

    def test_empty_database_returns_empty_list(self, api_client: Any) -> None:
        response = api_client.get("/todos")

        assert response.status_code == 200
        assert response.json() == []

    def test_user_id_query_isolates_data(self, api_client: Any, api_repo: TodoRepository) -> None:
        api_repo.add("alice 的", user_id="alice")
        api_repo.add("bob 的", user_id="bob")

        response = api_client.get("/todos", params={"user_id": "alice"})

        assert [todo["title"] for todo in response.json()] == ["alice 的"]


class TestRepositoryDependency:
    """``get_repository`` 的兜底分支：没走 lifespan 时退回进程级默认仓库。"""

    def test_falls_back_to_default_repository_when_state_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from app import api, storage

        monkeypatch.setattr(storage, "_DEFAULT_REPOSITORY", None)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

        first = api.get_repository(request)
        second = api.get_repository(request)

        assert isinstance(first, storage.TodoRepository)
        assert first is second, "默认仓库必须是单例，否则每次请求都新建连接"
        assert request.app.state.repo is first, "兜底取到的仓库要挂回 app.state 复用"
        # 兜底路径也建好表了，纯读接口不会因为依赖没初始化而 500。
        assert first.list() == []


class TestChatWithoutApiKey:
    def test_missing_api_key_returns_503_not_500(self, api_client: Any) -> None:
        """配置缺失属于"服务端没准备好"：503 + 一句能照做的人话。"""
        response = api_client.post("/chat", json={"message": "帮我记一下买牛奶"})

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "API_KEY" in detail, f"要告诉调用方去配 API_KEY，实际：{detail}"
        assert_no_stack_leak(response.text)


class TestMessageLengthLimit:
    """``message`` 有 2000 字上限：超长输入由 Pydantic 挡在门外，不丢给模型烧 token。

    这条约束是 ``ChatRequest.max_length`` 的对外契约，用边界值钉住，防止将来被悄悄删掉。
    """

    def test_exactly_at_limit_passes_validation(self, api_client: Any) -> None:
        response = api_client.post("/chat", json={"message": "长" * 2000})

        # 本测试环境没配 API_KEY，所以通过校验后必然落到 503；关键是"没被 422 拦下"。
        assert response.status_code == 503

    @pytest.mark.parametrize("length", [2001, 10_000, 100_000])
    def test_over_limit_is_rejected_with_422(self, api_client: Any, length: int) -> None:
        response = api_client.post("/chat", json={"message": "长" * length})

        assert response.status_code == 422, "超长 message 必须是 422，不能 500"
        assert_no_stack_leak(response.text)
        error = response.json()["detail"][0]
        assert error["type"] == "string_too_long"
        assert error["loc"] == ["body", "message"], "错误必须能定位到 message 字段"

    def test_whitespace_only_counts_as_too_short(self, api_client: Any) -> None:
        """``str_strip_whitespace`` 先于长度约束：全空格等于空字符串，应报 422。"""
        response = api_client.post("/chat", json={"message": "   "})

        assert response.status_code == 422


class TestChatWithMockedAgent:
    """把 agent 换成离线 Mock，验证 HTTP 层到存储层的完整链路。"""

    def test_chat_returns_reply_tool_calls_and_turns(
        self, api_client: Any, api_repo: TodoRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app import api

        script = [
            ScriptedTurn(
                tool_calls=(ScriptedToolCall(name="add_todo", arguments='{"title":"买牛奶"}'),)
            ),
            ScriptedTurn(content="已经帮你记下「买牛奶」。"),
        ]

        def fake_build_agent(repo: TodoRepository, **_: Any) -> TodoAgent:
            return TodoAgent(
                client=MockLLMClient(script), repo=repo, max_turns=8, model="mock-model"
            )

        monkeypatch.setattr(api, "build_agent", fake_build_agent)

        response = api_client.post("/chat", json={"message": "帮我记一下买牛奶"})

        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "已经帮你记下「买牛奶」。"
        assert body["turns"] == 2
        assert len(body["tool_calls"]) == 1
        assert body["tool_calls"][0]["name"] == "add_todo"
        assert body["tool_calls"][0]["ok"] is True

        # 端到端：接口写进去的数据，能在纯读接口里查到。
        assert [todo["title"] for todo in api_client.get("/todos").json()] == ["买牛奶"]
        assert [todo.title for todo in api_repo.list()] == ["买牛奶"]

    def test_model_failure_is_502_without_stack_trace(
        self, api_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app import api

        def boom(*_: Any, **__: Any) -> None:
            raise RuntimeError("upstream exploded: secret-token-123")

        monkeypatch.setattr(api, "build_agent", boom)

        response = api_client.post("/chat", json={"message": "你好"})

        assert response.status_code == 502
        assert_no_stack_leak(response.text)
        assert "secret-token-123" not in response.text, "不要把上游原始异常文本透给调用方"

    def test_max_turns_fallback_is_surfaced_through_api(
        self, api_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app import api

        script = [
            ScriptedTurn(tool_calls=(ScriptedToolCall(name="list_todos", arguments="{}"),))
            for _ in range(5)
        ]

        def fake_build_agent(repo: TodoRepository, **_: Any) -> TodoAgent:
            return TodoAgent(
                client=MockLLMClient(script), repo=repo, max_turns=2, model="mock-model"
            )

        monkeypatch.setattr(api, "build_agent", fake_build_agent)

        response = api_client.post("/chat", json={"message": "刷"})

        assert response.status_code == 200
        body = response.json()
        assert body["turns"] == 2
        assert "拆小" in body["reply"], "turns 与兜底话术要一起暴露给前端"
