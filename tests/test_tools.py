"""工具层测试：JSON Schema 契约、参数校验、以及"永不抛异常"的错误回传。

被验证的两条硬约束（都是这个项目的卖点，不是风格问题）：

* ``TOOLS`` 里**只有** add / list / complete 三个工具，没有任何删除能力 ——
  工具集合本身就是权限边界，这条要能被回归测试固定住。
* ``dispatch`` 遇到任何坏输入都返回**结构化 JSON**，绝不抛异常、绝不把
  Pydantic 的英文原文/文档链接糊给模型。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.storage import TodoRepository
from app.tools import TOOL_NAMES, TOOLS, ToolExecutor, summarize_result, tool_names

# 错误回传里绝不能出现的痕迹：出现任何一个都说明有人图省事直接用了 str(exc)。
FORBIDDEN_TRACES: tuple[str, ...] = (
    "docs.pydantic.dev",
    "errors.pydantic.dev",
    "Value error",
    "Traceback",
    "pydantic_core",
)


def parse(raw: str) -> dict[str, Any]:
    """dispatch 的返回值必须是合法 JSON 对象——这一条本身就该被断言。"""
    payload = json.loads(raw)
    assert isinstance(payload, dict), f"dispatch 应返回 JSON 对象，实际是 {type(payload)}"
    return payload


def error_fields(payload: dict[str, Any]) -> set[str]:
    """取出 ``detail`` 里的字段名集合，便于断言"错在哪个字段"。"""
    return {str(item["field"]) for item in payload["detail"]}


def assert_no_leak(raw: str) -> None:
    """错误文本必须干净：不含英文原文、不含文档链接、不留堆栈。"""
    for trace in FORBIDDEN_TRACES:
        assert trace not in raw, f"错误回传泄漏了 {trace!r}：{raw}"


# ---------------------------------------------------------------------------
# 1) 工具定义（JSON Schema）：这是模型唯一能看到的"说明书"
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_only_three_tools_and_no_delete_capability(self) -> None:
        """安全设计：只暴露查询/新增/完成，不存在删除与批量删除。"""
        names = [tool["function"]["name"] for tool in TOOLS]

        assert len(TOOLS) == 3
        assert names == ["add_todo", "list_todos", "complete_todo"]
        assert tuple(names) == TOOL_NAMES
        assert tool_names() == TOOL_NAMES

        forbidden_words = (
            "delete",
            "remove",
            "clear",
            "drop",
            "purge",
            "truncate",
            "batch",
            "bulk",
        )
        for name in names:
            lowered = name.lower()
            assert not any(word in lowered for word in forbidden_words), (
                f"出现了疑似破坏性工具 {name!r}，安全边界被破坏"
            )

    def test_every_tool_declares_required_and_enum_where_needed(self) -> None:
        """``required`` 与 ``enum`` 是给模型的硬约束，缺了模型就会乱填。"""
        by_name = {tool["function"]["name"]: tool["function"] for tool in TOOLS}

        for name, function in by_name.items():
            parameters = function["parameters"]
            assert parameters["type"] == "object", name
            assert isinstance(parameters.get("required"), list), f"{name} 缺 required"
            assert parameters.get("additionalProperties") is False, f"{name} 应禁止额外字段"
            assert function["description"].strip(), f"{name} 的 description 不能为空"

        assert by_name["add_todo"]["parameters"]["required"] == ["title"]
        assert by_name["complete_todo"]["parameters"]["required"] == ["todo_id"]
        assert by_name["list_todos"]["parameters"]["required"] == []

        status_schema = by_name["list_todos"]["parameters"]["properties"]["status"]
        assert status_schema["enum"] == ["pending", "done"], "status 必须是枚举，否则模型会传 all/completed"

        # 每个 property 都要有 description：它是提示词，模型靠它决定填什么。
        for name, function in by_name.items():
            for prop, schema in function["parameters"]["properties"].items():
                assert schema.get("description", "").strip(), f"{name}.{prop} 缺 description"


# ---------------------------------------------------------------------------
# 2) 正常路径
# ---------------------------------------------------------------------------


class TestDispatchSuccess:
    def test_add_todo_writes_through_to_storage(
        self, executor: ToolExecutor, repo: TodoRepository
    ) -> None:
        raw = executor.dispatch("add_todo", '{"title": "买牛奶", "due_at": "2026-10-21 09:30"}')

        payload = parse(raw)
        assert payload["ok"] is True
        todo = payload["data"]["todo"]
        assert todo["title"] == "买牛奶"
        assert todo["due_at"] == "2026-10-21 09:30"
        assert todo["status"] == "pending"
        assert todo["id"] >= 1

        # 关键：不是"返回了 ok 就算成功"，必须真的落库。
        stored = repo.get(todo["id"])
        assert stored is not None
        assert (stored.title, stored.due_at) == ("买牛奶", "2026-10-21 09:30")
        assert "已创建待办" in payload["data"]["message"]

    def test_list_then_complete_end_to_end(self, executor: ToolExecutor) -> None:
        executor.dispatch("add_todo", '{"title": "交实验报告"}')

        listed = parse(executor.dispatch("list_todos", '{"status": "pending"}'))
        assert listed["ok"] is True
        assert listed["data"]["count"] == 1
        todo_id = listed["data"]["todos"][0]["id"]

        completed = parse(executor.dispatch("complete_todo", json.dumps({"todo_id": todo_id})))
        assert completed["ok"] is True
        assert completed["data"]["todo"]["status"] == "done"
        assert completed["data"]["todo"]["done_at"]

        assert parse(executor.dispatch("list_todos", '{"status": "pending"}'))["data"]["count"] == 0

    def test_empty_arguments_string_is_treated_as_empty_object(
        self, executor: ToolExecutor
    ) -> None:
        """模型对无参工具常传空字符串，这不该被当成坏 JSON。"""
        payload = parse(executor.dispatch("list_todos", ""))

        assert payload["ok"] is True
        assert payload["data"]["count"] == 0

    def test_summarize_result_extracts_human_readable_summary(self, executor: ToolExecutor) -> None:
        ok, summary = summarize_result(executor.dispatch("add_todo", '{"title": "买牛奶"}'))

        assert ok is True
        assert "买牛奶" in summary


# ---------------------------------------------------------------------------
# 3) 参数校验：失败必须"回传"而不是"抛异常"
# ---------------------------------------------------------------------------


class TestDispatchValidation:
    def test_invalid_due_at_format_returns_structured_error(self, executor: ToolExecutor) -> None:
        raw = executor.dispatch("add_todo", '{"title": "交实验报告", "due_at": "10月25日"}')

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] == "invalid_arguments"
        detail = payload["detail"]
        assert "due_at" in error_fields(payload)
        due_msg = next(item["msg"] for item in detail if item["field"] == "due_at")
        assert "YYYY-MM-DD HH:MM" in due_msg, f"应告诉模型正确格式，实际：{due_msg}"
        assert_no_leak(raw)

    def test_broken_json_returns_structured_error(self, executor: ToolExecutor) -> None:
        raw = executor.dispatch("add_todo", '{"title":')

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] == "bad_json"
        assert payload["detail"][0]["field"] == "arguments"
        assert_no_leak(raw)

    def test_unknown_tool_returns_structured_error(self, executor: ToolExecutor) -> None:
        raw = executor.dispatch("delete_all_todos", "{}")

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] == "unknown_tool"
        # 把可用工具列出来，模型下一轮才有依据改。
        message = payload["detail"][0]["msg"]
        assert "add_todo" in message and "list_todos" in message and "complete_todo" in message
        assert_no_leak(raw)

    @pytest.mark.parametrize("arguments", ["", "null", "[]", "  ", "123", '"str"'])
    def test_degenerate_arguments_never_raise(self, executor: ToolExecutor, arguments: str) -> None:
        raw = executor.dispatch("add_todo", arguments)

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] in {"bad_json", "invalid_arguments"}
        assert_no_leak(raw)

    def test_unknown_extra_field_is_rejected(self, executor: ToolExecutor) -> None:
        """模型爱自作多情传 due_date / priority，必须打回而不是静默忽略。"""
        raw = executor.dispatch("add_todo", '{"title": "买牛奶", "priority": "high"}')

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] == "invalid_arguments"
        assert "priority" in error_fields(payload)
        assert_no_leak(raw)

    def test_blank_title_is_rejected(self, executor: ToolExecutor, repo: TodoRepository) -> None:
        raw = executor.dispatch("add_todo", '{"title": "   "}')

        payload = parse(raw)
        assert payload["ok"] is False
        assert "title" in error_fields(payload)
        assert repo.list() == [], "空白 title 绝不能入库"

    @pytest.mark.parametrize("payload", ['{"todo_id": 0}', '{"todo_id": -3}', '{"todo_id": "abc"}'])
    def test_invalid_todo_id_is_rejected_in_chinese(
        self, executor: ToolExecutor, payload: str
    ) -> None:
        raw = executor.dispatch("complete_todo", payload)

        result = parse(raw)
        assert result["ok"] is False
        assert result["error"] == "invalid_arguments"
        assert "todo_id" in error_fields(result)
        message = result["detail"][0]["msg"]
        assert any(token in message for token in ("应为整数", "大于等于 1")), (
            f"错误提示必须是人能看懂的中文，实际：{message}"
        )
        assert_no_leak(raw)

    def test_invalid_status_enum_is_rejected(self, executor: ToolExecutor) -> None:
        raw = executor.dispatch("list_todos", '{"status": "completed"}')

        payload = parse(raw)
        assert payload["ok"] is False
        assert "status" in error_fields(payload)
        assert_no_leak(raw)

    def test_complete_missing_todo_returns_not_found_error(self, executor: ToolExecutor) -> None:
        """不存在的 id 属于"模型的输入错了"，要回传让它先去查 list_todos。"""
        raw = executor.dispatch("complete_todo", '{"todo_id": 999}')

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] == "todo_not_found"
        assert "list_todos" in payload["detail"][0]["msg"]

    def test_validation_errors_carry_no_pydantic_documentation(
        self, executor: ToolExecutor
    ) -> None:
        """集中断言：所有失败路径的文本都不含文档链接/英文原文。"""
        bad_inputs = [
            ("add_todo", "{}"),
            ("add_todo", '{"title": ""}'),
            ("add_todo", '{"title": 1}'),
            ("list_todos", "[]"),
            ("complete_todo", "{}"),
        ]

        for name, arguments in bad_inputs:
            raw = executor.dispatch(name, arguments)
            payload = parse(raw)
            assert payload["ok"] is False, f"{name} {arguments} 应该失败"
            assert_no_leak(raw)
            for item in payload["detail"]:
                assert item["msg"].strip(), "错误信息不能为空"


class TestExecutorFaultTolerance:
    """最后一层闸：工具实现内部炸了，也只能回传 internal_error，不能让异常冒到 HTTP。"""

    class _BrokenRepository:
        """任何操作都抛异常的假仓库，用来模拟磁盘满/表被删。"""

        def add(self, **_: Any) -> None:
            raise RuntimeError("disk on fire")

        def list(self, **_: Any) -> None:
            raise RuntimeError("disk on fire")

        def complete(self, **_: Any) -> None:
            raise RuntimeError("disk on fire")

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("add_todo", '{"title": "买牛奶"}'),
            ("list_todos", "{}"),
            ("complete_todo", '{"todo_id": 1}'),
        ],
    )
    def test_unexpected_failure_becomes_internal_error(self, name: str, arguments: str) -> None:
        executor = ToolExecutor(self._BrokenRepository())  # type: ignore[arg-type]

        raw = executor.dispatch(name, arguments)

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] == "internal_error"
        assert "服务端" in payload["detail"][0]["msg"]
        # 原始异常文本不能进模型上下文（既浪费 token，也可能泄漏内部信息）。
        assert "disk on fire" not in raw
        assert_no_leak(raw)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(123, "123"), (None, "default"), ("   ", "default"), ("alice", "alice")],
    )
    def test_user_id_is_normalized_at_construction(
        self, repo: TodoRepository, raw: Any, expected: str
    ) -> None:
        """``user_id`` 也走同一类归一化：把 app 当库用时传 None / 数字不该抛 AttributeError。"""
        executor = ToolExecutor(repo, user_id=raw)

        assert executor.user_id == expected

    def test_non_string_user_id_does_not_break_dispatch(self, repo: TodoRepository) -> None:
        executor = ToolExecutor(repo, user_id=None)

        payload = parse(executor.dispatch("add_todo", '{"title": "买牛奶"}'))

        assert payload["ok"] is True
        assert [todo.user_id for todo in repo.list(user_id="default")] == ["default"]


class TestNonStringArguments:
    """``arguments`` 不是字符串时的类型闸（曾经会抛 AttributeError，现已修）。

    协议规定它一定是 string，但"手写调用 / 网关先解析成 dict"都可能破坏这个前提。
    既然 ``dispatch`` 对外承诺"永不抛异常"，这道闸就必须在 ``.strip()`` 之前。
    """

    @pytest.mark.parametrize(
        ("label", "arguments"),
        [
            ("dict", {"title": "买牛奶"}),
            ("int", 123),
            ("list", ["买牛奶"]),
            ("tuple", ("买牛奶",)),
            ("bytes", b'{"title": "buy milk"}'),
            ("empty_list", []),
        ],
    )
    def test_non_string_arguments_returns_structured_error(
        self, executor: ToolExecutor, label: str, arguments: Any
    ) -> None:
        raw = executor.dispatch("add_todo", arguments)

        payload = parse(raw)
        assert payload["ok"] is False, f"{label} 应该被类型闸拦下"
        assert payload["error"] == "bad_json"
        assert payload["detail"][0]["field"] == "arguments"
        # 提示里要教模型"传序列化后的字符串"，否则它不知道怎么改。
        assert "字符串" in payload["detail"][0]["msg"]
        assert_no_leak(raw)

    def test_none_arguments_keeps_old_semantics(self, executor: ToolExecutor) -> None:
        """``None`` 等价于"没有参数"，不能被类型闸误判成 bad_json。"""
        listed = parse(executor.dispatch("list_todos", None))

        assert listed["ok"] is True, "无参工具传 None 必须照常成功"
        assert listed["data"]["count"] == 0

        created = parse(executor.dispatch("add_todo", None))
        assert created["ok"] is False
        assert created["error"] == "invalid_arguments", "None 应走缺字段校验，而不是 bad_json"
        assert "title" in error_fields(created)

    def test_type_gate_runs_before_tool_lookup(self, executor: ToolExecutor) -> None:
        """未知工具 + 非字符串参数：先报参数类型问题，仍然是结构化错误、不抛异常。"""
        raw = executor.dispatch("delete_all_todos", {"x": 1})

        payload = parse(raw)
        assert payload["ok"] is False
        assert payload["error"] in {"bad_json", "unknown_tool"}
        assert_no_leak(raw)


# ---------------------------------------------------------------------------
# 4) 模块级便捷函数
# ---------------------------------------------------------------------------


class TestModuleLevelDispatch:
    def test_dispatch_accepts_explicit_repository(self, repo: TodoRepository) -> None:
        from app.tools import dispatch as module_dispatch

        payload = parse(module_dispatch("add_todo", '{"title": "买牛奶"}', repo=repo))

        assert payload["ok"] is True
        assert len(repo.list()) == 1
