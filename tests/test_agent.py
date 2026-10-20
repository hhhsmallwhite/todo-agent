"""agent 工具调用循环的测试（全程离线，靠 MockLLMClient 按剧本回放）。

这里测的是**协议正确性**，不是"能跑通"：

* 一个 ``tool_call`` 必须配一条 ``role="tool"`` 消息，``tool_call_id`` 严格对齐。
  并行调用时少回一条，真实 OpenAI 接口下一轮直接 400 —— 这是本文件最重要的用例。
* 校验失败要作为 ``role="tool"`` 的内容回灌给模型，让它自己改，而不是抛异常。
* 流式下 ``arguments`` 是**分片**到达的，必须按 ``index`` 累加成完整 JSON 才能解析。
  Mock 故意按 8 字符切片，拼接写错就一定露馅。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent import EMPTY_REPLY_FALLBACK, MAX_TURNS_REPLY, TodoAgent
from app.llm import (
    ScriptedToolCall,
    ScriptedTurn,
    script_bad_args_then_ok,
    script_parallel_tool_calls,
)
from app.storage import TodoRepository
from app.tools import TOOLS

ADD_MILK_SCRIPT: list[ScriptedTurn] = [
    ScriptedTurn(tool_calls=(ScriptedToolCall(name="add_todo", arguments='{"title":"买牛奶"}'),)),
    ScriptedTurn(content="已经帮你记下「买牛奶」。"),
]


def dump(payload: dict[str, Any]) -> str:
    """紧凑 JSON，和真实模型输出一致（也让"分片拼接"更有意义）。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def role_messages(request: dict[str, Any], role: str) -> list[dict[str, Any]]:
    return [message for message in request["messages"] if message["role"] == role]


def endless_tool_script(turns: int = 10) -> list[ScriptedTurn]:
    """永远只调工具、永不给出最终回复的剧本，用来撞 max_turns 兜底。"""
    return [
        ScriptedTurn(tool_calls=(ScriptedToolCall(name="list_todos", arguments="{}"),))
        for _ in range(turns)
    ]


# ---------------------------------------------------------------------------
# 1) 单次工具调用：assistant 与 tool 消息必须成对
# ---------------------------------------------------------------------------


class TestSingleToolCallLoop:
    def test_loop_builds_protocol_valid_message_sequence(
        self, make_agent: Any, repo: TodoRepository
    ) -> None:
        agent, client = make_agent(ADD_MILK_SCRIPT)

        outcome = agent.chat("帮我记一下买牛奶")

        assert outcome.reply == "已经帮你记下「买牛奶」。"
        assert outcome.turns == 2, "一轮调工具 + 一轮给结论 = 2 轮"
        assert len(client.requests) == 2

        # 每轮都必须带 tools，否则模型根本不知道有工具可调。
        for request in client.requests:
            assert request["tool_count"] == len(TOOLS)
            assert request["tool_choice"] == "auto"

        messages = client.requests[1]["messages"]
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]

        assistant, tool_message = messages[2], messages[3]
        declared = [call["id"] for call in assistant["tool_calls"]]
        assert declared == [tool_message["tool_call_id"]], "tool_call_id 必须与 assistant 声明的 id 完全一致"
        assert assistant["tool_calls"][0]["function"]["name"] == "add_todo"
        assert json.loads(tool_message["content"])["ok"] is True

        records = outcome.tool_calls
        assert len(records) == 1
        assert records[0].name == "add_todo"
        assert records[0].ok is True
        assert records[0].arguments == '{"title":"买牛奶"}', "审计记录要保留模型给的参数原文"

        assert [todo.title for todo in repo.list()] == ["买牛奶"]

    def test_plain_answer_without_tool_calls_exits_immediately(self, make_agent: Any) -> None:
        agent, client = make_agent([ScriptedTurn(content="你好，我可以帮你记待办。")])

        outcome = agent.chat("你好")

        assert outcome.reply == "你好，我可以帮你记待办。"
        assert outcome.turns == 1
        assert outcome.tool_calls == []
        assert len(client.requests) == 1, "没有工具调用就不要再多请求一次"

    def test_empty_content_falls_back_to_human_message(self, make_agent: Any) -> None:
        agent, _ = make_agent([ScriptedTurn(content="   ")])

        outcome = agent.chat("嗯")

        assert outcome.reply == EMPTY_REPLY_FALLBACK

    def test_non_string_content_falls_back_without_raising(self, repo: TodoRepository) -> None:
        """畸形网关把 content 给成 dict 时不能抛异常，也不能把 str(dict) 当回复。"""

        class _DictContentClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            def _create(self, **_: Any) -> Any:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content={"text": "垃圾"}, tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )

        agent = TodoAgent(client=_DictContentClient(), repo=repo, max_turns=2, model="mock-model")

        outcome = agent.chat("你好")

        assert outcome.reply == EMPTY_REPLY_FALLBACK, "非字符串正文按'没有正文'处理，不外泄 str(dict)"


# ---------------------------------------------------------------------------
# 2) 并行工具调用：最容易漏的一条 tool 消息
# ---------------------------------------------------------------------------


class TestParallelToolCalls:
    def test_two_tool_calls_produce_two_matching_tool_messages(
        self, make_agent: Any, repo: TodoRepository
    ) -> None:
        script = script_parallel_tool_calls()
        agent, client = make_agent(script)

        outcome = agent.chat("帮我记两件事：交实验报告、买牛奶")

        messages = client.requests[1]["messages"]
        assistants = role_messages(client.requests[1], "assistant")
        tool_messages = role_messages(client.requests[1], "tool")

        assert len(assistants) == 1
        declared = [call["id"] for call in assistants[0]["tool_calls"]]
        assert len(declared) == 2, "剧本这一轮声明了 2 个 tool_call"
        assert len(set(declared)) == 2, "两个 id 不能重复"

        # 核心断言：2 个 tool_call -> 2 条 tool 消息，且 id 顺序一一对应。
        assert len(tool_messages) == 2, "少一条 tool 消息，真实接口下一轮会 400"
        assert [message["tool_call_id"] for message in tool_messages] == declared

        # 位置契约：tool 消息必须紧跟它对应的 assistant 消息之后。
        assert messages.index(assistants[0]) == 2
        assert [message["role"] for message in messages[3:]] == ["tool", "tool"]

        for message in tool_messages:
            assert json.loads(message["content"])["ok"] is True

        assert len(outcome.tool_calls) == 2
        assert all(record.ok for record in outcome.tool_calls)
        assert [todo.title for todo in repo.list()] == ["交操作系统实验报告", "买牛奶"]


# ---------------------------------------------------------------------------
# 3) 校验失败 -> 回传给模型 -> 模型改正
# ---------------------------------------------------------------------------


class TestValidationFeedbackLoop:
    def test_model_sees_structured_error_then_corrects(
        self, make_agent: Any, repo: TodoRepository
    ) -> None:
        agent, client = make_agent(script_bad_args_then_ok())

        outcome = agent.chat("帮我记一下 10月25日 19点 复习高数")

        assert outcome.turns == 3, "错一次 -> 改一次 -> 给结论"
        assert [record.ok for record in outcome.tool_calls] == [False, True]

        # 第二轮请求里必须能看到第一轮的错误回传（否则模型无从修正）。
        feedback_message = role_messages(client.requests[1], "tool")[0]
        feedback = json.loads(feedback_message["content"])
        assert feedback["ok"] is False
        assert feedback["error"] == "invalid_arguments"
        assert {"title", "due_at"} <= {item["field"] for item in feedback["detail"]}

        # 失败的那次绝不能留下脏数据：库里只有改正后的那一条。
        todos = repo.list()
        assert [todo.title for todo in todos] == ["复习高数"]
        assert todos[0].due_at == "2026-10-25 19:00"

    def test_failed_call_is_recorded_as_not_ok(self, make_agent: Any) -> None:
        agent, _ = make_agent(script_bad_args_then_ok())

        outcome = agent.chat("记一下复习高数")

        first = outcome.tool_calls[0]
        assert first.ok is False
        assert "invalid_arguments" in first.result
        assert "due_at" in first.result


# ---------------------------------------------------------------------------
# 4) max_turns 兜底：不抛裸异常，给中文人话
# ---------------------------------------------------------------------------


class TestMaxTurnsFallback:
    def test_sync_stops_after_max_turns_with_readable_reply(self, make_agent: Any) -> None:
        agent, client = make_agent(endless_tool_script(), max_turns=2)

        outcome = agent.chat("一直刷")

        assert outcome.reply == MAX_TURNS_REPLY
        assert "拆小" in outcome.reply, "兜底话术要给用户下一步建议"
        assert outcome.turns == 2
        assert len(client.requests) == 2, "max_turns 是硬上限，不能多打一轮"

    def test_max_turns_is_at_least_one(self, make_agent: Any) -> None:
        agent, _ = make_agent([ScriptedTurn(content="ok")], max_turns=0)

        assert agent.max_turns == 1


# ---------------------------------------------------------------------------
# 5) 流式：分片 arguments 必须被完整拼回
# ---------------------------------------------------------------------------


class TestChatStream:
    def test_fragmented_arguments_are_reassembled(
        self, make_agent: Any, repo: TodoRepository
    ) -> None:
        script = script_parallel_tool_calls()
        agent, client = make_agent(script)

        events = list(agent.chat_stream("帮我记两件事"))

        expected_arguments = [call.arguments for call in script[0].tool_calls]
        # 前提自检：参数确实长于一个分片（Mock 每片 8 字符），否则这条用例测不出拼接 bug。
        assert all(len(arguments) > 8 for arguments in expected_arguments)

        assistant = role_messages(client.requests[1], "assistant")[-1]
        actual_arguments = [call["function"]["arguments"] for call in assistant["tool_calls"]]
        assert actual_arguments == expected_arguments, "流式分片拼接丢了内容"

        # 拼回来的必须是可解析、且语义正确的 JSON。
        for arguments in actual_arguments:
            assert json.loads(arguments)

        tool_messages = role_messages(client.requests[1], "tool")
        assert len(tool_messages) == 2, "流式下的并行调用也要一条 tool_call 配一条 tool 消息"
        assert all(json.loads(message["content"])["ok"] is True for message in tool_messages)

        assert [todo.title for todo in repo.list()] == ["交操作系统实验报告", "买牛奶"]

        tool_results = [text for name, text in events if name == "tool_result"]
        assert len(tool_results) == 2
        assert "交操作系统实验报告" in tool_results[0]
        assert "买牛奶" in tool_results[1]
        assert events[-1][0] == "done"

    def test_stream_emits_tokens_then_done(self, make_agent: Any) -> None:
        agent, _ = make_agent([ScriptedTurn(content="你好，我可以帮你记待办。")])

        events = list(agent.chat_stream("你好"))

        tokens = "".join(text for name, text in events if name == "token")
        assert tokens == "你好，我可以帮你记待办。"
        assert events[-1] == ("done", "你好，我可以帮你记待办。")

    def test_stream_with_tool_without_arguments(self, make_agent: Any) -> None:
        """无参工具在流式下一片 arguments 都没有，不能因此丢掉这次调用。"""
        script = [
            ScriptedTurn(tool_calls=(ScriptedToolCall(name="list_todos", arguments=""),)),
            ScriptedTurn(content="当前没有待办。"),
        ]
        agent, client = make_agent(script)

        events = list(agent.chat_stream("我还有什么事"))

        assert ("tool_call", "list_todos") in events
        tool_message = role_messages(client.requests[1], "tool")[0]
        assert json.loads(tool_message["content"])["ok"] is True
        assert events[-1] == ("done", "当前没有待办。")

    def test_stream_skips_chunks_without_choices_or_delta(
        self, make_agent: Any, repo: TodoRepository
    ) -> None:
        """有些网关会发只有 usage 的空 chunk，不能因此崩掉或吞掉正文。"""

        class _ChunkOnlyClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            def _create(self, **_: Any) -> Iterator[Any]:
                yield SimpleNamespace(choices=[])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=None)])
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="好的", tool_calls=None))]
                )

        agent = TodoAgent(client=_ChunkOnlyClient(), repo=repo, max_turns=2, model="mock-model")

        events = list(agent.chat_stream("你好"))

        assert events == [("token", "好的"), ("done", "好的")]

    def test_stream_ignores_non_string_fragments(self, repo: TodoRepository) -> None:
        """畸形网关把正文/参数分片给成非字符串时，流式循环不能抛 TypeError。"""

        class _GarbageStreamClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            def _create(self, **_: Any) -> Iterator[Any]:
                # 正文给 dict、参数分片给 dict：两处都只认 str，其余丢弃。
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content={"text": "垃圾"}, tool_calls=None))
                    ]
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_x",
                                        type="function",
                                        function=SimpleNamespace(
                                            name="list_todos", arguments={"title": "垃圾"}
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )

        agent = TodoAgent(client=_GarbageStreamClient(), repo=repo, max_turns=2, model="mock-model")

        events = list(agent.chat_stream("你好"))

        assert all(name != "token" for name, _ in events), "非字符串正文不产生 token 事件"
        assert ("tool_call", "list_todos") in events, "函数名是 str，这次调用仍然成立"
        tool_results = [text for name, text in events if name == "tool_result"]
        assert tool_results, "参数按空串处理 -> 无参调用 list_todos，仍有结构化结果"
        assert events[-1][0] == "done"

    def test_stream_tool_call_ids_match_assistant_message(self, make_agent: Any) -> None:
        agent, client = make_agent(ADD_MILK_SCRIPT)

        events = list(agent.chat_stream("记一下买牛奶"))

        assert ("tool_call", "add_todo") in events
        assistant = role_messages(client.requests[1], "assistant")[-1]
        tool_message = role_messages(client.requests[1], "tool")[0]
        assert assistant["tool_calls"][0]["id"] == tool_message["tool_call_id"]

    def test_stream_max_turns_does_not_crash(self, make_agent: Any) -> None:
        agent, client = make_agent(endless_tool_script(), max_turns=2)

        events = list(agent.chat_stream("一直刷"))

        assert events[-1] == ("done", MAX_TURNS_REPLY)
        assert sum(1 for name, _ in events if name == "tool_call") == 2
        assert len(client.requests) == 2


# ---------------------------------------------------------------------------
# 6) 其它协议细节
# ---------------------------------------------------------------------------


class TestProtocolDetails:
    def test_assistant_tool_calls_are_replayed_verbatim(self, make_agent: Any) -> None:
        """assistant 消息里的 name/arguments 必须原样回灌，不能丢字段。"""
        arguments = dump({"title": "买牛奶", "due_at": "2026-10-21 09:30"})
        scripted_call = ScriptedToolCall(name="add_todo", arguments=arguments, call_id="call_fixed")
        agent, client = make_agent(
            [
                ScriptedTurn(tool_calls=(scripted_call,)),
                ScriptedTurn(content="记好了"),
            ]
        )

        agent.chat("记一下")

        assistant = role_messages(client.requests[1], "assistant")[0]
        call = assistant["tool_calls"][0]
        assert call["id"] == "call_fixed"
        assert call["type"] == "function"
        assert call["function"] == {"name": "add_todo", "arguments": arguments}
        assert assistant["content"] is None, "带 tool_calls 的一轮通常没有正文"

    def test_system_prompt_and_user_message_are_first(self, make_agent: Any) -> None:
        from app.agent import SYSTEM_PROMPT

        agent, client = make_agent(ADD_MILK_SCRIPT)

        agent.chat("帮我记一下买牛奶")

        messages = client.requests[0]["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "帮我记一下买牛奶"}

    def test_user_id_isolation_through_agent(self, make_agent: Any, repo: TodoRepository) -> None:
        """agent 必须把 user_id 一路带到工具层，不然多用户就串了。"""
        agent, _ = make_agent(ADD_MILK_SCRIPT)

        agent.chat("记一下买牛奶", user_id="alice")

        assert [todo.title for todo in repo.list(user_id="alice")] == ["买牛奶"]
        assert repo.list(user_id="default") == []


@pytest.mark.parametrize("max_turns", [1, 2, 5])
def test_max_turns_never_exceeds_limit(make_agent: Any, max_turns: int) -> None:
    agent, client = make_agent(endless_tool_script(), max_turns=max_turns)

    outcome = agent.chat("刷")

    assert outcome.turns == max_turns
    assert len(client.requests) == max_turns
    assert outcome.reply == MAX_TURNS_REPLY
