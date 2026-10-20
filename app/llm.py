"""模型客户端层：真实 OpenAI 客户端 + 可离线运行、且**流式真的分片**的 Mock。

为什么要有 Mock，而且这个 Mock 必须"接口形状完全兼容"：
1. 现场演示时网络和额度不可控，``scripts/demo_offline.py`` 靠它保证一定能跑通；
2. 更重要的：流式 ``tool_calls`` 的拼接逻辑（按 index 累加 ``function.arguments`` 字符串）
   是全项目最容易写错的一段。如果 Mock 一次性把完整参数吐出来，这段代码写错了也测不出来，
   所以 Mock 必须**真的把 arguments 切成若干小片**返回，这样拼接才会被真正验证。

Mock 对外暴露的属性名严格对齐 OpenAI SDK：
``client.chat.completions.create(...)`` -> ``resp.choices[0].message.content/.tool_calls``，
``tool_call.id/.type/.function.name/.function.arguments``；流式 chunk 是
``chunk.choices[0].delta.content/.tool_calls[i].index/.id/.function.name/.function.arguments``。
用 ``types.SimpleNamespace`` 构造即可，agent.py 全程用属性访问，不需要区分真假客户端。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import openai

from .config import Settings, get_settings


class LLMConfigError(RuntimeError):
    """模型配置缺失/不合法时抛出，携带给用户看的中文提示。"""


def build_client(settings: Settings | None = None) -> openai.OpenAI:
    """按配置构造真实客户端。

    ``BASE_URL`` 可配是刚需：国内基本不会直连 api.openai.com，而是用
    硅基流动 / DeepSeek 这类 OpenAI 兼容网关。它们只是 host 不同，
    请求体和 ``tool_calls`` 协议完全一致，所以换 base_url 就能换模型提供方。
    """
    config = settings if settings is not None else get_settings()
    if not config.has_api_key:
        raise LLMConfigError(
            "没有读到 API_KEY。请把项目根目录的 .env.example 复制成 .env，"
            "填上 API_KEY（BASE_URL 可选，国内可填硅基流动/DeepSeek 的 OpenAI 兼容地址），"
            "或者用 --mock / scripts/demo_offline.py 离线体验。"
        )
    kwargs: dict[str, Any] = {"api_key": config.api_key, "timeout": 30.0, "max_retries": 1}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return openai.OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Mock 剧本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptedToolCall:
    """剧本里的一次工具调用请求。``arguments`` 必须是 JSON 字符串（和真协议一致）。"""

    name: str
    arguments: str
    call_id: str = ""


@dataclass(frozen=True)
class ScriptedTurn:
    """剧本里的一轮模型输出：要么带 tool_calls，要么只有 content（=最终回复）。"""

    content: str | None = None
    tool_calls: tuple[ScriptedToolCall, ...] = field(default=())


def _split(text: str, size: int) -> list[str]:
    """把字符串切成固定长度的小片，用于模拟流式分片。"""
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _dump(payload: dict[str, Any]) -> str:
    """剧本里的参数用紧凑 JSON（不带空格），更接近真实模型输出，也更考验拼接。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class MockLLMClient:
    """按顺序回放剧本的假客户端。

    :param script: 剧本轮次列表，按顺序消费。
    :param fallback_reply: 剧本用完后继续被调用时的兜底回复（模拟"模型没话说了"）。
    """

    def __init__(
        self,
        script: Sequence[ScriptedTurn],
        model: str = "mock-model",
        fallback_reply: str = "这个问题我先处理到这里，你可以再补充一句细节吗？",
    ) -> None:
        self._script: list[ScriptedTurn] = list(script)
        self._cursor: int = 0
        self._call_seq: int = 0
        self.model = model
        self._fallback_reply = fallback_reply
        # 记录每次请求，测试/演示可以断言"tool 消息数量 == tool_call 数量"。
        self.requests: list[dict[str, Any]] = []
        # 关键：暴露和 OpenAI SDK 一样的访问路径 client.chat.completions.create。
        self.chat = SimpleNamespace(completions=_MockCompletions(self))

    # -- 剧本控制 -------------------------------------------------------------

    @property
    def remaining(self) -> int:
        """剧本还剩几轮，方便调试脚本写多/写少。"""
        return max(0, len(self._script) - self._cursor)

    def next_turn(self) -> ScriptedTurn:
        if self._cursor < len(self._script):
            turn = self._script[self._cursor]
            self._cursor += 1
            return turn
        return ScriptedTurn(content=self._fallback_reply)

    # -- 构造响应对象 ---------------------------------------------------------

    def build_message(self, turn: ScriptedTurn) -> SimpleNamespace:
        """把剧本轮次翻译成 ``message`` 对象（属性名与 OpenAI SDK 一致）。"""
        tool_calls: list[SimpleNamespace] = []
        for scripted in turn.tool_calls:
            self._call_seq += 1
            tool_calls.append(
                SimpleNamespace(
                    id=scripted.call_id or f"call_mock_{self._call_seq}",
                    type="function",
                    function=SimpleNamespace(
                        name=scripted.name,
                        arguments=scripted.arguments,
                    ),
                )
            )
        return SimpleNamespace(
            role="assistant",
            content=turn.content,
            tool_calls=tool_calls or None,
        )

    def build_response(self, turn: ScriptedTurn) -> SimpleNamespace:
        message = self.build_message(turn)
        finish_reason = "tool_calls" if message.tool_calls else "stop"
        return SimpleNamespace(
            id=f"chatcmpl-mock-{self._call_seq}",
            model=self.model,
            choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
        )

    def stream_chunks(self, turn: ScriptedTurn) -> Iterator[SimpleNamespace]:
        """把一轮输出切成流式 chunk。

        故意分两段发送 tool_call：先 ``index + id + function.name``（args 为空串），
        再若干只带 ``function.arguments`` 片段的 chunk（id/name 为 None）。
        这正是真实服务端的行为，也是 agent 里"按 index 拼字符串"必须存在的原因。
        """
        yield _chunk(SimpleNamespace(role="assistant", content=None, tool_calls=None))
        for piece in _split(turn.content or "", 6):
            yield _chunk(SimpleNamespace(role=None, content=piece, tool_calls=None))
        message = self.build_message(turn)
        for index, tool_call in enumerate(message.tool_calls or []):
            yield _chunk(
                SimpleNamespace(
                    role=None,
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=index,
                            id=tool_call.id,
                            type="function",
                            function=SimpleNamespace(name=tool_call.function.name, arguments=""),
                        )
                    ],
                )
            )
            for piece in _split(tool_call.function.arguments, 8):
                yield _chunk(
                    SimpleNamespace(
                        role=None,
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=index,
                                id=None,
                                type=None,
                                function=SimpleNamespace(name=None, arguments=piece),
                            )
                        ],
                    )
                )


def _chunk(delta: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-mock-stream",
        model="mock-model",
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
    )


class _MockCompletions:
    """``client.chat.completions`` 的替身，方法签名和 SDK 一样是 **kwargs。"""

    def __init__(self, client: MockLLMClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools") or []
        self._client.requests.append(
            {
                "model": kwargs.get("model"),
                "stream": bool(kwargs.get("stream")),
                "tool_choice": kwargs.get("tool_choice"),
                "tool_count": len(tools),
                "messages": [dict(message) for message in messages if isinstance(message, dict)],
            }
        )
        turn = self._client.next_turn()
        if kwargs.get("stream"):
            return self._client.stream_chunks(turn)
        return self._client.build_response(turn)


# ---------------------------------------------------------------------------
# 现成剧本：演示、测试、讲解都用得上
# ---------------------------------------------------------------------------


def script_parallel_tool_calls() -> list[ScriptedTurn]:
    """一次返回**两个** tool_call。

    用来验证 agent 是否逐个执行、并给每个 ``tool_call_id`` 各回一条 tool 消息。
    少回一条，真实接口下一轮会直接 400（这是最容易踩的坑）。
    """
    return [
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "交操作系统实验报告", "due_at": "2026-10-21 09:30"}),
                ),
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "买牛奶"}),
                ),
            )
        ),
        ScriptedTurn(content="已经帮你记下两件事：交操作系统实验报告（2026-10-21 09:30 前）和买牛奶。"),
    ]


def script_bad_args_then_ok() -> list[ScriptedTurn]:
    """先给错参数 -> 看到错误回传 -> 改正 -> 最终回复。

    第一轮故意同时错两处：``title`` 全是空格、``due_at`` 写成中文日期。
    用来演示"校验失败不是抛异常，而是回传给模型让它自己改"。
    """
    return [
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "   ", "due_at": "10月25日"}),
                ),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "复习高数", "due_at": "2026-10-25 19:00"}),
                ),
            )
        ),
        ScriptedTurn(content="第一次时间格式写错了，现在已按 2026-10-25 19:00 记好「复习高数」。"),
    ]


def script_full_demo() -> list[ScriptedTurn]:
    """离线 demo 用的完整剧本：查空 -> 并行新增 -> 参数错 -> 改正 -> 完成 -> 复查 -> 总结。"""
    return [
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(name="list_todos", arguments=_dump({"status": "pending"})),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "交操作系统实验报告", "due_at": "2026-10-21 09:30"}),
                ),
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "买牛奶"}),
                ),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                # 故意把时间写成"10月25日"，交给校验层打回给模型
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "复习高数", "due_at": "10月25日"}),
                ),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments=_dump({"title": "复习高数", "due_at": "2026-10-25 19:00"}),
                ),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(name="complete_todo", arguments=_dump({"todo_id": 2})),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(name="list_todos", arguments=_dump({"status": "pending"})),
            )
        ),
        ScriptedTurn(
            content=(
                "帮你记了三件事：①交操作系统实验报告（10-21 09:30 前）"
                "②复习高数（10-25 19:00 前）③买牛奶。买牛奶已经标记完成，"
                "现在还剩两件没做。"
            )
        ),
    ]


__all__ = [
    "LLMConfigError",
    "MockLLMClient",
    "ScriptedToolCall",
    "ScriptedTurn",
    "build_client",
    "script_bad_args_then_ok",
    "script_full_demo",
    "script_parallel_tool_calls",
]
