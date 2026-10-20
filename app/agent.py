"""工具调用循环（agent 的核心）。

为什么是**同步**实现而不是 ``async def``：
OpenAI 的同步 SDK 是阻塞 IO。如果在 ``async def`` 里直接 ``client.chat.completions.create()``，
整个事件循环会被这一次网络请求卡住，其他请求全部排队。要么用 AsyncOpenAI + await，
要么就让这段代码保持同步、交给 FastAPI 的线程池跑（``def`` 路由 = 线程池，天然安全）。
这个项目选后者：代码更短、更好讲，也不会写出"看似异步实际串行"的假异步。

循环里三件必须讲清楚的事：
1. 每一轮都要把 ``tools=TOOLS`` 一起发过去，否则模型不会调用工具（它不知道有什么工具）。
2. 模型返回的 assistant 消息（含 ``tool_calls`` 结构）必须**原样入库**，
   因为 OpenAI 协议要求 tool 消息必须能对应上前面那条 assistant 的 tool_calls。
3. **一个 tool_call 对应一条 tool 消息**，``tool_call_id`` 一一对应。
   少一条、或者 id 对不上，下一轮请求会直接 400 invalid_request_error。
   一次返回多个 tool_call（并行调用）时这条尤其容易漏。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from .config import get_settings
from .llm import ScriptedTurn, build_client
from .models import ChatOutcome, ToolCallRecord
from .storage import TodoRepository
from .tools import TOOLS, ToolExecutor, summarize_result

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "你是一个待办事项助手，通过调用工具帮用户管理待办。\n"
    "规则：\n"
    "1. 需要读写待办时必须调用工具，不要凭空编造待办内容或 id。\n"
    "2. 标记完成前，id 必须来自 list_todos 的真实返回，不要自己编号。\n"
    "3. 用户没有明确说时间，就不要填 due_at。时间格式只能是 YYYY-MM-DD HH:MM。\n"
    "4. 工具返回 {\"ok\": false} 时，按 detail 里的提示修正参数后重试，"
    "不要把内部错误原样甩给用户。\n"
    "5. 最终回复用中文，简短说明你做了什么。"
)

MAX_TURNS_REPLY: str = (
    "这个需求需要的步骤有点多，我先在这里停下。你可以把它拆小一点再问我，"
    "比如「帮我加一条明天九点交实验报告的待办」。"
)

EMPTY_REPLY_FALLBACK: str = "我这边没有得到有效回复，请再说一次或换个说法试试。"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """安全地按属性取值。

    真实 SDK 的 message 是 Pydantic 模型，Mock 是 ``SimpleNamespace``，
    两者都支持属性访问；但字段可能缺失（例如没有 tool_calls 时 SDK 给 None，
    Mock 干脆没有这个属性），所以统一用一个 getattr 包装，agent 里不出现裸属性访问。
    """
    return getattr(obj, name, default)


class TodoAgent:
    """把「用户一句话」变成「工具调用 + 自然语言回复」的循环。"""

    def __init__(
        self,
        client: Any,
        repo: TodoRepository,
        max_turns: int = 8,
        model: str | None = None,
    ) -> None:
        """
        :param client: 真实的 ``openai.OpenAI`` 或 :class:`app.llm.MockLLMClient`。
        :param repo: 待办仓库，工具执行时使用。
        :param max_turns: 最多几轮模型调用。兜底用，防止模型陷入"调工具-报错-再调"的死循环。
        :param model: 模型名；不传则读配置。
        """
        self._client = client
        self._repo = repo
        self._max_turns = max(1, int(max_turns))
        self._model = model or get_settings().model

    @property
    def max_turns(self) -> int:
        return self._max_turns

    # -- 非流式 ---------------------------------------------------------------

    def chat(self, message: str, user_id: str = "default") -> ChatOutcome:
        """跑完整个工具调用循环，返回最终回复和工具调用记录。"""
        executor = ToolExecutor(self._repo, user_id=user_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        records: list[ToolCallRecord] = []
        turns = 0

        for turns in range(1, self._max_turns + 1):
            response = self._create_completion(messages, stream=False)
            choice = response.choices[0]
            assistant_message = choice.message
            tool_calls = list(_get(assistant_message, "tool_calls", None) or [])

            if not tool_calls:
                # 没有工具调用 = 模型认为可以直接回答，这一轮就是终点。
                # content 只认 str：畸形网关可能给出非字符串（dict / None / 数字），
                # 一律按"没有正文"处理走兜底话术——把 str(dict) 甩给用户比报错更难看。
                content = _get(assistant_message, "content", None)
                reply = content.strip() if isinstance(content, str) else ""
                return ChatOutcome(
                    reply=reply or EMPTY_REPLY_FALLBACK,
                    tool_calls=records,
                    turns=turns,
                )

            # 关键：assistant 消息要带着 tool_calls 结构入库（协议要求），
            # 而且必须排在它对应的 tool 消息**前面**。
            messages.append(_assistant_message_dict(assistant_message))
            LOGGER.info("第 %d 轮：模型请求 %d 个工具调用", turns, len(tool_calls))

            for tool_call in tool_calls:
                name = _get(_get(tool_call, "function", None), "name", "") or ""
                arguments = _get(_get(tool_call, "function", None), "arguments", "") or ""
                call_id = _get(tool_call, "id", "") or ""

                # dispatch 内部已经把所有异常吞成结构化 JSON，这里不会抛。
                result = executor.dispatch(name, arguments)
                # 一个 tool_call 必须配一条 tool 消息，且 tool_call_id 严格对齐。
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": result}
                )

                ok, summary = summarize_result(result)
                records.append(
                    ToolCallRecord(name=name, arguments=arguments, ok=ok, result=summary)
                )
                LOGGER.info("工具 %s ok=%s 参数=%s", name, ok, arguments)

        # 走到这里说明轮数用尽还没拿到最终回复。不抛裸异常，给一句人话 + 真实轮数。
        LOGGER.warning("达到 max_turns=%d 仍未得到最终回复", self._max_turns)
        return ChatOutcome(reply=MAX_TURNS_REPLY, tool_calls=records, turns=turns)

    # -- 流式 -----------------------------------------------------------------

    def chat_stream(self, message: str, user_id: str = "default") -> Iterator[tuple[str, str]]:
        """流式版本，yield ``(事件名, 内容)`` 供 CLI/前端边收边显示。

        事件：
        * ``("token", 文本片段)`` —— 正文增量
        * ``("tool_call", 工具名)`` —— 模型决定调用某工具
        * ``("tool_result", 结果 JSON 摘要)`` —— 工具执行完
        * ``("done", 最终文本)`` —— 本轮对话结束

        核心难点：流式下 tool_calls 是**分片**到达的。
        第一片带 ``index`` + ``id`` + ``function.name``，后续片只带 ``function.arguments``
        的一小段。所以必须用 ``index`` 做 key 累积参数串，**流结束后再 json 解析执行**；
        边收边解析一定会拿到半截 JSON。
        """
        executor = ToolExecutor(self._repo, user_id=user_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        for turn in range(1, self._max_turns + 1):
            stream = self._create_completion(messages, stream=True)
            # index -> {"id": str, "name": str, "arguments": str}
            buffers: dict[int, dict[str, str]] = {}
            content_parts: list[str] = []

            for chunk in stream:
                choices = list(_get(chunk, "choices", None) or [])
                if not choices:
                    continue  # 有些网关会发只有 usage 的空 chunk，跳过即可
                delta = _get(choices[0], "delta", None)
                if delta is None:
                    continue

                text = _get(delta, "content", None)
                # 和 chat() 里的 content 同一道闸：流式分片按协议必须是 str，
                # 非字符串的"正文"直接丢弃，不能 append 进去（join 时会 TypeError）。
                if isinstance(text, str) and text:
                    content_parts.append(text)
                    yield ("token", text)

                for delta_call in list(_get(delta, "tool_calls", None) or []):
                    index = _get(delta_call, "index", 0)
                    index = 0 if index is None else int(index)
                    slot = buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    call_id = _get(delta_call, "id", None)
                    if call_id:
                        slot["id"] = call_id
                    function = _get(delta_call, "function", None)
                    if function is not None:
                        # 分片同样只认 str：非字符串片段会导致 += 抛 TypeError，
                        # 把整条流打断。丢弃即可——参数缺失会走"空串 -> 无参"分支。
                        name_piece = _get(function, "name", None)
                        if isinstance(name_piece, str) and name_piece:
                            slot["name"] += name_piece
                        arguments_piece = _get(function, "arguments", None)
                        if isinstance(arguments_piece, str) and arguments_piece:
                            # 这里就是"按 index 拼字符串"的那一行，缺了它参数只会剩最后一片。
                            slot["arguments"] += arguments_piece

            if not buffers:
                # 没有任何工具调用 -> 这一轮就是最终回复。
                final_text = "".join(content_parts).strip() or EMPTY_REPLY_FALLBACK
                yield ("done", final_text)
                return

            # 有工具调用：先把 assistant 消息补进上下文（协议要求带 tool_calls 结构）。
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": buffers[index]["id"],
                            "type": "function",
                            "function": {
                                "name": buffers[index]["name"],
                                "arguments": buffers[index]["arguments"],
                            },
                        }
                        for index in sorted(buffers)
                    ],
                }
            )

            for index in sorted(buffers):
                slot = buffers[index]
                yield ("tool_call", slot["name"])
                result = executor.dispatch(slot["name"], slot["arguments"])
                messages.append(
                    {"role": "tool", "tool_call_id": slot["id"], "content": result}
                )
                _, summary = summarize_result(result)
                yield ("tool_result", summary)

            if turn == self._max_turns:
                break

        yield ("done", MAX_TURNS_REPLY)

    # -- 内部 ----------------------------------------------------------------

    def _create_completion(self, messages: list[dict[str, Any]], stream: bool) -> Any:
        """统一的一处 SDK 调用，方便讲解时指出"参数就这几个"。

        ``tool_choice="auto"`` 让模型自己决定用不用工具；
        ``temperature=0.2`` 是因为工具参数需要稳定（温度高时格式错误率明显上升）。
        """
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            stream=stream,
        )


def _assistant_message_dict(assistant_message: Any) -> dict[str, Any]:
    """把 SDK 的 assistant 消息转成"可以回灌给接口"的 dict。

    不直接用 ``model_dump()``：Mock 对象没有这个方法，而且我们只需要这三个字段，
    手工构造对真/假客户端都成立，讲解时也一眼能看清协议要求的最小结构。
    """
    tool_calls = []
    for tool_call in list(_get(assistant_message, "tool_calls", None) or []):
        function = _get(tool_call, "function", None)
        tool_calls.append(
            {
                "id": _get(tool_call, "id", "") or "",
                "type": _get(tool_call, "type", "function") or "function",
                "function": {
                    "name": _get(function, "name", "") or "",
                    "arguments": _get(function, "arguments", "") or "",
                },
            }
        )
    return {
        "role": "assistant",
        "content": _get(assistant_message, "content", None),
        "tool_calls": tool_calls,
    }


def build_agent(
    repo: TodoRepository,
    client: Any | None = None,
    max_turns: int | None = None,
    model: str | None = None,
) -> TodoAgent:
    """工厂函数：把配置、客户端、仓库组装成 agent。

    ``client`` 传 None 时走真实客户端（缺 Key 会抛 ``LLMConfigError``）。
    这样 API 层和 CLI 层都不需要知道客户端怎么建。
    """
    settings = get_settings()
    resolved_client = client if client is not None else build_client(settings)
    return TodoAgent(
        client=resolved_client,
        repo=repo,
        max_turns=max_turns if max_turns is not None else settings.max_turns,
        model=model or settings.model,
    )


def dump_script(script: list[ScriptedTurn]) -> str:
    """把剧本打印成人类可读的文本，demo 里用来先亮出"模型会说什么"。"""
    lines = []
    for number, turn in enumerate(script, start=1):
        if turn.tool_calls:
            calls = "; ".join(
                f"{call.name}({call.arguments})" for call in turn.tool_calls
            )
            lines.append(f"  第{number}轮 -> 调用工具: {calls}")
        else:
            lines.append(f"  第{number}轮 -> 最终回复: {turn.content}")
    return "\n".join(lines)


__all__ = [
    "EMPTY_REPLY_FALLBACK",
    "MAX_TURNS_REPLY",
    "SYSTEM_PROMPT",
    "TodoAgent",
    "build_agent",
    "dump_script",
]
