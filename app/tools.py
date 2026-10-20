"""工具层：给模型看的 JSON Schema + 真正执行的分发器。

这是整个项目最值得逐行讲的文件，因为它包含三件"简历上很容易吹、实际很容易错"的事：

1. **``parameters`` 是 JSON Schema，不是 Python 签名。**
   模型只能看到这段 JSON：``required`` 告诉它哪些必填，``enum`` 告诉它取值只能是这几个。
   不写 ``enum``，模型就可能给 ``status`` 填 ``"all"``、``"completed"`` 之类，然后在执行层炸掉。

2. **``arguments`` 是"JSON 字符串"，不是 dict。**
   OpenAI 协议里 ``tool_call.function.arguments`` 的类型是 string。所以这里第一件事是
   ``json.loads``，并且必须 try/except——模型会输出截断的、带尾逗号的、甚至空字符串的 JSON。

3. **参数校验失败要"回传给模型"，而不是抛异常。**
   抛异常只会让这一轮崩掉，模型永远不知道自己错在哪。正确做法是把 Pydantic 的
   ``ValidationError`` 压缩成一两个字段的短 JSON 塞回 ``role="tool"`` 消息，
   模型看到 ``{"ok": false, "detail": [{"field": "due_at", "msg": "..."}]}`` 后下一轮就会改对。
   注意**不要直接 ``str(e)``**：那玩意带文档 URL、输入值和换行，几百个 token 全是噪音。

安全设计：这里只有 add/list/complete 三个工具，**没有** delete、没有批量删除、
没有"按条件删除"。原因见 README「安全设计」一节——工具集合本身就是权限边界。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from .models import AddTodoArgs, CompleteTodoArgs, ListTodosArgs, Todo
from .storage import TodoRepository, get_default_repository

LOGGER = logging.getLogger(__name__)

# 工具名常量，避免各处手写字符串写错（拼错一个字母就是 KeyError 级别的 bug）。
TOOL_ADD_TODO: str = "add_todo"
TOOL_LIST_TODOS: str = "list_todos"
TOOL_COMPLETE_TODO: str = "complete_todo"

TOOL_NAMES: tuple[str, ...] = (TOOL_ADD_TODO, TOOL_LIST_TODOS, TOOL_COMPLETE_TODO)

# ---------------------------------------------------------------------------
# 给模型看的工具定义。字段名 / description / required / enum 都会进 prompt，
# 所以这里的中文描述是"提示词"，不是注释。
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_ADD_TODO,
            "description": "新增一条待办事项。用户说「记一下 / 提醒我 / 帮我加个任务」时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "待办内容，一句话，不允许空白。例如「交操作系统实验报告」。"
                            "不要把编号或时间写进标题。"
                        ),
                    },
                    "due_at": {
                        "type": "string",
                        "description": (
                            "截止时间，可选。格式必须严格是 YYYY-MM-DD HH:MM（24 小时制），"
                            "例如 2026-10-21 09:30。用户没说时间就不要传，不要自己猜。"
                        ),
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_LIST_TODOS,
            "description": (
                "查询待办列表。用户问「我还有什么事 / 做完了哪些」时必须先调用它。"
                "返回的 id 是后续 complete_todo 唯一合法的输入来源。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        # enum 是硬约束：不写的话模型爱传 "all"、"completed"、"未完成"。
                        "enum": ["pending", "done"],
                        "description": (
                            "pending=只看未完成，done=只看已完成。不传则两者都返回。"
                        ),
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_COMPLETE_TODO,
            "description": (
                "把一条待办标记为已完成。必须使用 list_todos 返回的真实 id，"
                "不允许凭猜测或自增编号填写。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {
                        "type": "integer",
                        "description": "待办 id（整数，≥1），只能来自 list_todos 的返回结果。",
                    }
                },
                "required": ["todo_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _ok(data: Any) -> str:
    """成功统一返回 {"ok": true, "data": ...}。结构固定，模型和测试都好处理。"""
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _error(code: str, detail: list[dict[str, str]]) -> str:
    """失败统一返回 {"ok": false, "error": ..., "detail": [...]}。

    detail 永远是「短字段名 + 一句人话」，因为这段文本会原样进模型上下文，
    多一个换行都是浪费 token，还会稀释真正的错误信息。
    """
    return json.dumps({"ok": False, "error": code, "detail": detail}, ensure_ascii=False)


# Pydantic v2 的错误类型 -> 中文人话。只覆盖这个项目真会碰到的几种，
# 其余情况退回原始 msg（英文也比丢掉信息强）。
_ERROR_TYPE_MESSAGES: dict[str, str] = {
    "missing": "缺少必填字段",
    "extra_forbidden": "出现了不认识的字段，请只使用工具定义里列出的字段",
    "string_type": "应为字符串",
    "int_type": "应为整数",
    "int_parsing": "应为整数",
    "greater_than_equal": "数值必须大于等于 1",
    "literal_error": "取值不合法，只能是定义里列出的枚举值",
    "model_attributes_type": "应为 JSON 对象",
}


def _humanize(error: dict[str, Any]) -> str:
    """把一条 Pydantic 错误压成一句中文。

    v2 会把 ``raise ValueError("xxx")`` 的文本包成 ``"Value error, xxx"``，
    这里把前缀剥掉，只留我们自己写的人话。
    """
    raw_message = str(error.get("msg", "")).strip()
    value_error_prefix = "Value error, "
    if raw_message.startswith(value_error_prefix):
        return raw_message[len(value_error_prefix) :]
    error_type = str(error.get("type", ""))
    mapped = _ERROR_TYPE_MESSAGES.get(error_type)
    if mapped is None:
        return raw_message or "参数不合法"
    if error_type in ("missing", "extra_forbidden", "int_parsing", "greater_than_equal"):
        # 这几种映射本身已经解释清楚，不再拼接英文原文。
        return mapped
    return f"{mapped}（{raw_message}）" if raw_message else mapped


def format_validation_error(exc: ValidationError) -> list[dict[str, str]]:
    """``ValidationError`` -> ``[{"field": ..., "msg": ...}]``。

    只取前 3 条：参数模型都很小，多出来的错误基本是同一个原因，
    全塞回模型反而干扰它定位。
    """
    details: list[dict[str, str]] = []
    for error in exc.errors()[:3]:
        location = error.get("loc") or ("<root>",)
        field = ".".join(str(part) for part in location) if location else "<root>"
        details.append({"field": field, "msg": _humanize(error)})
    return details or [{"field": "<root>", "msg": "参数不合法"}]


def _empty_model_error(model_name: str) -> list[dict[str, str]]:
    return [{"field": "<root>", "msg": f"{model_name} 必须是 JSON 对象，例如 {{\"title\": \"买牛奶\"}}"}]


def summarize_result(raw: str, limit: int = 300) -> tuple[bool, str]:
    """从工具返回值里抽出 ``(ok, 摘要)``，供 ChatResponse 的审计记录使用。

    这里再 json.loads 一次是刻意的：dispatch 的返回值是"给模型看的字符串"，
    而审计记录是"给人看的文本"，两者用途不同，不能共用同一个结构。
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, raw[:limit]
    ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
    if ok:
        data = payload.get("data")
        if isinstance(data, dict) and "message" in data:
            text = str(data["message"])
        else:
            text = json.dumps(data, ensure_ascii=False)
    else:
        detail = payload.get("detail") if isinstance(payload, dict) else raw
        text = json.dumps(
            {"error": payload.get("error") if isinstance(payload, dict) else "unknown", "detail": detail},
            ensure_ascii=False,
        )
    return ok, text[:limit]


class ToolExecutor:
    """把"工具名 + 参数字符串"执行成"结果字符串"。

    绑定 ``repo`` 和 ``user_id`` 的原因：工具本身不应该从全局拿数据，
    否则多用户隔离就退化成"谁都能改别人的待办"。
    """

    def __init__(self, repo: TodoRepository, user_id: str = "default") -> None:
        self._repo = repo
        # 和 dispatch 里 arguments 那道闸同一类问题：协议上 user_id 是 str，但把 app 当库用时
        # （或者哪天 HTTP 层被换掉）可能直接传 None / 数字。直接 .strip() 会抛
        # AttributeError: 'NoneType' object has no attribute 'strip'，而这个异常会从
        # agent.chat() 冒到 API 层被兜成 502——虽然不致命，但拿不到结构化信息。
        # 这里统一归一化成字符串：None/空串 -> "default"，其余按 str() 收下再 strip。
        raw_user_id = "" if user_id is None else str(user_id)
        self._user_id = raw_user_id.strip() or "default"

    @property
    def user_id(self) -> str:
        return self._user_id

    def dispatch(self, name: str, arguments: str) -> str:
        """执行一次工具调用。

        :param name: 工具名（来自模型）。
        :param arguments: **JSON 字符串**（来自 ``tool_call.function.arguments``）。
        :return: 一定是合法 JSON 的字符串——成功 ``{"ok": true, ...}``，
                 失败 ``{"ok": false, "error": ..., "detail": [...]}``。永不抛异常。
        """
        # 类型判断必须在 .strip() 之前：协议规定 arguments 是字符串，但上游（或手写调用、
        # 或某些网关）可能已经把 JSON 解析成 dict 传进来。没有这道闸，
        # 下面 `.strip()` 会抛 AttributeError: 'dict' object has no attribute 'strip'，
        # 那就违背了"永不抛异常"的承诺——异常一旦冒到上层，/chat 就只能 500。
        # None 例外：它等价于"没有参数"，按空字符串处理，走下面 "{}" 的分支。
        if arguments is None:
            arguments = ""
        elif not isinstance(arguments, str):
            return _error(
                "bad_json",
                [
                    {
                        "field": "arguments",
                        "msg": (
                            "arguments 必须是 JSON 字符串（不是已解析好的对象），"
                            '例如 {"title": "买牛奶"} 序列化后的字符串'
                        ),
                    }
                ],
            )

        raw_arguments = arguments.strip()
        # 无参工具（如 list_todos）模型有时会传 ""，有时传 "{}"。
        # 空字符串直接当 "{}" 处理，否则 json.loads 会在这一行炸掉。
        if not raw_arguments:
            raw_arguments = "{}"

        try:
            payload = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            # 常见原因：模型把 JSON 截断了（max_tokens 不够）、多写了尾逗号、
            # 或者把参数包成了字符串。把原始片段也回传，模型才有依据修正。
            return _error(
                "bad_json",
                [
                    {
                        "field": "arguments",
                        "msg": (
                            f"arguments 不是合法 JSON（{exc.msg}，位置 {exc.pos}）。"
                            "请传一个 JSON 对象字符串，例如 {\"title\": \"买牛奶\"}"
                        ),
                    }
                ],
            )

        if not isinstance(payload, dict):
            return _error("bad_json", _empty_model_error("arguments"))

        try:
            if name == TOOL_ADD_TODO:
                return self._add_todo(payload)
            if name == TOOL_LIST_TODOS:
                return self._list_todos(payload)
            if name == TOOL_COMPLETE_TODO:
                return self._complete_todo(payload)
            return _error(
                "unknown_tool",
                [
                    {
                        "field": "name",
                        "msg": f"没有名为 {name!r} 的工具，可用工具只有：{', '.join(TOOL_NAMES)}",
                    }
                ],
            )
        except Exception:  # noqa: BLE001 - 这一层是"绝不让异常冒到 HTTP 层"的最后一道闸
            # 数据库被删、磁盘满了、编码错误……都会落到这里。日志留全栈，模型只看到一句人话。
            LOGGER.exception("工具 %s 执行时发生未预期异常", name)
            return _error(
                "internal_error",
                [{"field": "<root>", "msg": f"工具 {name} 执行失败，属于服务端问题，请告知用户稍后重试"}],
            )

    # -- 具体工具实现 ---------------------------------------------------------

    def _add_todo(self, payload: dict[str, Any]) -> str:
        try:
            args = AddTodoArgs.model_validate(payload)
        except ValidationError as exc:
            return _error("invalid_arguments", format_validation_error(exc))
        todo = self._repo.add(title=args.title, due_at=args.due_at, user_id=self._user_id)
        return _ok({"todo": todo.model_dump(), "message": _describe_created(todo)})

    def _list_todos(self, payload: dict[str, Any]) -> str:
        try:
            args = ListTodosArgs.model_validate(payload)
        except ValidationError as exc:
            return _error("invalid_arguments", format_validation_error(exc))
        todos = self._repo.list(status=args.status, user_id=self._user_id)
        return _ok(
            {
                "count": len(todos),
                "status_filter": args.status,
                "todos": [todo.model_dump() for todo in todos],
                "message": _describe_list(todos, args.status),
            }
        )

    def _complete_todo(self, payload: dict[str, Any]) -> str:
        try:
            args = CompleteTodoArgs.model_validate(payload)
        except ValidationError as exc:
            return _error("invalid_arguments", format_validation_error(exc))
        todo = self._repo.complete(args.todo_id, user_id=self._user_id)
        if todo is None:
            # 不抛异常：这属于"模型的输入错了"，要让它自己回去查 list_todos 拿真 id。
            return _error(
                "todo_not_found",
                [
                    {
                        "field": "todo_id",
                        "msg": f"没有找到 id={args.todo_id} 的待办，请先调用 list_todos 取得真实 id 再试",
                    }
                ],
            )
        return _ok({"todo": todo.model_dump(), "message": f"已把 #{todo.id}「{todo.title}」标记为完成"})


def _describe_created(todo: Todo) -> str:
    if todo.due_at:
        return f"已创建待办 #{todo.id}：{todo.title}（截止 {todo.due_at}）"
    return f"已创建待办 #{todo.id}：{todo.title}（没有截止时间）"


def _describe_list(todos: list[Todo], status: str | None) -> str:
    if not todos:
        scope = {"pending": "未完成", "done": "已完成"}.get(status or "", "全部")
        return f"当前{scope}待办为空"
    lines = []
    for todo in todos:
        mark = "已完成" if todo.status == "done" else "未完成"
        due = f"，截止 {todo.due_at}" if todo.due_at else ""
        lines.append(f"#{todo.id} {todo.title}（{mark}{due}）")
    return "；".join(lines)


def dispatch(
    name: str,
    arguments: str,
    repo: TodoRepository | None = None,
    user_id: str = "default",
) -> str:
    """模块级便捷函数：不想持有执行器对象时用它。

    真正的逻辑在 :class:`ToolExecutor` 里，这里只是给一个默认仓库的糖衣。
    """
    executor = ToolExecutor(repo if repo is not None else get_default_repository(), user_id=user_id)
    return executor.dispatch(name, arguments)


def tool_names() -> tuple[str, ...]:
    """返回可用工具名，供 README/测试断言"没有删除类工具"这一安全设计。"""
    return TOOL_NAMES


__all__ = [
    "TOOLS",
    "TOOL_ADD_TODO",
    "TOOL_COMPLETE_TODO",
    "TOOL_LIST_TODOS",
    "TOOL_NAMES",
    "ToolExecutor",
    "dispatch",
    "format_validation_error",
    "summarize_result",
    "tool_names",
]
