"""全部数据契约（Pydantic v2 模型）。

说明两件事，都是"技术点"而非"风格"问题：

1. 为什么用 Pydantic v2 的 ``@field_validator`` + ``@classmethod``：
   v1 的 ``@validator`` 在 v2 里已废弃，混用会出现 "PydanticDeprecatedSince20" 警告，
   而且 v1 写法在 v2 下对 Optional 字段的行为不一样（v1 里 validator 默认不处理缺省值）。
   这里统一用 v2 写法，校验顺序是：类型解析(含 strip) -> @field_validator -> 字段约束。

2. 为什么工具参数模型（``AddTodoArgs`` 等）单独存在、并且字段都写 description：
   它们的 ``model_json_schema()`` 会被直接塞进 OpenAI function calling 的
   ``parameters`` 字段。description 不是写给人看的注释，而是**写给模型看的提示词**——
   模型靠它决定"这个字段填什么、能不能不填"。所以这里的 description 都写得像指令。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 时间统一用这一个格式：既能被人读懂，也能按字符串直接排序（SQLite 没有 datetime 类型）。
DUE_AT_FORMAT: str = "%Y-%m-%d %H:%M"
DUE_AT_ERROR_MESSAGE: str = "时间格式应为 YYYY-MM-DD HH:MM（例如 2026-10-21 09:30）"
TITLE_ERROR_MESSAGE: str = "title 不能为空，请用一句话写清要做的事"

TodoStatus = Literal["pending", "done"]


def normalize_title(value: str) -> str:
    """标题去空格后不能为空。

    strip 已由 ``str_strip_whitespace=True`` 做过，这里做业务校验并返回干净值。
    抽成模块级函数是为了让「待办模型」和「工具参数模型」共用同一份规则，
    避免两处规则漂移导致"工具能存、API 存不进去"。
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(TITLE_ERROR_MESSAGE)
    return cleaned


def normalize_due_at(value: str | None) -> str | None:
    """校验并规范化截止时间格式。

    为什么不直接信模型：模型非常喜欢输出 ``10月25日``、``2026/10/25 19:00``、
    ``2026-10-25T19:00:00`` 这几种变体。放行任何一种，后面按字符串排序和展示都会乱，
    所以在入口处就钉死一种格式，错了就报错让它重填。
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        # 模型偶尔用空字符串表达"没有时间"，等价于不填，不算错误。
        return None
    try:
        parsed = datetime.strptime(text, DUE_AT_FORMAT)
    except ValueError as exc:
        # ValueError 的文本会被 Pydantic 包进 msg（前缀 "Value error, "），
        # tools.py 会剥掉该前缀再回传给模型，所以这里只写人话。
        raise ValueError(DUE_AT_ERROR_MESSAGE) from exc
    return parsed.strftime(DUE_AT_FORMAT)


class AddTodoArgs(BaseModel):
    """``add_todo`` 工具的参数模型。"""

    # extra="forbid" 很关键：模型经常自作多情多传字段（due_date / priority / id ...）。
    # 静默忽略会让"我明明说了时间、它却记成没时间"这类 bug 极难排查；
    # 直接拒绝并把"不认识的字段"回传给模型，它下一轮就会改对。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(
        description=(
            "待办事项的内容，一句话，必填，不能是空白字符串。"
            "例如「交操作系统实验报告」。不要把编号或时间写进标题。"
        ),
    )
    due_at: str | None = Field(
        default=None,
        description=(
            "截止时间，可选。格式必须是 YYYY-MM-DD HH:MM（24 小时制）。"
            "用户没有明确给出时间就不要传这个字段，不要自己猜。"
        ),
    )

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        return normalize_title(value)

    @field_validator("due_at")
    @classmethod
    def _validate_due_at(cls, value: str | None) -> str | None:
        return normalize_due_at(value)


class ListTodosArgs(BaseModel):
    """``list_todos`` 工具的参数模型。字段均可选。"""

    model_config = ConfigDict(extra="forbid")

    status: TodoStatus | None = Field(
        default=None,
        description=(
            "只列出某个状态的待办：pending=未完成，done=已完成。"
            "不传则两种都返回。用户问「还有哪些没做」时传 pending。"
        ),
    )


class CompleteTodoArgs(BaseModel):
    """``complete_todo`` 工具的参数模型。"""

    model_config = ConfigDict(extra="forbid")

    todo_id: int = Field(
        ge=1,
        description=(
            "要标记为完成的待办 id（整数）。id 必须来自 list_todos 的真实返回结果，"
            "不允许自己编号或猜。"
        ),
    )


class Todo(BaseModel):
    """数据库里一条待办的对外表示。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    title: str
    due_at: str | None = None
    status: TodoStatus = "pending"
    created_at: str
    done_at: str | None = None


class TodoCreate(BaseModel):
    """新增待办的入参（供 API/脚本复用；工具层用的是 ``AddTodoArgs``）。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(description="待办内容，必填")
    due_at: str | None = Field(default=None, description="截止时间，格式 YYYY-MM-DD HH:MM")
    status: TodoStatus = Field(default="pending", description="状态：pending 或 done")

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        return normalize_title(value)

    @field_validator("due_at")
    @classmethod
    def _validate_due_at(cls, value: str | None) -> str | None:
        return normalize_due_at(value)


class ToolCallRecord(BaseModel):
    """一次工具调用的审计记录，会随 ``ChatResponse`` 一起返回给前端。"""

    name: str = Field(description="工具名")
    arguments: str = Field(description="模型给出的参数原文（JSON 字符串，未加工）")
    ok: bool = Field(default=True, description="这次调用是否成功")
    result: str = Field(default="", description="工具返回内容的摘要，便于展示")


class ChatRequest(BaseModel):
    """``POST /chat`` 的请求体。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    # 上限 2000 字符（不是 2000 字节）：这个接口的输入会被原样塞进 messages，
    # 再随每一轮工具往返重复发送。不设上限的话，10 万字的输入会一次性把上下文窗口、
    # 单次请求的 token 成本和响应延迟全部打满，而且这种请求一定是误用而不是真实需求。
    # 业务上单条待办的描述远用不到 2000 字，所以这个限制不会挡住正常用法。
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="用户这句话（1~2000 字符）",
    )
    user_id: str = Field(default="default", description="用户标识，用于隔离各自的待办")

    @field_validator("user_id")
    @classmethod
    def _validate_user_id(cls, value: str) -> str:
        cleaned = value.strip()
        return cleaned or "default"


class ChatResponse(BaseModel):
    """``POST /chat`` 的响应体。"""

    reply: str = Field(description="模型给出的最终自然语言回复")
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="本次对话中的工具调用记录"
    )
    turns: int = Field(default=0, description="实际消耗的模型轮数（用于暴露 max_turns 兜底）")


class ChatOutcome(BaseModel):
    """agent 内部返回值。和 ChatResponse 分开，是为了让内部结构不绑死 HTTP 契约。"""

    reply: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    turns: int = 0
