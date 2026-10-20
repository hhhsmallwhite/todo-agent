"""FastAPI 应用。

三个刻意的设计点（都是面试/答辩会被问的）：

1. **路由用同步 ``def``，不用 ``async def``。**
   agent 内部是用 OpenAI **同步** SDK 的（阻塞 IO）。如果路由写成 ``async def``，
   这一次网络请求会把事件循环整条堵住，其他请求全部排队——这就是典型的"假异步"。
   写成 ``def``，FastAPI 会自动把函数丢到线程池里跑，事件循环不被阻塞，
   这也是官方文档明确推荐的用法（"如果函数内部是阻塞调用，就用 def"）。

2. **``Depends`` 提供 repository。**
   即使只有一个依赖也要走注入：这是分层与可替换性的落地点。
   测试里用 ``app.dependency_overrides[get_repository]`` 就能把真实 SQLite 换成临时库，
   业务代码一行都不用改。

3. **绝不把 500 泄漏出去。**
   模型侧异常（没 Key、超时、4xx）和工具侧异常都被兜住，转成可读的中文 502/503。
   工具层的 ``dispatch`` 本身已经返回结构化错误，所以这里主要兜"模型调用"这一段。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from .agent import TodoAgent, build_agent
from .config import get_settings
from .llm import LLMConfigError
from .models import ChatRequest, ChatResponse, Todo
from .storage import TodoRepository, get_default_repository

LOGGER = logging.getLogger(__name__)

TodoStatusQuery = Literal["pending", "done"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时准备数据库。

    把仓库挂到 ``app.state`` 上而不是模块全局，是为了让应用实例和资源一一对应：
    测试里 create 一个新 app 就是一份独立资源，不会互相污染。
    """
    repository = TodoRepository(get_settings().db_path)
    repository.init_schema()
    app.state.repo = repository
    LOGGER.info("数据库已就绪：%s", repository.db_path)
    yield


app = FastAPI(
    title="Todo Agent API",
    version="0.1.0",
    description=(
        "基于大模型工具调用的待办事项智能体。"
        "文档由 FastAPI 自动生成：/docs（Swagger UI）、/openapi.json（OpenAPI 规范）。"
    ),
    lifespan=lifespan,
)


def get_repository(request: Request) -> TodoRepository:
    """依赖注入：提供 TodoRepository。

    优先用 lifespan 里准备好、挂在 app.state 上的实例；
    直接 ``TestClient(app)`` 而不走 lifespan 的极端情况下退回进程级默认仓库，
    保证 /todos 这类纯读接口不会因为依赖没初始化而 500。
    """
    repository = getattr(request.app.state, "repo", None)
    if repository is None:
        repository = get_default_repository()
        request.app.state.repo = repository
    return repository


RepositoryDep = Annotated[TodoRepository, Depends(get_repository)]


def _resolve_agent(repo: TodoRepository) -> TodoAgent:
    """每次请求组装一个 agent。

    为什么不缓存：agent 是无状态的一次性执行器，缓存反而要处理"配置热更新"、
    "客户端连接复用"这些额外问题。真正的连接复用由 openai SDK 内部的 httpx 连接池负责。
    """
    settings = get_settings()
    return build_agent(repo=repo, max_turns=settings.max_turns, model=settings.model)


@app.post("/chat", response_model=ChatResponse, summary="和待办助手对话")
def chat(payload: ChatRequest, repo: RepositoryDep) -> ChatResponse:
    """把用户这句话交给 agent 跑完整工具调用循环，返回最终回复 + 工具调用记录。

    路由体是同步的（见模块文档第 1 点）。异常全部转成 4xx/5xx 的 detail 字符串，
    不把 Python 堆栈返回给调用方。
    """
    try:
        agent = _resolve_agent(repo)
        outcome = agent.chat(payload.message, user_id=payload.user_id)
    except LLMConfigError as exc:
        # 配置问题属于"服务端没准备好"，用 503 而不是 500：调用方知道该去配 Key。
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 这里就是"不让 500 泄漏堆栈"的那道闸
        LOGGER.exception("调用模型失败")
        raise HTTPException(
            status_code=502,
            detail=f"调用模型失败（{exc.__class__.__name__}）。请检查 API_KEY / BASE_URL / 网络。",
        ) from exc
    return ChatResponse(
        reply=outcome.reply,
        tool_calls=outcome.tool_calls,
        turns=outcome.turns,
    )


@app.get("/todos", response_model=list[Todo], summary="查询待办列表")
def list_todos(
    repo: RepositoryDep,
    status: Annotated[TodoStatusQuery | None, Query(description="pending 或 done，不传则全部")] = None,
    user_id: Annotated[str, Query(description="用户标识")] = "default",
) -> list[Todo]:
    """直接读库，不经过大模型。

    这条接口的存在是为了讲清分层：查询这种确定性操作没必要烧 token，
    真正需要模型的只有"把自然语言翻译成结构化操作"。
    """
    return repo.list(status=status, user_id=user_id)
