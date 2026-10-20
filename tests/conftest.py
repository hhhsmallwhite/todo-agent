"""测试公共设施（fixture 都在这里，用例文件里不再重复造）。

三条隔离原则，缺一条就会互相污染：

1. **绝不写项目根目录的 ``todos.db``。**
   autouse fixture 把 ``DB_PATH`` 指到 pytest 的 ``tmp_path``，并清掉 ``API_KEY``。
   这样测试既不碰真实数据，也不会因为本机配了 Key 而真的跑去请求网络。
2. **配置缓存必须清。** ``app.config.get_settings`` 是 ``lru_cache`` 单例，
   改了环境变量不 ``cache_clear()``，下一个用例读到的还是上一个用例的路径。
3. **一个用例一个 sqlite 文件。** ``tmp_path`` 每个用例都不同，
   跑完由 pytest 自己回收，不需要手写清理逻辑。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from app.agent import TodoAgent
from app.config import get_settings
from app.llm import MockLLMClient
from app.storage import TodoRepository
from app.tools import ToolExecutor


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把进程级配置钉死在一个临时库里，并保证没有 API_KEY。

    autouse=True 的理由：任何一条用例（包括将来新加的）只要不小心走到
    ``get_default_repository()``，也不会在项目根目录拉出一个 todos.db。
    """
    db_path = tmp_path / "isolated.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        yield db_path
    finally:
        # 先把缓存清掉，避免下一个用例拿到这个已被删除的临时路径。
        get_settings.cache_clear()


@pytest.fixture
def repo(tmp_path: Path) -> TodoRepository:
    """一个建好表结构的空仓库。"""
    repository = TodoRepository(tmp_path / "todos.db")
    repository.init_schema()
    return repository


@pytest.fixture
def executor(repo: TodoRepository) -> ToolExecutor:
    """绑定在 ``repo`` 上的工具执行器（user_id 用默认值）。"""
    return ToolExecutor(repo)


@pytest.fixture
def make_agent(repo: TodoRepository) -> Callable[..., tuple[TodoAgent, MockLLMClient]]:
    """工厂 fixture：按剧本造一个完全离线的 agent。

    返回 ``(agent, client)``；``client`` 保留了每次请求的 messages 快照，
    用例靠它断言"送给模型的上下文是否符合 OpenAI 协议"。
    """

    def _make(
        script: list[Any],
        max_turns: int = 8,
    ) -> tuple[TodoAgent, MockLLMClient]:
        client = MockLLMClient(script)
        agent = TodoAgent(client=client, repo=repo, max_turns=max_turns, model="mock-model")
        return agent, client

    return _make


@pytest.fixture
def api_repo() -> TodoRepository:
    """API 用例用的仓库（路径就是 autouse fixture 指定的临时库）。"""
    repository = TodoRepository(get_settings().db_path)
    repository.init_schema()
    return repository


@pytest.fixture
def api_client(api_repo: TodoRepository) -> Iterator[Any]:
    """带依赖覆盖的 TestClient。

    覆盖 ``get_repository`` 而不是靠全局单例，是 FastAPI 官方推荐的测试方式：
    业务代码一行不改，把数据库换成临时文件。
    """
    from fastapi.testclient import TestClient

    from app import api

    api.app.dependency_overrides[api.get_repository] = lambda: api_repo
    try:
        # with 语句会触发 lifespan（同时验证了启动钩子没写坏）。
        with TestClient(api.app) as client:
            yield client
    finally:
        api.app.dependency_overrides.clear()
