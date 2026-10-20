"""配置层：把"环境相关的值"集中到一处。

为什么要单独一个模块而不是到处 ``os.getenv``：
1. 演示/答辩现场最常被问的是"你的 Key 存在哪、换模型要改几个地方"——
   答案应该只有一句"改 .env，代码里只读 config"。
2. 测试里可以构造一个 Settings 传进去，不必污染真实进程的环境变量。
3. .env 只在导入时加载一次；用 lru_cache 缓存解析结果，避免每次请求都读盘。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = app/ 的上一层。全部路径都从这里派生，避免写死 Windows 反斜杠路径。
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# load_dotenv 默认不会覆盖已存在的环境变量，这一点很重要：
# 部署时用真实环境变量（如容器注入）应当优先于仓库里的 .env 文件。
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_DB_NAME: str = "todos.db"
DEFAULT_MAX_TURNS: int = 8


def _read_int(name: str, default: int) -> int:
    """读取整数型环境变量；写错格式时退回默认值而不是让进程起不来。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _read_path(name: str, default_name: str) -> Path:
    """读取路径型环境变量。相对路径统一按项目根目录解析。

    按"当前工作目录"解析是新手最容易踩的坑：从不同目录启动脚本会连到不同的
    sqlite 文件，于是出现"昨天记的待办不见了"。这里固定以 PROJECT_ROOT 为基准。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return PROJECT_ROOT / default_name
    candidate = Path(raw.strip())
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    """一份不可变的配置快照。frozen 是为了防止运行期被某个模块偷偷改掉。"""

    api_key: str
    base_url: str
    model: str
    db_path: Path
    max_turns: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())


def load_settings() -> Settings:
    """从环境变量（含 .env）构造配置快照。"""
    return Settings(
        api_key=os.getenv("API_KEY", "").strip(),
        # BASE_URL 留空表示走官方 https://api.openai.com/v1。
        # 国内实际多用 OpenAI 兼容服务（硅基流动 / DeepSeek 等），所以必须可配。
        base_url=os.getenv("BASE_URL", "").strip(),
        model=os.getenv("MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        db_path=_read_path("DB_PATH", DEFAULT_DB_NAME),
        max_turns=max(1, _read_int("MAX_TURNS", DEFAULT_MAX_TURNS)),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试要换配置时用 ``get_settings.cache_clear()``。"""
    return load_settings()
