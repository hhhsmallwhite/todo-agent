"""配置与模型客户端层的测试。

这两层的价值在"换模型/换 Key"这类问题上：
* 没有 Key 时必须抛 ``LLMConfigError`` 并给出**可照做**的中文提示（API 层据此回 503）；
* ``BASE_URL`` 必须真的传给 SDK（国内 OpenAI 兼容网关全靠它）；
* 环境变量写错格式时不能把进程搞挂，要退回默认值。
"""

from __future__ import annotations

from pathlib import Path

import openai
import pytest

from app.config import (
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    Settings,
    get_settings,
    load_settings,
)
from app.llm import LLMConfigError, build_client


def make_settings(api_key: str = "", base_url: str = "", **overrides: object) -> Settings:
    payload: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url,
        "model": DEFAULT_MODEL,
        "db_path": Path("unused.db"),
        "max_turns": DEFAULT_MAX_TURNS,
    }
    payload.update(overrides)
    return Settings(**payload)  # type: ignore[arg-type]


class TestBuildClient:
    def test_missing_api_key_raises_actionable_chinese_error(self) -> None:
        with pytest.raises(LLMConfigError) as excinfo:
            build_client(make_settings(api_key="   "))

        message = str(excinfo.value)
        assert "API_KEY" in message
        assert ".env" in message
        assert "--mock" in message, "要告诉用户还有离线模式这条退路"

    def test_base_url_is_forwarded_to_sdk(self) -> None:
        client = build_client(
            make_settings(api_key="sk-test-not-real", base_url="https://example.invalid/v1")
        )

        assert isinstance(client, openai.OpenAI)
        assert str(client.base_url).startswith("https://example.invalid/v1")

    def test_empty_base_url_falls_back_to_official_endpoint(self) -> None:
        client = build_client(make_settings(api_key="sk-test-not-real"))

        assert "api.openai.com" in str(client.base_url)


class TestSettings:
    def test_reads_env_values(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_file = tmp_path / "from-env.db"
        monkeypatch.setenv("MODEL", "  deepseek-chat  ")
        monkeypatch.setenv("MAX_TURNS", " 3 ")
        monkeypatch.setenv("DB_PATH", str(db_file))
        monkeypatch.setenv("BASE_URL", "https://api.example.com/v1")

        settings = load_settings()

        assert settings.model == "deepseek-chat"
        assert settings.max_turns == 3
        assert settings.db_path.resolve() == db_file.resolve()
        assert settings.base_url == "https://api.example.com/v1"

    def test_blank_model_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL", "   ")

        assert load_settings().model == DEFAULT_MODEL

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("abc", DEFAULT_MAX_TURNS), ("", DEFAULT_MAX_TURNS)],
    )
    def test_bad_max_turns_falls_back_instead_of_crashing(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
    ) -> None:
        monkeypatch.setenv("MAX_TURNS", raw)

        assert load_settings().max_turns == expected

    def test_max_turns_is_floored_at_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_TURNS", "0")

        assert load_settings().max_turns == 1

    def test_relative_db_path_is_anchored_to_project_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """从不同目录启动必须连到同一个库，否则会出现"昨天记的待办不见了"。"""
        monkeypatch.setenv("DB_PATH", "data/relative.db")

        assert load_settings().db_path == (PROJECT_ROOT / "data" / "relative.db").resolve()

    def test_get_settings_is_cached_singleton(self) -> None:
        assert get_settings() is get_settings()

    def test_has_api_key_ignores_whitespace(self) -> None:
        assert make_settings(api_key="   ").has_api_key is False
        assert make_settings(api_key="sk-1").has_api_key is True
