"""命令行入口的冒烟测试。

CLI 是面试现场最容易"现场跑一遍"的入口，所以至少要保证：
* ``--mock`` 离线模式能跑完并返回 0（不联网、不读 stdin）；
* 没有 API_KEY 时不是抛堆栈，而是打印一句人话并返回退出码 2；
* ``--db`` 能覆盖数据库路径（否则会在项目根目录乱建文件）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import _print_events, main


def test_mock_mode_runs_offline_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    db_file = tmp_path / "cli.db"

    code = main(["--mock", "--db", str(db_file)])

    output = capsys.readouterr().out
    assert code == 0
    assert "离线模式" in output
    assert "轮数：" in output
    assert db_file.exists(), "--db 指定的库要被真正创建"


def test_interactive_mode_without_api_key_exits_gracefully(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """没有 Key 时不能进 REPL 等输入，更不能甩堆栈：直接提示 + 退出码 2。"""
    code = main(["--db", str(tmp_path / "cli2.db")])

    output = capsys.readouterr().out
    assert code == 2
    assert "配置缺失" in output
    assert "API_KEY" in output


def test_print_events_collects_final_text(capsys: pytest.CaptureFixture) -> None:
    events = [
        ("tool_call", "add_todo"),
        ("tool_result", "已创建待办 #1"),
        ("token", "已经"),
        ("token", "记好了"),
        ("done", "已经记好了"),
    ]

    final_text = _print_events(iter(events))

    output = capsys.readouterr().out
    assert final_text == "已经记好了"
    assert "[工具调用] add_todo" in output
    assert "[工具返回] 已创建待办 #1" in output
    assert "已经记好了" in output
