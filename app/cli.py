"""命令行入口。

用法：
    python -m app.cli            # 交互模式，需要 .env 里配好 API_KEY
    python -m app.cli --mock     # 离线模式，用 MockLLMClient 的剧本跑一遍，不需要 Key

CLI 的意义不是"能用就行"，而是**让工具调用循环可见**：
每轮打印「模型选了什么工具、参数原文、返回摘要」，老师一眼就能看到
"模型不是直接回答，而是先调工具、拿到结果再回答"这件事真的发生了。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable

from .agent import TodoAgent, build_agent
from .config import get_settings
from .llm import LLMConfigError, MockLLMClient, script_full_demo
from .storage import TodoRepository

BANNER: str = """=== 待办事项智能体（工具调用版） ===
输入一句话管理待办，例如：
  明天九点半要交操作系统实验报告
  我还有哪些没做
  把第二个标记完成
输入 quit 退出。
"""


def _prepare_stdout() -> None:
    """Windows 控制台默认是 GBK，直接打中文偶发 UnicodeEncodeError。

    统一改成 UTF-8 + replace，宁可出现替换字符也不要因为一个表情把进程打崩。
    """
    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _print_events(events: Iterable[tuple[str, str]]) -> str:
    """消费 chat_stream 的事件流并打印，返回最终文本。"""
    final_text = ""
    for kind, payload in events:
        if kind == "token":
            print(payload, end="", flush=True)
        elif kind == "tool_call":
            print(f"\n  [工具调用] {payload}", flush=True)
        elif kind == "tool_result":
            print(f"  [工具返回] {payload}", flush=True)
        elif kind == "done":
            final_text = payload
    print()
    return final_text


def _run_mock(repo: TodoRepository) -> int:
    """离线演示：跑一遍剧本，把每一轮的决策都打印出来。"""
    print("=== 离线模式（MockLLMClient，不联网）===")
    client = MockLLMClient(script_full_demo())
    agent = TodoAgent(client, repo, max_turns=8, model="mock-model")
    message = "帮我记一下这周的事，然后告诉我还有哪些没做"
    print(f"用户：{message}")
    outcome = agent.chat(message, user_id="cli-demo")
    for index, record in enumerate(outcome.tool_calls, start=1):
        flag = "成功" if record.ok else "失败"
        print(f"  {index}. {record.name}({record.arguments}) -> {flag}：{record.result}")
    print(f"轮数：{outcome.turns}")
    print(f"助手：{outcome.reply}")
    return 0


def _run_interactive(repo: TodoRepository) -> int:
    settings = get_settings()
    try:
        agent = build_agent(repo=repo, max_turns=settings.max_turns)
    except LLMConfigError as exc:
        print(f"[配置缺失] {exc}")
        return 2

    print(BANNER)
    while True:
        try:
            raw = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("再见。")
            return 0
        try:
            final_text = _print_events(agent.chat_stream(raw, user_id="cli-user"))
        except Exception as exc:  # noqa: BLE001 - 交互式工具不该因为一次网络错误退出
            print(f"\n[出错] {exc.__class__.__name__}: {exc}")
            continue
        if final_text:
            print(f"助手：{final_text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="待办事项智能体命令行入口")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="离线模式：用 MockLLMClient 剧本跑一遍，不需要 API_KEY",
    )
    parser.add_argument("--db", default=None, help="覆盖数据库路径（默认取配置里的 DB_PATH）")
    parser.add_argument("--verbose", action="store_true", help="打印 INFO 级别日志")
    args = parser.parse_args(argv)

    _prepare_stdout()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    db_path = args.db or settings.db_path
    repo = TodoRepository(db_path)
    repo.init_schema()
    print(f"[db] {repo.db_path}")

    return _run_mock(repo) if args.mock else _run_interactive(repo)


if __name__ == "__main__":
    raise SystemExit(main())
