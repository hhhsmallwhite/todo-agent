"""离线演示脚本（不需要 API Key、不联网）。

为什么要专门写这个脚本：
现场演示时会议室网络、模型额度、Key 有效期都不可控。用 ``MockLLMClient`` 回放一段剧本，
可以保证"模型选工具 -> 参数校验 -> 执行 -> 结果回填 -> 最终回复"这五步一定能跑完，
并且每一步都打印出来，方便对着代码讲。

Mock 的流式是真的分片（参数被切成 8 个字符一片），所以这里的流式演示同时也在验证
agent 里"按 index 累加 function.arguments"的拼接逻辑。

运行：python scripts/demo_offline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 允许直接 `python scripts/demo_offline.py` 运行：把项目根目录塞进 sys.path，
# 否则 `import app` 会因为 sys.path[0] 是 scripts/ 而失败。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import TodoAgent, dump_script  # noqa: E402 - 必须在 sys.path 调整之后导入
from app.llm import (  # noqa: E402
    MockLLMClient,
    ScriptedToolCall,
    ScriptedTurn,
    script_full_demo,
)
from app.storage import TodoRepository  # noqa: E402

DEFAULT_USER: str = "demo"


def prepare_stdout() -> None:
    """Windows 控制台按 UTF-8 输出，避免中文/符号导致的编码异常。"""
    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def fresh_db() -> Path:
    """准备一个干净的临时数据库。

    只删三个具体文件（.db / -wal / -shm），不做任何递归删除。
    """
    db_path = Path(tempfile.gettempdir()) / f"todo_agent_demo_{os.getpid()}.db"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{db_path}{suffix}")
        if candidate.exists():
            candidate.unlink()
    return db_path


def print_section(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def run_sync_demo(repo: TodoRepository) -> None:
    """第一步：非流式跑完整个循环，展示轮数、每次工具调用与最终回复。"""
    print_section("1) 非流式完整循环（MockLLMClient 回放剧本）")
    script = script_full_demo()
    print("剧本（模型每一轮打算做什么）：")
    print(dump_script(script))

    client = MockLLMClient(script, model="mock-model")
    agent = TodoAgent(client, repo, max_turns=8, model="mock-model")

    user_message = "帮我记一下这周的事，然后告诉我还有哪些没做"
    print()
    print(f"用户：{user_message}")
    outcome = agent.chat(user_message, user_id=DEFAULT_USER)

    print()
    print("工具调用轨迹：")
    for index, record in enumerate(outcome.tool_calls, start=1):
        flag = "ok" if record.ok else "FAILED"
        print(f"  [{index}] {record.name}  args={record.arguments}")
        print(f"      -> {flag}: {record.result}")

    print()
    print(f"消耗模型轮数：{outcome.turns} / 上限 {agent.max_turns}")
    print(f"助手最终回复：{outcome.reply}")

    print()
    print("请求协议自检（验证 tool_call_id 一一对应）：")
    for number, request in enumerate(client.requests, start=1):
        messages = request["messages"]
        assistant_calls = sum(len(message.get("tool_calls") or []) for message in messages)
        tool_messages = sum(1 for message in messages if message.get("role") == "tool")
        tools_sent = request["tool_count"]
        print(
            f"  第{number}次请求: 消息{len(messages)}条, 传入工具{tools_sent}个, "
            f"assistant声明的tool_call={assistant_calls}, role=tool消息={tool_messages} "
            f"-> {'一致' if assistant_calls == tool_messages else '不一致(会 400)'}"
        )


def demo_bad_args_script() -> list[ScriptedTurn]:
    """演示专用剧本：故意先给错参数，再看模型怎么根据错误回传改正。

    为什么不直接用 ``app.llm.script_bad_args_then_ok()``：
    第 1 步的 ``script_full_demo()`` 里已经有一条「复习高数」，两段剧本共用同一个
    临时库，如果这里也写「复习高数」，演示输出就会出现两条一模一样的待办——
    现场看起来像 bug（其实是两次独立调用）。所以演示脚本自带一份标题不同的剧本，
    共享剧本保持原样（它被测试用作固定夹具，标题不该被演示需求改动）。
    """
    return [
        ScriptedTurn(
            tool_calls=(
                # 第一轮故意错两处：title 全是空格、due_at 写成中文日期
                ScriptedToolCall(
                    name="add_todo",
                    arguments='{"title": "   ", "due_at": "10月25日"}',
                ),
            )
        ),
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments='{"title": "复习线代", "due_at": "2026-10-25 19:00"}',
                ),
            )
        ),
        ScriptedTurn(content="第一次时间格式写错了，现在已按 2026-10-25 19:00 记好「复习线代」。"),
    ]


def run_validation_demo(repo: TodoRepository) -> None:
    """第二步：单独演示"参数错了 -> 回传错误 -> 模型改正"。"""
    print_section("2) 校验失败回传给模型（不是抛异常）")
    script = demo_bad_args_script()
    print("剧本：第1轮故意把 due_at 写成『10月25日』、title 写成空格")
    client = MockLLMClient(script, model="mock-invalid")
    agent = TodoAgent(client, repo, max_turns=6, model="mock-invalid")
    outcome = agent.chat("下周日晚上七点提醒我复习线代", user_id=DEFAULT_USER)

    print()
    print("工具调用轨迹：")
    for index, record in enumerate(outcome.tool_calls, start=1):
        flag = "ok" if record.ok else "FAILED"
        print(f"  [{index}] {record.name}  args={record.arguments}")
        print(f"      -> {flag}: {record.result}")
    print()
    print(f"助手最终回复：{outcome.reply}")


def run_stream_demo(repo: TodoRepository) -> None:
    """第三步：流式演示，重点看 tool_calls 分片后是否被正确拼接。"""
    print_section("3) 流式拼接（tool_calls 分片到达，按 index 累加 arguments）")
    script = [
        ScriptedTurn(
            tool_calls=(
                ScriptedToolCall(
                    name="add_todo",
                    arguments=(
                        '{"title": "写编译原理词法分析实验报告", '
                        '"due_at": "2026-10-28 22:00"}'
                    ),
                ),
            )
        ),
        ScriptedTurn(content="已记好：写编译原理词法分析实验报告，2026-10-28 22:00 前完成。"),
    ]
    client = MockLLMClient(script, model="mock-stream")
    agent = TodoAgent(client, repo, max_turns=4, model="mock-stream")

    print("事件流（token 会一行行刷出来，tool_call/tool_result 是工具轨迹）：")
    print("助手：", end="")
    final_text = ""
    for kind, payload in agent.chat_stream("10月28号晚上十点前写完编译原理实验报告", user_id=DEFAULT_USER):
        if kind == "token":
            print(payload, end="", flush=True)
        elif kind == "tool_call":
            print(f"\n  [tool_call] {payload}", flush=True)
        elif kind == "tool_result":
            print(f"  [tool_result] {payload}", flush=True)
        elif kind == "done":
            final_text = payload
    print()
    print(f"最终回复（done 事件的 payload）：{final_text}")


def run_state_demo(repo: TodoRepository) -> None:
    """第四步：直接查库，证明数据真的落盘了（不是模型自己编的）。"""
    print_section("4) 直接读库验证（数据真落盘，不是模型编的）")
    print(f"数据库文件：{repo.db_path}")
    for todo in repo.list(user_id=DEFAULT_USER):
        due = todo.due_at or "无"
        print(
            f"  #{todo.id} [{todo.status}] {todo.title}  截止={due}  "
            f"created_at={todo.created_at}"
        )
    pending = repo.list(status="pending", user_id=DEFAULT_USER)
    done = repo.list(status="done", user_id=DEFAULT_USER)
    print(f"统计：未完成 {len(pending)} 条，已完成 {len(done)} 条")


def main() -> int:
    prepare_stdout()
    db_path = fresh_db()
    repo = TodoRepository(db_path)
    repo.init_schema()

    print("离线演示开始：不需要 API Key，不联网。")
    run_sync_demo(repo)
    run_validation_demo(repo)
    run_stream_demo(repo)
    run_state_demo(repo)

    print()
    print("=" * 68)
    print("演示结束：模型选工具 -> 参数校验 -> 执行 -> 结果回填 -> 最终回复，全链路已跑通。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
