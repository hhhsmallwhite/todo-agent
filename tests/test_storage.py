"""存储层测试：CRUD 全链路、异常边界（不抛异常）、过滤、索引与多用户隔离。

这一层没什么玄学，但"不存在的 id 该返回什么"是最容易被写崩的地方：
上层（agent）依赖 ``None`` / ``False`` 来翻译成人话，这里抛异常就会一路炸到 HTTP。
"""

from __future__ import annotations

import sqlite3

import pytest

from app.models import Todo
from app.storage import TodoRepository


def test_init_schema_is_idempotent(repo: TodoRepository) -> None:
    """启动时反复调用不能报错，也不能把数据清掉。"""
    repo.add("买牛奶")

    repo.init_schema()
    repo.init_schema()

    assert len(repo.list()) == 1


def test_full_lifecycle_add_get_list_complete_delete(repo: TodoRepository) -> None:
    created = repo.add("交操作系统实验报告", due_at="2026-10-21 09:30")

    assert isinstance(created, Todo)
    assert created.id >= 1
    assert created.title == "交操作系统实验报告"
    assert created.due_at == "2026-10-21 09:30"
    assert created.status == "pending"
    assert created.done_at is None
    assert created.created_at  # 秒级 ISO 字符串，非空

    fetched = repo.get(created.id)
    assert fetched is not None
    assert fetched.model_dump() == created.model_dump()

    assert [todo.id for todo in repo.list()] == [created.id]

    completed = repo.complete(created.id)
    assert completed is not None
    assert completed.status == "done"
    assert completed.done_at

    assert repo.delete(created.id) is True
    assert repo.get(created.id) is None
    assert repo.list() == []


def test_get_missing_id_returns_none(repo: TodoRepository) -> None:
    assert repo.get(12345) is None


def test_complete_missing_id_returns_none_without_raising(repo: TodoRepository) -> None:
    """上层靠 None 判断"没找到"，这里绝不能抛。"""
    assert repo.complete(12345) is None


def test_delete_missing_id_returns_false_without_raising(repo: TodoRepository) -> None:
    assert repo.delete(12345) is False


def test_complete_is_idempotent(repo: TodoRepository) -> None:
    """用户重复说"做完了"不该报错，也不该刷新 done_at。"""
    created = repo.add("买牛奶")
    first = repo.complete(created.id)
    assert first is not None

    second = repo.complete(created.id)

    assert second is not None
    assert second.status == "done"
    assert second.done_at == first.done_at


def test_list_filters_by_status(repo: TodoRepository) -> None:
    pending = repo.add("还没做的")
    done = repo.add("已经做完的")
    repo.complete(done.id)

    pending_ids = [todo.id for todo in repo.list(status="pending")]
    done_ids = [todo.id for todo in repo.list(status="done")]
    all_ids = [todo.id for todo in repo.list()]

    assert pending_ids == [pending.id]
    assert done_ids == [done.id]
    assert set(all_ids) == {pending.id, done.id}


def test_list_is_sorted_pending_first_then_due_at_then_id(repo: TodoRepository) -> None:
    """排序契约：未完成在前 -> 按截止时间（无时间的垫底） -> 按 id。"""
    no_due = repo.add("没有截止时间")
    later = repo.add("后天交", due_at="2026-10-22 09:00")
    sooner = repo.add("明天交", due_at="2026-10-21 09:00")
    finished = repo.add("已完成的")
    repo.complete(finished.id)

    order = [todo.title for todo in repo.list()]

    assert order == ["明天交", "后天交", "没有截止时间", "已完成的"]
    assert [todo.id for todo in repo.list()][:3] == [sooner.id, later.id, no_due.id]


def test_users_are_isolated(repo: TodoRepository) -> None:
    repo.add("alice 的任务", user_id="alice")
    bob = repo.add("bob 的任务", user_id="bob")

    assert [todo.title for todo in repo.list(user_id="alice")] == ["alice 的任务"]
    assert [todo.title for todo in repo.list(user_id="bob")] == ["bob 的任务"]

    # 跨用户读、改、删都必须无效，否则"谁都能改别人的待办"。
    assert repo.get(bob.id, user_id="alice") is None
    assert repo.complete(bob.id, user_id="alice") is None
    assert repo.delete(bob.id, user_id="alice") is False

    assert repo.get(bob.id, user_id="bob") is not None


def test_user_status_index_exists(repo: TodoRepository) -> None:
    """联合索引 (user_id, status) 是查询性能的落点，用 sqlite_master 固定住。"""
    connection = sqlite3.connect(repo.db_path)
    try:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'todos'"
            )
        }
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()

    assert "todos" in table_names
    assert "idx_todos_user_status" in indexes
    assert journal_mode.lower() == "wal", "WAL 没开，读写并发时会互相阻塞"


def test_created_at_is_sortable_iso_string(repo: TodoRepository) -> None:
    """时间存字符串的前提是它能被字典序正确比较，所以必须是零填充的 ISO 格式。"""
    created = repo.add("随便一条")

    assert len(created.created_at) >= 19
    assert created.created_at[4] == "-" and created.created_at[10] == "T"


def test_failed_insert_rolls_back_and_leaves_no_partial_row(repo: TodoRepository) -> None:
    """写失败必须回滚，不能留半条脏数据；连接也要能正常关闭（后续操作不受影响）。

    这里断言的是 ``sqlite3.Error`` 而不是某个具体子类，是踩过的坑：
    同一个"把 dict 绑成 TEXT"的动作，**Python 3.10 抛 InterfaceError，
    3.11+ 抛 ProgrammingError**，两者是兄弟不是父子。写死任一个都会让
    CI 在另一个版本上红——而这条用例真正要证明的是"异常发生后回滚了"，
    不是"驱动抛了哪个类"。（这个问题是 GitHub Actions 跑 3.10 矩阵时暴露的。）
    """
    with pytest.raises(sqlite3.Error):
        # sqlite3 不支持把 dict 绑成 TEXT，这里故意制造一次"驱动层异常"。
        repo.add(title={"不": "是字符串"})  # type: ignore[arg-type]

    assert repo.list() == []
    # 回滚之后连接必须已经关闭，后续操作照常（`_session` 的 finally 里 close）
    assert repo.add("回滚之后还能继续写").title == "回滚之后还能继续写"


@pytest.mark.parametrize(
    ("due_at", "expected"),
    [("2026-10-21 09:30", "2026-10-21 09:30"), (None, None)],
)
def test_due_at_roundtrip(repo: TodoRepository, due_at: str | None, expected: str | None) -> None:
    created = repo.add("带时间和不带时间", due_at=due_at)

    assert created.due_at == expected
    assert repo.get(created.id) is not None
