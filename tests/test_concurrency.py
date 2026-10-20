"""并发验证：把 README「已知边界」里的"没做并发压测"这一条变成实测结论。

之前这个项目只断言了"PRAGMA 设置正确"，没证明"并发真的不炸"。老师问
"两个请求同时写会怎样"时，光指着注释说 WAL 不阻塞是站不住的——那是判断，
不是证据。这个文件用真实多线程把它跑出来。

为什么能并发安全，三句话：
1. 每个方法都是**短连接**（`_session()` 用完立刻 close），连接不在线程间共享，
   sqlite3 的 "connection can't be used across threads" 根本不会发生；
2. WAL 让读不挡写；
3. `busy_timeout=5000` 让抢不到写锁的线程**等**而不是立刻报
   "database is locked"（SQLite 同一时刻只允许一个写者）。

规模刻意压到 8 线程 × 25 次（共 200 次写）：够暴露锁问题，又不至于让
CI 跑几秒以上。这不是压测报告，是"能扛住并发"的最小证据。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from app.storage import TodoRepository

THREADS = 8
WRITES_PER_THREAD = 25


def _run_in_threads(worker: Any, count: int = THREADS) -> list[BaseException]:
    """并发跑 ``worker(index)``，收集所有异常后返回（不让异常丢在线程里被吞掉）。

    线程里的异常不会冒泡到主线程——直接 ``t.join()`` 的话测试会假绿。
    所以这里用一个屏障先让所有线程同时起步（真正抢锁），再把异常收回来。
    """
    errors: list[BaseException] = []
    barrier = threading.Barrier(count)

    def _wrapped(index: int) -> None:
        barrier.wait()
        try:
            worker(index)
        except BaseException as exc:  # noqa: BLE001 - 收集而非捕获：异常要原样交回主线程断言
            errors.append(exc)

    threads = [threading.Thread(target=_wrapped, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


def test_concurrent_adds_all_persist(repo: TodoRepository) -> None:
    """8 线程各写 25 条：一条都不能丢，一条都不能报 "database is locked"。"""

    def _worker(index: int) -> None:
        for n in range(WRITES_PER_THREAD):
            repo.add(f"线程{index}的任务{n}", user_id="default")

    errors = _run_in_threads(_worker)

    assert errors == [], f"并发写入出现异常：{errors!r}"
    todos = repo.list()
    assert len(todos) == THREADS * WRITES_PER_THREAD
    # 自增 id 在并发下也必须唯一，否则说明 RETURNING 拿到了别的线程的行。
    assert len({todo.id for todo in todos}) == THREADS * WRITES_PER_THREAD


def test_concurrent_complete_keeps_first_done_at(repo: TodoRepository) -> None:
    """多线程同时标完成同一条：`COALESCE(done_at, ?)` 必须保住**第一次**完成的时间。

    先独自完成一次并记下时间，等系统时钟走过 1 秒后再让 8 个线程同时重复完成；
    若 done_at 被刷新过，说明幂等写成了覆盖。
    """
    todo = repo.add("只需要做一次的事")
    first = repo.complete(todo.id)
    assert first is not None
    first_done_at = first.done_at

    time.sleep(1.1)  # 让 _now() 一定产出不同的秒，否则这条用例测不出覆盖

    errors = _run_in_threads(lambda _index: repo.complete(todo.id))

    assert errors == [], f"并发 complete 出现异常：{errors!r}"
    after = repo.get(todo.id)
    assert after is not None
    assert after.status == "done"
    assert after.done_at == first_done_at, "重复完成把第一次的完成时间覆盖掉了"
    # 幂等还意味着：重复标记不会把别的待办搞坏
    assert len(repo.list(status="done")) == 1


def test_concurrent_read_does_not_block_write(repo: TodoRepository) -> None:
    """一边写一边读：WAL 的意义就在这里，读线程不应该卡死或报错。"""
    stop = threading.Event()
    reads: list[int] = []

    def _reader(_index: int) -> None:
        deadline = time.time() + 3
        while not stop.is_set() and time.time() < deadline:
            reads.append(len(repo.list()))
            time.sleep(0.01)

    def _writer(index: int) -> None:
        for n in range(10):
            repo.add(f"读写并发{index}-{n}")

    reader_threads = [threading.Thread(target=_reader, args=(i,)) for i in range(2)]
    for t in reader_threads:
        t.start()

    errors = _run_in_threads(_writer)
    stop.set()
    for t in reader_threads:
        t.join(timeout=10)

    assert errors == [], f"读写并发下写入出现异常：{errors!r}"
    assert reads, "读线程一次都没读成功（被写锁堵死了）"
    assert len(repo.list()) == THREADS * 10


@pytest.mark.parametrize("threads", [4, 8])
def test_concurrent_writes_do_not_corrupt_each_others_rows(repo: TodoRepository, threads: int) -> None:
    """不同 user_id 并发写：user_id 隔离不能在并发下串味。"""

    def _worker(index: int) -> None:
        for n in range(5):
            repo.add(f"u{index}-{n}", user_id=f"user{index}")

    errors = _run_in_threads(_worker, count=threads)

    assert errors == [], f"多用户并发写出现异常：{errors!r}"
    for index in range(threads):
        rows = repo.list(user_id=f"user{index}")
        assert len(rows) == 5
        assert all(row.user_id == f"user{index}" for row in rows)
