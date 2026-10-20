"""SQLite 存储层。

几个刻意的选择：

* **不用 ORM**。这个项目的数据只有一个表，SQL 一共五条。引入 SQLAlchemy 只会让
  "老师追问时你得再解释一层"，用标准库 ``sqlite3`` 反而能把事务、索引、WAL 讲清楚。
* **每个操作一条短连接**。sqlite3 的连接不是线程安全的；FastAPI 的同步路由跑在线程池里，
  如果共享一个全局连接，多请求并发时会踩 "SQLite objects created in a thread can only be
  used in that same thread" 或写锁冲突。短连接 + WAL + busy_timeout 是最省心的组合。
* **时间存字符串**。SQLite 没有 datetime 类型，``datetime.now().isoformat(timespec="seconds")``
  产生的 ``2026-10-20T20:15:03`` 既能直接比较排序，也能原样再解析回来。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .models import Todo, TodoStatus

# 表结构（含索引）。IF NOT EXISTS 让 init_schema 可以反复调用。
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS todos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      title TEXT NOT NULL,
      due_at TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      created_at TEXT NOT NULL,
      done_at TEXT
    )
    """,
    # 查询几乎永远是 "某个用户的某状态待办"，所以建联合索引；顺序 (user_id, status) 不能反，
    # 因为最左前缀匹配要求 user_id 在前。
    "CREATE INDEX IF NOT EXISTS idx_todos_user_status ON todos(user_id, status)",
)


def _now() -> str:
    """统一的时间戳格式：秒级 ISO 字符串。"""
    return datetime.now().isoformat(timespec="seconds")


class TodoRepository:
    """待办数据访问对象。所有方法都是同步阻塞的，调用方负责放到线程里跑。"""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        parent = self._db_path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """建立连接并立刻设置三个 PRAGMA。"""
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # WAL：读写不互相阻塞（读操作不会挡住写），演示时一边查一边写不会卡住。
        conn.execute("PRAGMA journal_mode=WAL")
        # busy_timeout：拿不到写锁时最多等 5 秒再报 "database is locked"，
        # 而不是立刻抛错。等价于 sqlite3.connect(timeout=5.0)，但显式写出来更好讲。
        conn.execute("PRAGMA busy_timeout=5000")
        # 这个项目其实没有外键，但保持开启是个好习惯（以后加表不用再想）。
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """带事务与关闭的连接上下文。

        不用 ``with sqlite3.connect(...)`` 的原因：那个写法只提交事务，**不关闭连接**，
        短连接场景下会积累一堆未关闭的句柄，WAL 文件也可能迟迟不回收。
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """建表建索引。幂等，可以在服务启动时随便调。"""
        with self._session() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)

    def add(self, title: str, due_at: str | None = None, user_id: str = "default") -> Todo:
        """新增一条待办，返回落库后的完整记录（含自增 id）。

        用 ``INSERT ... RETURNING *`` 一次往返拿到整行。老写法是"先 INSERT，再开一条
        连接 SELECT 回读"：多一次连接开销，而且两条语句之间理论上有并发插队的窗口。
        RETURNING 需要 SQLite >= 3.35（Python 3.13 自带 3.45+）。
        """
        created_at = _now()
        with self._session() as conn:
            row = conn.execute(
                """
                INSERT INTO todos (user_id, title, due_at, status, created_at, done_at)
                VALUES (?, ?, ?, 'pending', ?, NULL)
                RETURNING *
                """,
                (user_id, title, due_at, created_at),
            ).fetchone()
        if row is None:  # RETURNING 拿不到行说明 INSERT 没生效，必须立刻暴露而不是返回 None
            raise RuntimeError("写入待办后没有拿到返回行")
        return _row_to_todo(row)

    def get(self, todo_id: int, user_id: str = "default") -> Todo | None:
        """按 id 查一条。查不到返回 None（不抛异常，由上层翻译成人话）。"""
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM todos WHERE id = ? AND user_id = ?",
                (todo_id, user_id),
            ).fetchone()
        return _row_to_todo(row) if row is not None else None

    def list(self, status: TodoStatus | None = None, user_id: str = "default") -> list[Todo]:
        """列待办。未完成的排前面，再按截止时间（无时间的排最后），最后按 id。"""
        sql = "SELECT * FROM todos WHERE user_id = ?"
        params: list[object] = [user_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += """
            ORDER BY
              CASE status WHEN 'pending' THEN 0 ELSE 1 END,
              COALESCE(due_at, '9999-12-31 23:59'),
              id
        """
        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_todo(row) for row in rows]

    def complete(self, todo_id: int, user_id: str = "default") -> Todo | None:
        """标记完成（单语句、原子）。

        老写法是 get -> UPDATE -> get 三次往返：检查与更新之间存在 TOCTOU 窗口
        （行可能在两次连接之间被并发改掉），也多开了两次连接。现在一条
        ``UPDATE ... RETURNING`` 全部完成：

        * 查不到行（id 不存在 / 不是这个用户的）-> RETURNING 无返回 -> ``None``，
          上层翻译成「没找到 id=3 的待办」；
        * ``COALESCE(done_at, ?)`` 保证幂等：已完成过的再标一次**不刷新 done_at**，
          用户重复说"做完了"不该改变第一次完成的时间。
        """
        done_at = _now()
        with self._session() as conn:
            row = conn.execute(
                """
                UPDATE todos
                SET status = 'done', done_at = COALESCE(done_at, ?)
                WHERE id = ? AND user_id = ?
                RETURNING *
                """,
                (done_at, todo_id, user_id),
            ).fetchone()
        return _row_to_todo(row) if row is not None else None

    def delete(self, todo_id: int, user_id: str = "default") -> bool:
        """物理删除一条。返回是否真的删掉了。"""
        with self._session() as conn:
            cursor = conn.execute(
                "DELETE FROM todos WHERE id = ? AND user_id = ?",
                (todo_id, user_id),
            )
            deleted = cursor.rowcount
        return bool(deleted)


def _row_to_todo(row: sqlite3.Row) -> Todo:
    """sqlite3.Row -> Pydantic 模型。用 dict(row) 保留列名，不依赖列顺序。"""
    return Todo.model_validate(dict(row))


_DEFAULT_REPOSITORY: TodoRepository | None = None


def get_default_repository() -> TodoRepository:
    """进程级默认仓库（API 的 Depends 和工具层兜底都会用到）。

    延迟导入 config 是为了避免模块级循环导入。
    """
    global _DEFAULT_REPOSITORY
    if _DEFAULT_REPOSITORY is None:
        from .config import get_settings

        repository = TodoRepository(get_settings().db_path)
        repository.init_schema()
        _DEFAULT_REPOSITORY = repository
    return _DEFAULT_REPOSITORY
