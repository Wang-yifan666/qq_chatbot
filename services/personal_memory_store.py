"""个人资料键值存储（v0.2.5，管理员维护的 Personal Memory）。

与 v0.2.2 的 LLM 自动提取记忆（memory_store.py，chat_history.db 的
user_memories 表）不同：本模块管理的是管理员通过 `\\debug memory set`
显式维护的键值资料（name / hobby / skill / project / ...），存于独立数据库
data/qq_ai_bot.db（可用环境变量 MEMORY_DB_PATH 覆盖），供 Mini-RAG
Retriever 检索。本版本**不做**自动学习：普通聊天不会写入本表。

设计要点：
- 独立数据库文件：个人资料与群聊历史分离；所有 *.db 文件均已被 .gitignore 忽略；
- aiosqlite 异步访问 + WAL + busy_timeout；连接懒恢复（启动失败后首次读写重试）；
- UNIQUE(group_id, user_id, memory_key)：set 即 upsert，重复设置覆盖旧值；
- 所有接口失败只记 ERROR 日志并安全降级（返回 False / [] / 0）：
  Memory 是增强能力，绝不让数据库故障拖垮聊天功能；
- SQL 全部集中在本模块，不散落到 plugins/。
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from nonebot import logger

from services import redact_secrets

# 项目根目录（services/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 记忆库文件路径：默认 <项目根>/data/qq_ai_bot.db；
# 可用环境变量 MEMORY_DB_PATH 覆盖（相对路径按进程工作目录解析）。
DB_PATH = Path(os.getenv("MEMORY_DB_PATH") or (_PROJECT_ROOT / "data" / "qq_ai_bot.db"))

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    nickname TEXT,
    memory_key TEXT NOT NULL,
    memory_value TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(group_id, user_id, memory_key)
);
"""

# 按“群 + 人”快速取资料
_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_user_memories_group_user "
    "ON user_memories(group_id, user_id);"
)

_PRAGMAS = ("PRAGMA journal_mode=WAL;", "PRAGMA busy_timeout=5000;")

_db: aiosqlite.Connection | None = None
_init_lock = asyncio.Lock()
# close_memory_db() 后置 True：退出阶段禁止懒重连
# （与 database.py 相同：避免后台任务重开连接导致进程退出挂起）
_closed = False


@dataclass(frozen=True)
class PersonalMemory:
    """user_memories（个人资料库）中的一行。"""

    id: int
    group_id: int
    user_id: int
    nickname: str | None
    memory_key: str
    memory_value: str
    created_at: str | None
    updated_at: str | None


async def init_memory_db() -> None:
    """初始化记忆库：自动创建 data 目录 / 库文件 / 表 / 索引。

    幂等；失败时抛出异常，由 bot.py 的启动钩子记录清晰日志，
    之后各接口会尝试懒恢复。
    """
    global _db, _closed
    async with _init_lock:
        if _db is not None:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(DB_PATH))
        conn.row_factory = aiosqlite.Row
        try:
            for pragma in _PRAGMAS:
                await conn.execute(pragma)
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_INDEX_SQL)
            await conn.commit()
        except BaseException:
            await conn.close()
            raise
        _db = conn
        _closed = False  # 显式初始化成功 → 清除“已关闭”状态


async def close_memory_db() -> None:
    """关闭记忆库连接（进程退出钩子使用）。

    关闭后 ensure 不再懒重连：退出阶段禁止重新打开连接。
    """
    global _db, _closed
    async with _init_lock:
        _closed = True
        if _db is not None:
            await _db.close()
            _db = None


async def _ensure_db() -> bool:
    """确保记忆库可用；启动时初始化失败可在这里懒恢复。"""
    global _closed
    if _closed:
        return False
    if _db is not None:
        return True
    try:
        await init_memory_db()
        return True
    except Exception as exc:
        logger.error(
            "[MEMORY] 记忆库初始化失败，本次操作降级跳过：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def ping_memory_db() -> bool:
    """记忆库健康检查（\\debug status 使用）。"""
    if not await _ensure_db():
        return False
    conn = _db
    if conn is None:
        return False
    try:
        cursor = await conn.execute("SELECT 1")
        await cursor.fetchone()
        await cursor.close()
        return True
    except Exception as exc:
        logger.error(
            "[MEMORY] ping 失败：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def set_memory(
    group_id: int,
    user_id: int,
    key: str,
    value: str,
    nickname: str | None = None,
) -> bool:
    """设置 / 覆盖一条个人资料（同群同人同 key 覆盖旧值）。

    成功返回 True，失败只记 ERROR 日志并返回 False。
    普通日志只打印 key，不打印 value（避免私人资料进入日志）。
    """
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        logger.error("[MEMORY] set 参数非法：key/value 不能为空")
        return False
    if not await _ensure_db():
        return False
    conn = _db
    if conn is None:
        return False
    try:
        await conn.execute(
            "INSERT INTO user_memories "
            "(group_id, user_id, nickname, memory_key, memory_value) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(group_id, user_id, memory_key) DO UPDATE SET "
            "  memory_value = excluded.memory_value, "
            "  nickname = COALESCE(excluded.nickname, user_memories.nickname), "
            "  updated_at = CURRENT_TIMESTAMP",
            (group_id, user_id, nickname, key, value),
        )
        await conn.commit()
        logger.info("[MEMORY] set group_id={} user_id={} key={}", group_id, user_id, key)
        return True
    except Exception as exc:
        logger.error(
            "[MEMORY] set 失败 (group_id={} user_id={} key={}): {}: {}",
            group_id,
            user_id,
            key,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def get_user_memories(group_id: int, user_id: int) -> list[PersonalMemory]:
    """读取某人在本群的全部个人资料（按 key 排序）。"""
    if not await _ensure_db():
        return []
    conn = _db
    if conn is None:
        return []
    try:
        cursor = await conn.execute(
            "SELECT id, group_id, user_id, nickname, memory_key, memory_value, "
            "       created_at, updated_at "
            "FROM user_memories WHERE group_id = ? AND user_id = ? "
            "ORDER BY memory_key",
            (group_id, user_id),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_memory(row) for row in rows]
    except Exception as exc:
        logger.error(
            "[MEMORY] 读取个人资料失败 (group_id={} user_id={}): {}: {}",
            group_id,
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return []


async def delete_memory(group_id: int, user_id: int, key: str) -> bool:
    """删除一条资料；返回是否真的删掉了（存在过）。"""
    key = (key or "").strip()
    if not key:
        return False
    if not await _ensure_db():
        return False
    conn = _db
    if conn is None:
        return False
    try:
        cursor = await conn.execute(
            "DELETE FROM user_memories WHERE group_id = ? AND user_id = ? AND memory_key = ?",
            (group_id, user_id, key),
        )
        await conn.commit()
        deleted = cursor.rowcount > 0
        await cursor.close()
        logger.info(
            "[MEMORY] delete group_id={} user_id={} key={} deleted={}",
            group_id,
            user_id,
            key,
            deleted,
        )
        return deleted
    except Exception as exc:
        logger.error(
            "[MEMORY] delete 失败 (group_id={} user_id={} key={}): {}: {}",
            group_id,
            user_id,
            key,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def clear_user_memories(group_id: int, user_id: int) -> int:
    """清空某人在本群的全部资料；返回删除条数。"""
    if not await _ensure_db():
        return 0
    conn = _db
    if conn is None:
        return 0
    try:
        cursor = await conn.execute(
            "DELETE FROM user_memories WHERE group_id = ? AND user_id = ?",
            (group_id, user_id),
        )
        await conn.commit()
        deleted = cursor.rowcount
        await cursor.close()
        logger.info(
            "[MEMORY] clear group_id={} user_id={} deleted={}",
            group_id,
            user_id,
            deleted,
        )
        return deleted
    except Exception as exc:
        logger.error(
            "[MEMORY] clear 失败 (group_id={} user_id={}): {}: {}",
            group_id,
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return 0


async def get_group_memories(group_id: int) -> list[PersonalMemory]:
    """读取本群全部个人资料（Retriever 的候选集）。"""
    if not await _ensure_db():
        return []
    conn = _db
    if conn is None:
        return []
    try:
        cursor = await conn.execute(
            "SELECT id, group_id, user_id, nickname, memory_key, memory_value, "
            "       created_at, updated_at "
            "FROM user_memories WHERE group_id = ? "
            "ORDER BY user_id, memory_key",
            (group_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_memory(row) for row in rows]
    except Exception as exc:
        logger.error(
            "[MEMORY] 读取群资料失败 (group_id={}): {}: {}",
            group_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return []


async def count_memories(group_id: int | None = None) -> int:
    """资料总条数（可指定群）。失败返回 0。"""
    if not await _ensure_db():
        return 0
    conn = _db
    if conn is None:
        return 0
    try:
        if group_id is None:
            cursor = await conn.execute("SELECT COUNT(*) AS c FROM user_memories")
        else:
            cursor = await conn.execute(
                "SELECT COUNT(*) AS c FROM user_memories WHERE group_id = ?",
                (group_id,),
            )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["c"]) if row is not None else 0
    except Exception as exc:
        logger.error(
            "[MEMORY] count 失败：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return 0


def _row_to_memory(row: aiosqlite.Row) -> PersonalMemory:
    return PersonalMemory(
        id=row["id"],
        group_id=row["group_id"],
        user_id=row["user_id"],
        nickname=row["nickname"],
        memory_key=row["memory_key"],
        memory_value=row["memory_value"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
