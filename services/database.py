"""SQLite 连接与建表（v0.2.2）。

本模块是 SQLite 的唯一入口：连接、建表、索引、WAL 配置都集中在这里，
services 下各 store（context_store / user_store / memory_store /
relationship_service）共用同一个 aiosqlite 连接。

表清单：
- messages         群聊短期上下文（v0.2）
- users            用户身份（user_id 稳定身份 + 最近显示名）
- relationships    基础关系（base_level 只允许 stranger/acquaintance/familiar）
- user_memories    用户长期记忆（user_id + group_id 双重隔离）
- user_relationships  好感度 affection（0~100，管理员设定，v0.2.6）

注意：relationships 的 CHECK 约束刻意不允许 'close' —— close 是运行时派生状态，
唯一来源是 .env 的 CLOSE_USER_ID；数据库层直接保证任何代码都无法持久化 close。
"""

import asyncio
import os
from pathlib import Path

import aiosqlite

from nonebot import logger

from services import redact_secrets

# 项目根目录（services/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据库文件路径：默认 <项目根>/data/chat_history.db；
# 可用环境变量 CHAT_HISTORY_DB 覆盖（相对路径按进程工作目录解析）。
DB_PATH = Path(os.getenv("CHAT_HISTORY_DB") or (_PROJECT_ROOT / "data" / "chat_history.db"))

_CREATE_TABLES_SQL = (
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        nickname TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        latest_nickname TEXT,
        first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS relationships (
        user_id INTEGER PRIMARY KEY,
        base_level TEXT NOT NULL DEFAULT 'stranger',
        direct_interaction_count INTEGER NOT NULL DEFAULT 0,
        first_interaction_at DATETIME,
        last_interaction_at DATETIME,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        CHECK (base_level IN ('stranger', 'acquaintance', 'familiar'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS user_memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        group_id INTEGER NOT NULL,
        memory_type TEXT NOT NULL,
        content TEXT NOT NULL,
        importance INTEGER NOT NULL DEFAULT 1,
        source_message_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS user_relationships (
        group_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        affection INTEGER NOT NULL DEFAULT 50,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (group_id, user_id)
    );
    """,
)

_CREATE_INDEXES_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_messages_group_id_id ON messages (group_id, id);",
    "CREATE INDEX IF NOT EXISTS idx_user_memories_user_group "
    "ON user_memories(user_id, group_id);",
    # 去重：同一用户在同一群的同类型同内容记忆只存一条
    # （INSERT OR IGNORE 依赖此唯一索引）
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memories_dedup "
    "ON user_memories(user_id, group_id, memory_type, content);",
)

# WAL 提升并发读写性能；busy_timeout 降低同时写时 database is locked 概率
_PRAGMAS = ("PRAGMA journal_mode=WAL;", "PRAGMA busy_timeout=5000;")

_db: aiosqlite.Connection | None = None
_init_lock = asyncio.Lock()
# close_db() 后置 True：退出阶段禁止懒重连，避免仍在飞行的后台任务
# 重新打开连接（aiosqlite 工作线程非守护，会导致进程退出挂起）
_closed = False


async def init_db() -> None:
    """初始化数据库：自动创建 data 目录 / 库文件 / 全部表与索引。

    幂等：已初始化时直接返回。对已有 v0.2 数据库安全
    （CREATE IF NOT EXISTS 增量建新表，无需迁移）。
    失败时抛出异常，由 bot.py 的启动钩子记录清晰日志；
    之后各 store 的读写会尝试懒恢复。
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
            for sql in _CREATE_TABLES_SQL:
                await conn.execute(sql)
            for sql in _CREATE_INDEXES_SQL:
                await conn.execute(sql)
            await conn.commit()
        except BaseException:
            await conn.close()
            raise
        _db = conn
        _closed = False  # 显式初始化成功 → 清除“已关闭”状态


async def close_db() -> None:
    """关闭数据库连接（进程退出钩子使用）。

    关闭后 ensure_db 不再懒重连：退出阶段禁止重新打开连接。
    """
    global _db, _closed
    async with _init_lock:
        _closed = True
        if _db is not None:
            await _db.close()
            _db = None


async def ensure_db() -> bool:
    """确保数据库可用；启动时初始化失败可在这里懒恢复。返回是否可用。"""
    global _closed
    if _closed:
        return False
    if _db is not None:
        return True
    try:
        await init_db()
        return True
    except Exception as exc:
        logger.error(
            "[DB] 数据库初始化失败，本次操作降级跳过：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


def db_conn() -> aiosqlite.Connection | None:
    """获取当前数据库连接（仅供 services 内各 store 模块使用，调用前先 ensure_db）。"""
    return _db
