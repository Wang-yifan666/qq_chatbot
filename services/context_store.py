"""SQLite 群聊上下文存储（v0.2）。

职责（只做这一件事）：把 QQ 群的纯文本消息写入 SQLite，并按 group_id
读取同群最近 N 条消息，供 Prompt Builder 拼装上下文。

设计要点：
- 使用 aiosqlite 异步访问 SQLite，避免在 NoneBot2 的 asyncio Handler 里
  直接调用同步 sqlite3 阻塞事件循环；
- 不引入 ORM / DAO / Repository 等复杂分层，只有几个简单接口；
- 单连接 + WAL + busy_timeout：所有 SQL 经 aiosqlite 内部同一线程串行执行，
  短并发下天然安全；WAL 与 busy_timeout 进一步降低 database is locked 概率；
- 上下文只是“增强能力”：读写失败只记 ERROR 日志并降级
  （写返回 False、读返回 []），绝不让 SQLite 临时错误拖垮整个 Bot，
  也不会把异常细节发到 QQ 群；
- 数据库文件是运行时数据，已被 .gitignore 忽略（data/*.db 等），禁止提交。
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from nonebot import logger

from services import redact_secrets

# 项目根目录（services/ 的上一级）。数据库默认放根目录下 data/ 子目录，
# 不依赖进程启动时的当前工作目录。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据库文件路径：默认 <项目根>/data/chat_history.db；
# 可用环境变量 CHAT_HISTORY_DB 覆盖（相对路径按进程工作目录解析）。
DB_PATH = Path(os.getenv("CHAT_HISTORY_DB") or (_PROJECT_ROOT / "data" / "chat_history.db"))

# ===== 上下文窗口配置（环境变量 CONTEXT_MESSAGE_LIMIT，约束 1~50，默认 20） =====
CONTEXT_MESSAGE_LIMIT_DEFAULT = 20
CONTEXT_MESSAGE_LIMIT_MIN = 1
CONTEXT_MESSAGE_LIMIT_MAX = 50

# messages 表结构（role 当前只允许 user / assistant）
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    nickname TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# group_id + id 复合索引：快速查“某群最近消息”
_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_messages_group_id_id ON messages (group_id, id);"
)

# WAL 提升并发读写性能；busy_timeout 降低同时写时 database is locked 概率
_PRAGMAS = ("PRAGMA journal_mode=WAL;", "PRAGMA busy_timeout=5000;")

_db: aiosqlite.Connection | None = None
_init_lock = asyncio.Lock()


@dataclass(frozen=True)
class ChatMessage:
    """messages 表中的一条消息。"""

    id: int
    group_id: int
    user_id: int
    nickname: str
    role: str  # user | assistant
    content: str
    created_at: str


def get_context_message_limit() -> int:
    """读取环境变量 CONTEXT_MESSAGE_LIMIT（约束 1~50）。

    未配置或非法（非整数 / 超范围）时打 warning 并使用默认值 20。
    """
    raw = (os.getenv("CONTEXT_MESSAGE_LIMIT") or "").strip()
    if not raw:
        return CONTEXT_MESSAGE_LIMIT_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "[CONTEXT] CONTEXT_MESSAGE_LIMIT={} 不是合法整数，使用默认值 {}",
            raw,
            CONTEXT_MESSAGE_LIMIT_DEFAULT,
        )
        return CONTEXT_MESSAGE_LIMIT_DEFAULT
    if not (CONTEXT_MESSAGE_LIMIT_MIN <= value <= CONTEXT_MESSAGE_LIMIT_MAX):
        logger.warning(
            "[CONTEXT] CONTEXT_MESSAGE_LIMIT={} 超出范围 [{}, {}]，使用默认值 {}",
            value,
            CONTEXT_MESSAGE_LIMIT_MIN,
            CONTEXT_MESSAGE_LIMIT_MAX,
            CONTEXT_MESSAGE_LIMIT_DEFAULT,
        )
        return CONTEXT_MESSAGE_LIMIT_DEFAULT
    return value


# 进程启动时解析一次上下文窗口大小（ai_chat 读取历史时使用；改 .env 需重启生效）
CONTEXT_MESSAGE_LIMIT = get_context_message_limit()


async def init_db() -> None:
    """初始化数据库：自动创建 data 目录 / 数据库文件 / messages 表 / 索引。

    幂等：已初始化时直接返回。失败时抛出异常，由 bot.py 的启动钩子记录清晰日志；
    之后 add_message / get_recent_messages 会尝试懒恢复。
    """
    global _db
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


async def close_db() -> None:
    """关闭数据库连接（进程退出钩子使用）。"""
    global _db
    async with _init_lock:
        if _db is not None:
            await _db.close()
            _db = None


async def _ensure_db() -> bool:
    """确保数据库可用；启动时初始化失败可在这里懒恢复。返回是否可用。"""
    if _db is not None:
        return True
    try:
        await init_db()
        return True
    except Exception as exc:
        logger.error(
            "[CONTEXT] 数据库初始化失败，本次操作降级跳过：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def add_message(
    group_id: int,
    user_id: int,
    nickname: str,
    role: str,
    content: str,
) -> bool:
    """保存一条消息；成功返回 True，失败只记 ERROR 日志并返回 False。

    调用方（ai_chat / context_recorder）不需要 try/except，
    AI 回答流程不会因保存失败而中断。
    """
    if role not in ("user", "assistant"):
        logger.error("[CONTEXT] 非法 role={}，已拒绝写入", role)
        return False
    if not await _ensure_db():
        return False
    assert _db is not None  # _ensure_db 成功后必然已建连
    try:
        # aiosqlite 内部用同一线程串行执行所有 SQL，多个协程并发写也安全
        await _db.execute(
            "INSERT INTO messages (group_id, user_id, nickname, role, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (group_id, user_id, nickname, role, content),
        )
        await _db.commit()
        return True
    except Exception as exc:
        logger.error(
            "[CONTEXT] 保存消息失败 (group_id={} user_id={}): {}: {}",
            group_id,
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def get_recent_messages(group_id: int, limit: int = 20) -> list[ChatMessage]:
    """读取某群最近 limit 条消息（按 id 升序，即时间正序）。

    只查指定 group_id，不同群之间的上下文完全隔离；
    读取失败返回 []（调用方降级为单轮问答）。
    """
    if not await _ensure_db():
        return []
    if limit <= 0:
        return []
    assert _db is not None
    try:
        cursor = await _db.execute(
            "SELECT id, group_id, user_id, nickname, role, content, created_at "
            "FROM ("
            "  SELECT * FROM messages WHERE group_id = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (group_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            ChatMessage(
                id=row["id"],
                group_id=row["group_id"],
                user_id=row["user_id"],
                nickname=row["nickname"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
    except Exception as exc:
        logger.error(
            "[CONTEXT] 读取历史失败 (group_id={}): {}: {}",
            group_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return []
