"""SQLite 群聊上下文存储（v0.2）。

职责（只做这一件事）：把 QQ 群的纯文本消息写入 messages 表，并按 group_id
读取同群最近 N 条消息，供 Prompt Builder 拼装“最近群聊记录”。

数据库连接与全部建表逻辑在 services/database.py（v0.2.2 起统一入口）；
用户 / 关系 / 长期记忆分别见 user_store / relationship_service / memory_store。

设计要点：
- 使用 aiosqlite 异步访问 SQLite，避免在 NoneBot2 的 asyncio Handler 里
  直接调用同步 sqlite3 阻塞事件循环；
- 不引入 ORM / DAO / Repository 等复杂分层，只有几个简单接口；
- 上下文只是“增强能力”：读写失败只记 ERROR 日志并降级
  （写返回 False、读返回 []），绝不让 SQLite 临时错误拖垮整个 Bot，
  也不会把异常细节发到 QQ 群；
- 数据库文件是运行时数据，已被 .gitignore 忽略（data/*.db 等），禁止提交。
"""

import os
from dataclasses import dataclass

from nonebot import logger

from services import redact_secrets
from services.database import db_conn
from services.database import ensure_db

# ===== 上下文窗口配置（环境变量 CONTEXT_MESSAGE_LIMIT，约束 1~50，默认 20） =====
CONTEXT_MESSAGE_LIMIT_DEFAULT = 20
CONTEXT_MESSAGE_LIMIT_MIN = 1
CONTEXT_MESSAGE_LIMIT_MAX = 50


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
    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    try:
        # aiosqlite 内部用同一线程串行执行所有 SQL，多个协程并发写也安全
        await conn.execute(
            "INSERT INTO messages (group_id, user_id, nickname, role, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (group_id, user_id, nickname, role, content),
        )
        await conn.commit()
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
    if not await ensure_db():
        return []
    if limit <= 0:
        return []
    conn = db_conn()
    if conn is None:
        return []
    try:
        cursor = await conn.execute(
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
