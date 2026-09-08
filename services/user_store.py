"""用户身份存储（v0.2.2）。

QQ user_id 是用户的稳定身份；nickname / card 只是显示名：
- 相同 user_id：同一个人，改昵称只更新显示名，绝不新建第二行；
- 不同 user_id：不同用户，即使昵称完全相同也互不相干；
- 聊天文本永远不能修改 user_id（identity 由 OneBot Event 注入）。
"""

from dataclasses import dataclass

from nonebot import logger

from services import redact_secrets
from services.database import db_conn
from services.database import ensure_db


@dataclass(frozen=True)
class User:
    """users 表中的一行。"""

    user_id: int
    latest_nickname: str | None
    first_seen_at: str | None
    last_seen_at: str | None


async def upsert_user(user_id: int, display_name: str) -> bool:
    """记录 / 刷新用户。

    相同 user_id 只更新 latest_nickname 与 last_seen_at；
    成功返回 True，失败只记 ERROR 日志并返回 False（不影响任何消息流转）。
    """
    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    name = (display_name or "").strip() or str(user_id)
    try:
        await conn.execute(
            "INSERT INTO users (user_id, latest_nickname) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "  latest_nickname = excluded.latest_nickname, "
            "  last_seen_at = CURRENT_TIMESTAMP",
            (user_id, name),
        )
        await conn.commit()
        return True
    except Exception as exc:
        logger.error(
            "[USER] upsert 失败 (user_id={}): {}: {}",
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def get_user(user_id: int) -> User | None:
    """读取用户；不存在或读取失败返回 None。"""
    if not await ensure_db():
        return None
    conn = db_conn()
    if conn is None:
        return None
    try:
        cursor = await conn.execute(
            "SELECT user_id, latest_nickname, first_seen_at, last_seen_at "
            "FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return User(
            user_id=row["user_id"],
            latest_nickname=row["latest_nickname"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
        )
    except Exception as exc:
        logger.error(
            "[USER] 读取用户失败 (user_id={}): {}: {}",
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None
