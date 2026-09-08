"""用户长期记忆存储（v0.2.2）。

本版本刻意不用 RAG / Embedding / 向量数据库：
- 记忆量还很小，直接按 user_id + group_id 精确 SQL 查询，
  ORDER BY importance DESC, updated_at DESC LIMIT N；
- 双重隔离：A 用户的记忆绝不进入 B 用户 Prompt；
  同一用户在群 A 的记忆默认不进入群 B 的 Prompt（隐私边界）；
- 去重：同一用户在同一群的同类型同内容记忆只存一条
  （唯一索引 idx_user_memories_dedup + INSERT OR IGNORE）。

未来记忆膨胀后再做：user_id/group_id 精确权限过滤 → Embedding → Top-K，
向量相似度永远不是权限系统。
"""

import os
from dataclasses import dataclass

from nonebot import logger

from services import redact_secrets
from services.database import db_conn
from services.database import ensure_db

# ===== 配置（环境变量 USER_MEMORY_LIMIT，约束 1~50，默认 10） =====
USER_MEMORY_LIMIT_DEFAULT = 10
USER_MEMORY_LIMIT_MIN = 1
USER_MEMORY_LIMIT_MAX = 50

# 允许的记忆类型（与 memory_extractor 的输出 schema 一致）
VALID_MEMORY_TYPES = ("project", "skill", "preference", "goal", "fact")

MAX_CONTENT_LEN = 200
MAX_IMPORTANCE = 3


def get_user_memory_limit() -> int:
    """读取环境变量 USER_MEMORY_LIMIT；非法或超范围时告警并使用默认值 10。"""
    raw = (os.getenv("USER_MEMORY_LIMIT") or "").strip()
    if not raw:
        return USER_MEMORY_LIMIT_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "[MEMORY] USER_MEMORY_LIMIT={} 不是合法整数，使用默认值 {}",
            raw,
            USER_MEMORY_LIMIT_DEFAULT,
        )
        return USER_MEMORY_LIMIT_DEFAULT
    if not (USER_MEMORY_LIMIT_MIN <= value <= USER_MEMORY_LIMIT_MAX):
        logger.warning(
            "[MEMORY] USER_MEMORY_LIMIT={} 超出范围 [{}, {}]，使用默认值 {}",
            value,
            USER_MEMORY_LIMIT_MIN,
            USER_MEMORY_LIMIT_MAX,
            USER_MEMORY_LIMIT_DEFAULT,
        )
        return USER_MEMORY_LIMIT_DEFAULT
    return value


# 进程启动时解析一次（改 .env 需重启生效）
USER_MEMORY_LIMIT = get_user_memory_limit()


@dataclass(frozen=True)
class UserMemory:
    """user_memories 表中的一条记忆。"""

    id: int
    user_id: int
    group_id: int
    memory_type: str
    content: str
    importance: int
    created_at: str | None
    updated_at: str | None


async def add_memory(
    user_id: int,
    group_id: int,
    memory_type: str,
    content: str,
    importance: int = 1,
    source_message_id: int | None = None,
) -> bool:
    """保存一条长期记忆（重复内容自动忽略）。

    成功（含“已存在而忽略”）返回 True，失败只记 ERROR 日志并返回 False。
    """
    if memory_type not in VALID_MEMORY_TYPES:
        logger.error("[MEMORY] 非法 memory_type={}，已拒绝写入", memory_type)
        return False
    content = (content or "").strip()
    if not content:
        return False
    content = content[:MAX_CONTENT_LEN]
    try:
        importance = int(importance)
    except (TypeError, ValueError):
        importance = 1
    importance = min(MAX_IMPORTANCE, max(1, importance))

    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    try:
        # 唯一索引 (user_id, group_id, memory_type, content) 负责去重
        await conn.execute(
            "INSERT OR IGNORE INTO user_memories "
            "(user_id, group_id, memory_type, content, importance, source_message_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, group_id, memory_type, content, importance, source_message_id),
        )
        await conn.commit()
        return True
    except Exception as exc:
        logger.error(
            "[MEMORY] 保存记忆失败 (user_id={} group_id={}): {}: {}",
            user_id,
            group_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def get_user_memories(
    user_id: int,
    group_id: int,
    limit: int | None = None,
) -> list[UserMemory]:
    """读取某用户在某群的长期记忆（重要度优先，最近更新优先）。

    WHERE user_id = ? AND group_id = ? 双重过滤，用户与群双重隔离；
    读取失败返回 []（调用方降级为无记忆状态）。
    """
    if not await ensure_db():
        return []
    if limit is None:
        limit = USER_MEMORY_LIMIT
    if limit <= 0:
        return []
    conn = db_conn()
    if conn is None:
        return []
    try:
        cursor = await conn.execute(
            "SELECT id, user_id, group_id, memory_type, content, importance, "
            "       created_at, updated_at "
            "FROM user_memories "
            "WHERE user_id = ? AND group_id = ? "
            "ORDER BY importance DESC, updated_at DESC, id DESC "
            "LIMIT ?",
            (user_id, group_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            UserMemory(
                id=row["id"],
                user_id=row["user_id"],
                group_id=row["group_id"],
                memory_type=row["memory_type"],
                content=row["content"],
                importance=row["importance"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
    except Exception as exc:
        logger.error(
            "[MEMORY] 读取记忆失败 (user_id={} group_id={}): {}: {}",
            user_id,
            group_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return []
