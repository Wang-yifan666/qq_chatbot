"""关系服务（v0.2.2）：确定性关系升级 + 唯一 close 目标。

核心原则：
- 数据库只保存 base_level（stranger / acquaintance / familiar），
  relationships 表的 SQLite CHECK 约束直接禁止持久化 'close'；
- close 是运行时派生状态，唯一来源是 .env 的 CLOSE_USER_ID，
  整个 Bot 同一时间最多 1 个 close 用户；
- 关系升级只用确定性计数规则（direct_interaction_count），
  任何 LLM / 聊天文本都无权修改关系，更无权授予 close；
- 只有“直接和夜子产生有效互动”（@夜子 并成功得到回答）才计数；
  普通群聊消息虽然进入 Group Context，但不增加关系进度。
"""

import os

from nonebot import logger

from services import redact_secrets
from services.database import db_conn
from services.database import ensure_db

# 关系阈值（本版本不放 .env，集中在单一位置，以后按实际群聊体验调整）
ACQUAINTANCE_THRESHOLD = 5
FAMILIAR_THRESHOLD = 20

# 允许持久化的 base_level（close 绝不允许入库）
VALID_BASE_LEVELS = ("stranger", "acquaintance", "familiar")

# 有效关系等级（含运行时派生的 close）
VALID_EFFECTIVE_LEVELS = VALID_BASE_LEVELS + ("close",)


def _parse_close_user_id() -> int | None:
    """解析 .env 的 CLOSE_USER_ID：空 → None；合法 → int；非法 → ValueError。

    注意：错误信息不携带原始值（不把配置内容打进日志）。
    """
    raw = (os.getenv("CLOSE_USER_ID") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("CLOSE_USER_ID 配置非法：必须是纯数字 QQ 号") from None
    if value <= 0:
        raise ValueError("CLOSE_USER_ID 配置非法：必须是正整数 QQ 号")
    return value


# 模块导入时解析一次（bot.py 在 load_dotenv 之后导入，非法值直接报错退出）。
# 日志中绝不输出真实 CLOSE_USER_ID。
CLOSE_USER_ID: int | None = _parse_close_user_id()


def is_close_target(user_id: int) -> bool:
    """当前用户是否就是 .env 指定的唯一 close 目标。"""
    return CLOSE_USER_ID is not None and user_id == CLOSE_USER_ID


def calculate_base_level(count: int) -> str:
    """按互动次数计算 base_level（确定性规则，永不产生 close）。"""
    if count >= FAMILIAR_THRESHOLD:
        return "familiar"
    if count >= ACQUAINTANCE_THRESHOLD:
        return "acquaintance"
    return "stranger"


async def get_base_relationship(user_id: int) -> str:
    """读取数据库中的 base_level；无记录视为 stranger。"""
    if not await ensure_db():
        return "stranger"
    conn = db_conn()
    if conn is None:
        return "stranger"
    try:
        cursor = await conn.execute(
            "SELECT base_level FROM relationships WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return "stranger"
        level = row["base_level"]
        return level if level in VALID_BASE_LEVELS else "stranger"
    except Exception as exc:
        logger.error(
            "[RELATIONSHIP] 读取基础关系失败 (user_id={}): {}: {}",
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return "stranger"


async def get_effective_relationship(user_id: int) -> str:
    """运行时有效关系：close 目标 → close；其他人 → 数据库 base_level。

    真正发送给 Prompt Builder 的是这个值。
    """
    if is_close_target(user_id):
        return "close"
    return await get_base_relationship(user_id)


async def record_direct_interaction(user_id: int) -> bool:
    """记录一次有效直接互动（@夜子 且成功得到回答）。

    - 单条原子 SQL 完成 direct_interaction_count + 1 与 base_level 重算，
      两个并发请求同时读到 19 时也不会丢计数；
    - close 用户同样计数：以后 CLOSE_USER_ID 修改 / 清空时，
      原 close 用户能回落到自己的 base 等级（而不是 stranger）。
    """
    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    try:
        await conn.execute(
            "INSERT INTO relationships "
            "(user_id, base_level, direct_interaction_count, "
            " first_interaction_at, last_interaction_at, updated_at) "
            "VALUES (?, 'stranger', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "  direct_interaction_count = direct_interaction_count + 1, "
            "  base_level = CASE "
            "    WHEN direct_interaction_count + 1 >= ? THEN 'familiar' "
            "    WHEN direct_interaction_count + 1 >= ? THEN 'acquaintance' "
            "    ELSE 'stranger' "
            "  END, "
            "  last_interaction_at = CURRENT_TIMESTAMP, "
            "  updated_at = CURRENT_TIMESTAMP",
            (user_id, FAMILIAR_THRESHOLD, ACQUAINTANCE_THRESHOLD),
        )
        await conn.commit()
        return True
    except Exception as exc:
        logger.error(
            "[RELATIONSHIP] 更新互动计数失败 (user_id={}): {}: {}",
            user_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False
