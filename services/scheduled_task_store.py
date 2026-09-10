"""Scheduled Task 执行状态存储（v0.4）：原子 claim + 状态更新。

scheduled_task_runs 表（建表在 services/database.py）：
    UNIQUE(task_id, group_id, scheduled_date) 是幂等的数据库级保障：
    “每个任务 / 每个群 / 每天”最多一条记录。

claim 语义（at-most-once）：
- 执行前 INSERT OR IGNORE status='running'；插入失败 = 今天已有记录 → skip；
- 执行后 UPDATE 状态（success / failed / skipped_*）；
- 进程在 claim 之后、mark 之前崩溃：记录停留在 'running'，
  今天的任何重试 / 多实例 / catch-up 都会因为唯一约束直接跳过
  （宁少一次，不重复发送）。

scheduled_date 是 BOT_TIMEZONE 下的本地日期字符串 YYYY-MM-DD。
"""

from nonebot import logger

from services import redact_secrets
from services.database import db_conn
from services.database import ensure_db

# 合法终态（running 是中间态，只有进程崩溃才会残留）
VALID_STATUSES = ("running", "success", "failed", "skipped_active", "skipped_inactive", "skipped_unauthorized")


async def claim_scheduled_task(task_id: str, group_id: int, scheduled_date: str) -> bool:
    """原子认领“今天这个群这个任务”。返回 True = 认领成功（可继续执行）；
    False = 今天已有记录（成功/失败/进行中），必须跳过。

    失败（数据库不可用）也返回 False：保守跳过，绝不让 DB 故障导致重复发送。
    """
    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    try:
        cursor = await conn.execute(
            "INSERT OR IGNORE INTO scheduled_task_runs "
            "(task_id, group_id, scheduled_date, status) VALUES (?, ?, ?, 'running')",
            (task_id, group_id, scheduled_date),
        )
        await conn.commit()
        return cursor.rowcount > 0
    except Exception as exc:
        logger.error(
            "[SCHEDULED] claim 失败 (task={} group_id={} date={}): {}: {}",
            task_id,
            group_id,
            scheduled_date,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def mark_scheduled_task(
    task_id: str,
    group_id: int,
    scheduled_date: str,
    status: str,
) -> bool:
    """把已认领的记录更新为终态。失败只记日志并返回 False（绝不抛出）。"""
    if status not in VALID_STATUSES:
        status = "failed"
    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    try:
        await conn.execute(
            "UPDATE scheduled_task_runs SET status = ?, finished_at = CURRENT_TIMESTAMP "
            "WHERE task_id = ? AND group_id = ? AND scheduled_date = ?",
            (status, task_id, group_id, scheduled_date),
        )
        await conn.commit()
        return True
    except Exception as exc:
        logger.error(
            "[SCHEDULED] mark 失败 (task={} group_id={} date={} status={}): {}: {}",
            task_id,
            group_id,
            scheduled_date,
            status,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def get_scheduled_task_status(
    task_id: str, group_id: int, scheduled_date: str
) -> str | None:
    """读取某次执行的状态；没有记录 / 数据库不可用返回 None。"""
    if not await ensure_db():
        return None
    conn = db_conn()
    if conn is None:
        return None
    try:
        cursor = await conn.execute(
            "SELECT status FROM scheduled_task_runs "
            "WHERE task_id = ? AND group_id = ? AND scheduled_date = ? LIMIT 1",
            (task_id, group_id, scheduled_date),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return str(row["status"]) if row is not None else None
    except Exception as exc:
        logger.error(
            "[SCHEDULED] 读取执行状态失败 (task={} group_id={} date={}): {}: {}",
            task_id,
            group_id,
            scheduled_date,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None


async def is_task_done_today(task_id: str, group_id: int, scheduled_date: str) -> bool:
    """今天这个群这个任务是否已有执行记录（任何状态都算，用于 catch-up 预检）。"""
    if not await ensure_db():
        return False
    conn = db_conn()
    if conn is None:
        return False
    try:
        cursor = await conn.execute(
            "SELECT 1 FROM scheduled_task_runs "
            "WHERE task_id = ? AND group_id = ? AND scheduled_date = ? LIMIT 1",
            (task_id, group_id, scheduled_date),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None
    except Exception as exc:
        logger.error(
            "[SCHEDULED] 查询执行记录失败 (task={} group_id={} date={}): {}: {}",
            task_id,
            group_id,
            scheduled_date,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False
