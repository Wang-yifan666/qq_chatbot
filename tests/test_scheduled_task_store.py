"""services/scheduled_task_store.py：claim 幂等与执行状态测试（v0.4）。

使用 tests/conftest.py 指定的临时 SQLite；每个测试用独立 task_id / 日期，
避免跨测试互相污染。
"""

import asyncio
import uuid

import pytest

from services.context_store import add_message
from services.context_store import has_recent_bot_message
from services.scheduled_task_store import claim_scheduled_task
from services.scheduled_task_store import get_scheduled_task_status
from services.scheduled_task_store import is_task_done_today
from services.scheduled_task_store import mark_scheduled_task
from services.scheduled_task_store import reclaim_scheduled_task


def _tid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TestClaim:
    async def test_first_claim_wins_second_skipped(self):
        task_id = _tid("claim")
        date = "2025-01-01"
        assert await claim_scheduled_task(task_id, 111, date) is True
        # 同任务 / 同群 / 同一天：第二次必须失败（幂等）
        assert await claim_scheduled_task(task_id, 111, date) is False
        assert await get_scheduled_task_status(task_id, 111, date) == "running"

    async def test_different_groups_have_independent_state(self):
        task_id = _tid("groups")
        date = "2025-01-02"
        assert await claim_scheduled_task(task_id, 111, date) is True
        assert await claim_scheduled_task(task_id, 222, date) is True
        assert await claim_scheduled_task(task_id, 333, date) is True

    async def test_different_days_have_independent_state(self):
        task_id = _tid("days")
        assert await claim_scheduled_task(task_id, 111, "2025-01-03") is True
        # 第二天（模拟重启 / 第二天 cron）：可以再次执行
        assert await claim_scheduled_task(task_id, 111, "2025-01-04") is True

    async def test_different_tasks_have_independent_state(self):
        date = "2025-01-05"
        assert await claim_scheduled_task(_tid("t1"), 111, date) is True
        assert await claim_scheduled_task(_tid("t2"), 111, date) is True


class TestMarkAndQuery:
    async def test_mark_updates_status(self):
        task_id = _tid("mark")
        date = "2025-01-06"
        await claim_scheduled_task(task_id, 111, date)
        assert await mark_scheduled_task(task_id, 111, date, "success") is True
        assert await get_scheduled_task_status(task_id, 111, date) == "success"
        assert await is_task_done_today(task_id, 111, date) is True

    async def test_any_status_blocks_retry(self):
        # “今天已有记录”无论成功失败都阻止重试（宁少一次，不重复发送）
        task_id = _tid("blocked")
        date = "2025-01-07"
        await claim_scheduled_task(task_id, 111, date)
        await mark_scheduled_task(task_id, 111, date, "failed")
        assert await claim_scheduled_task(task_id, 111, date) is False

    async def test_no_record_returns_none(self):
        task_id = _tid("none")
        assert await get_scheduled_task_status(task_id, 111, "2025-01-08") is None
        assert await is_task_done_today(task_id, 111, "2025-01-08") is False


class TestReclaim:
    async def test_failed_can_be_reclaimed(self):
        task_id = _tid("reclaim")
        date = "2025-01-09"
        await claim_scheduled_task(task_id, 111, date)
        await mark_scheduled_task(task_id, 111, date, "failed")
        assert await reclaim_scheduled_task(task_id, 111, date) is True
        assert await get_scheduled_task_status(task_id, 111, date) == "running"

    async def test_success_never_reclaimed(self):
        task_id = _tid("reclaim-success")
        date = "2025-01-10"
        await claim_scheduled_task(task_id, 111, date)
        await mark_scheduled_task(task_id, 111, date, "success")
        assert await reclaim_scheduled_task(task_id, 111, date) is False
        assert await get_scheduled_task_status(task_id, 111, date) == "success"

    async def test_skipped_never_reclaimed(self):
        task_id = _tid("reclaim-skipped")
        date = "2025-01-11"
        await claim_scheduled_task(task_id, 111, date)
        await mark_scheduled_task(task_id, 111, date, "skipped_active")
        assert await reclaim_scheduled_task(task_id, 111, date) is False

    async def test_running_never_reclaimed(self):
        task_id = _tid("reclaim-running")
        date = "2025-01-12"
        await claim_scheduled_task(task_id, 111, date)
        # 仍是 running（可能已发出）→ 不允许 reclaim
        assert await reclaim_scheduled_task(task_id, 111, date) is False

    async def test_reclaim_without_record_fails(self):
        task_id = _tid("reclaim-none")
        assert await reclaim_scheduled_task(task_id, 111, "2025-01-13") is False


class TestHasRecentBotMessage:
    async def test_recent_assistant_message_detected(self):
        await add_message(111, 999, "夜子", "assistant", "刚刚说的")
        assert await has_recent_bot_message(111, 10) is True

    async def test_old_assistant_message_not_detected(self):
        from services.database import db_conn
        from services.database import ensure_db

        assert await ensure_db()
        conn = db_conn()
        # 用独立 group 并先清空：其它测试可能往常见 group 写过“刚刚”的 assistant 行
        group_id = 9002
        await conn.execute("DELETE FROM messages WHERE group_id = ?", (group_id,))
        await conn.execute(
            "INSERT INTO messages (group_id, user_id, nickname, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, 999, "夜子", "assistant", "很久以前", "2020-01-01 00:00:00"),
        )
        await conn.commit()
        assert await has_recent_bot_message(group_id, 10) is False

    async def test_other_group_not_affected(self):
        await add_message(333, 999, "夜子", "assistant", "333 群刚说过")
        assert await has_recent_bot_message(444, 10) is False

    async def test_user_messages_do_not_count(self):
        await add_message(555, 1001, "小明", "user", "用户消息不算")
        assert await has_recent_bot_message(555, 10) is False

    async def test_zero_minutes_disabled(self):
        assert await has_recent_bot_message(666, 0) is False
