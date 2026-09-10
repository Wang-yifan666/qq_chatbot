"""services/scheduled_tasks.py：morning_greeting 执行流程测试（v0.4）。

全部 mock：不连 QQ、不调真实 LLM、不发真实消息；scheduled_task_runs 用
conftest 的临时 SQLite，assistant 消息写入也走真实 context_store。
"""

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import services.scheduled_tasks as st
from services import context_store
from services.scheduled_task_store import get_scheduled_task_status
from services.scheduled_task_store import is_task_done_today

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2025, 1, 1, 8, 0, tzinfo=TZ)


def make_task(**overrides) -> st.ScheduledTask:
    base = dict(
        task_id=f"mg-{uuid.uuid4().hex[:8]}",
        event_type="morning_greeting",
        enabled=True,
        hour=8,
        minute=0,
        target_group_ids=frozenset({111}),
        allow_tools=False,
        catchup_minutes=30,
        skip_if_active_minutes=10,
    )
    base.update(overrides)
    return st.ScheduledTask(**base)


@pytest.fixture
def fake_llm(monkeypatch):
    """替代 ask_with_fallback：记录 messages 与 tools，返回固定回答。"""
    calls = {"messages": [], "tools": []}

    async def fake(messages, tools=None):
        calls["messages"].append(messages)
        calls["tools"].append(tools)
        return ("早上好。", "deepseek")

    monkeypatch.setattr(st, "ask_with_fallback", fake)
    return calls


@pytest.fixture
def fake_bot(monkeypatch):
    """替代 OneBot 连接 + 发送：记录发送内容，发送总是成功。"""
    bot = SimpleNamespace(self_id="999")
    sent: list[tuple[int, str]] = []

    async def send(bot, group_id, text):
        sent.append((group_id, text))
        return True

    monkeypatch.setattr(st, "get_onebot_bot", lambda: bot)
    monkeypatch.setattr(st, "send_group_message", send)
    return {"bot": bot, "sent": sent}


@pytest.fixture
def capture_build(monkeypatch):
    """替代 build_messages：捕获调用参数（验证 conversation_mode / 无 current_user）。"""
    captured: dict = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

    monkeypatch.setattr(st, "build_messages", fake_build)
    return captured


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(st, "get_now", lambda: NOW)


@pytest.fixture(autouse=True)
def _deterministic_activity(monkeypatch):
    """默认“最近没说过话”：真实 messages 表里可能有其它测试写入的
    assistant 行，会干扰 skip-if-active 判断；skip 测试自行覆盖为 True。"""

    async def inactive(group_id, minutes):
        return False

    monkeypatch.setattr(st, "has_recent_bot_message", inactive)


class TestExecuteMorningGreeting:
    async def test_unauthorized_group_never_sends(
        self, monkeypatch, fake_llm, capture_build, fixed_clock
    ):
        task = make_task(task_id="unauth")
        status = await st.execute_scheduled_task(333, task, NOW)
        assert status == "unauthorized"
        assert fake_llm["messages"] == []
        assert not await is_task_done_today(task.task_id, 333, "2025-01-01")

    async def test_full_success_flow(
        self, monkeypatch, fake_llm, fake_bot, capture_build, fixed_clock
    ):
        task = make_task(task_id="full")
        status = await st.execute_scheduled_task(111, task, NOW)
        assert status == "success"
        # Scheduled 默认不调用任何工具（tools=None）
        assert fake_llm["tools"] == [None]
        # 实际发送内容 = 模型输出
        assert fake_bot["sent"] == [(111, "早上好。")]
        # 发送成功后 assistant 消息写入 Context（DIRECT 之后能看到）
        rows = await context_store.get_recent_messages(111, 5)
        assert any(r.role == "assistant" and r.content == "早上好。" for r in rows)
        # 执行状态落 success
        assert await get_scheduled_task_status(task.task_id, 111, "2025-01-01") == "success"
        # Prompt：conversation_mode=scheduled，没有 current_user，有可信 ScheduledEvent
        assert capture_build["conversation_mode"] == "scheduled"
        assert capture_build["args"][0] is None
        assert capture_build["args"][4] == ""
        event = capture_build["scheduled_event"]
        assert event.event_type == "morning_greeting"
        assert event.scheduled_time == "08:00"
        assert event.local_datetime == "2025-01-01 08:00:00"

    async def test_night_greeting_runs_through_same_engine(
        self, monkeypatch, fake_llm, fake_bot, capture_build, fixed_clock
    ):
        """注册表新增任务（night_greeting）复用同一执行引擎，无需任何新执行代码。"""
        night_now = datetime(2025, 1, 1, 21, 0, tzinfo=TZ)
        monkeypatch.setattr(st, "get_now", lambda: night_now)
        task = make_task(
            task_id="night_greeting",
            event_type="night_greeting",
            hour=21,
            minute=0,
        )
        status = await st.execute_scheduled_task(111, task, night_now)
        assert status == "success"
        assert fake_llm["tools"] == [None]
        assert fake_bot["sent"] == [(111, "早上好。")]
        event = capture_build["scheduled_event"]
        assert event.event_type == "night_greeting"
        assert event.scheduled_time == "21:00"
        assert await get_scheduled_task_status("night_greeting", 111, "2025-01-01") == "success"

    async def test_second_run_same_day_skipped(
        self, monkeypatch, fake_llm, fake_bot, capture_build, fixed_clock
    ):
        task = make_task(task_id="twice")
        assert await st.execute_scheduled_task(111, task, NOW) == "success"
        status = await st.execute_scheduled_task(111, task, NOW)
        assert status == "claimed"
        # 第二次完全没有模型调用 / 发送
        assert len(fake_llm["messages"]) == 1
        assert len(fake_bot["sent"]) == 1

    async def test_each_group_has_independent_state(
        self, monkeypatch, fake_llm, fake_bot, fixed_clock
    ):
        task = make_task(task_id="groups")
        assert await st.execute_scheduled_task(111, task, NOW) == "success"
        assert await st.execute_scheduled_task(222, task, NOW) == "success"
        assert await get_scheduled_task_status(task.task_id, 111, "2025-01-01") == "success"
        assert await get_scheduled_task_status(task.task_id, 222, "2025-01-01") == "success"

    async def test_skip_when_recently_active(self, monkeypatch, fake_llm, fixed_clock):
        task = make_task(task_id="active")

        async def active(group_id, minutes):
            return True

        monkeypatch.setattr(st, "has_recent_bot_message", active)
        status = await st.execute_scheduled_task(111, task, NOW)
        assert status == "skipped_active"
        assert fake_llm["messages"] == []
        assert await get_scheduled_task_status(task.task_id, 111, "2025-01-01") == "skipped_active"

    async def test_bot_offline_fails_without_crash(self, monkeypatch, fake_llm, fixed_clock):
        task = make_task(task_id="offline")
        monkeypatch.setattr(st, "get_onebot_bot", lambda: None)
        status = await st.execute_scheduled_task(111, task, NOW)
        assert status == "failed"
        # bot 检查在生成之前：离线时不做任何模型调用
        assert fake_llm["messages"] == []
        assert await get_scheduled_task_status(task.task_id, 111, "2025-01-01") == "failed"

    async def test_provider_failure_failed(self, monkeypatch, fake_bot, fixed_clock):
        task = make_task(task_id="noprovider")

        async def fake(messages, tools=None):
            return (None, "deepseek")

        monkeypatch.setattr(st, "ask_with_fallback", fake)
        status = await st.execute_scheduled_task(111, task, NOW)
        assert status == "failed"
        assert fake_bot["sent"] == []
        assert await get_scheduled_task_status(task.task_id, 111, "2025-01-01") == "failed"

    async def test_send_failure_failed(self, monkeypatch, fake_llm, fixed_clock):
        task = make_task(task_id="sendfail")
        monkeypatch.setattr(
            st, "get_onebot_bot", lambda: SimpleNamespace(self_id="999")
        )

        async def send(bot, group_id, text):
            return False

        monkeypatch.setattr(st, "send_group_message", send)
        status = await st.execute_scheduled_task(111, task, NOW)
        assert status == "failed"
        assert await get_scheduled_task_status(task.task_id, 111, "2025-01-01") == "failed"

    async def test_job_iterates_all_target_groups(self, monkeypatch):
        executed: list[int] = []

        async def fake_exec(group_id, task=None, now=None):
            executed.append(group_id)
            return "success"

        monkeypatch.setattr(st, "execute_scheduled_task", fake_exec)
        monkeypatch.setattr(st, "get_now", lambda: NOW)
        st._run_task_job(make_task(task_id="job", target_group_ids=frozenset({111, 222})))
        await asyncio.sleep(0.05)
        assert sorted(executed) == [111, 222]

    async def test_scheduled_waits_for_shared_group_lock(
        self, monkeypatch, fake_llm, fake_bot, fixed_clock
    ):
        """DIRECT 持锁时，SCHEDULED 必须等待同一把 per-group 锁（不并发）。"""
        from services.group_conversation import get_group_conversation_state

        state = get_group_conversation_state(111)
        task = make_task(task_id="locktest")
        order: list[str] = []

        async def holder():
            async with state.lock:
                order.append("direct")
                await asyncio.sleep(0.15)

        async def scheduled():
            order.append("scheduled_start")
            await st.execute_scheduled_task(111, task, NOW)
            order.append("scheduled_end")

        t1 = asyncio.create_task(holder())
        await asyncio.sleep(0.03)
        t2 = asyncio.create_task(scheduled())
        await asyncio.sleep(0.06)
        # holder 还没释放：scheduled 已进入等待
        assert order == ["direct", "scheduled_start"]
        await asyncio.gather(t1, t2)
        assert order == ["direct", "scheduled_start", "scheduled_end"]


class TestStartupCatchup:
    async def test_catchup_executes_missed_task(self, monkeypatch):
        executed: list[int] = []

        async def fake_exec(group_id, task=None, now=None):
            executed.append(group_id)
            return "success"

        monkeypatch.setattr(st, "execute_scheduled_task", fake_exec)
        monkeypatch.setattr(
            st, "get_now", lambda: datetime(2025, 1, 1, 8, 10, tzinfo=TZ)
        )

        async def not_done(task_id, group_id, date):
            return False

        monkeypatch.setattr(st, "is_task_done_today", not_done)
        task = make_task(task_id="catchup", target_group_ids=frozenset({111, 222}))
        await st._make_catchup(task)()
        await asyncio.sleep(0.05)
        assert sorted(executed) == [111, 222]

    async def test_catchup_skips_groups_already_done_today(self, monkeypatch):
        executed: list[int] = []

        async def fake_exec(group_id, task=None, now=None):
            executed.append(group_id)
            return "success"

        monkeypatch.setattr(st, "execute_scheduled_task", fake_exec)
        monkeypatch.setattr(
            st, "get_now", lambda: datetime(2025, 1, 1, 8, 10, tzinfo=TZ)
        )

        async def done(task_id, group_id, date):
            return group_id == 111

        monkeypatch.setattr(st, "is_task_done_today", done)
        task = make_task(task_id="catchup2", target_group_ids=frozenset({111, 222}))
        await st._make_catchup(task)()
        await asyncio.sleep(0.05)
        assert executed == [222]

    async def test_catchup_not_run_before_scheduled_time(self, monkeypatch):
        executed: list[int] = []

        async def fake_exec(group_id, task=None, now=None):
            executed.append(group_id)
            return "success"

        monkeypatch.setattr(st, "execute_scheduled_task", fake_exec)
        # 07:59 启动：不提前补执行
        monkeypatch.setattr(
            st, "get_now", lambda: datetime(2025, 1, 1, 7, 59, tzinfo=TZ)
        )
        task = make_task(task_id="catchup3", target_group_ids=frozenset({111}))
        await st._make_catchup(task)()
        await asyncio.sleep(0.05)
        assert executed == []
