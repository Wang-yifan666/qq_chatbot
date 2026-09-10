"""services/scheduled_tasks.py：配置解析与 catch-up 窗口纯逻辑测试（v0.4）。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from services.scheduled_tasks import ScheduledTask
from services.scheduled_tasks import parse_morning_time
from services.scheduled_tasks import parse_task_time
from services.scheduled_tasks import within_catchup_window

TZ = ZoneInfo("Asia/Shanghai")


def make_task(**overrides) -> ScheduledTask:
    base = dict(
        task_id="morning_greeting",
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
    return ScheduledTask(**base)


class TestParseMorningTime:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("08:00", (8, 0)),
            ("8:00", (8, 0)),
            ("23:59", (23, 59)),
            ("0:00", (0, 0)),
            (" 08:00 ", (8, 0)),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_morning_time(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "abc", "8", "8:5", "08", "24:00", "08:60", "08:00:00", "-1:00", "08:-5", "8点"],
    )
    def test_invalid_raises_value_error(self, raw):
        with pytest.raises(ValueError):
            parse_morning_time(raw)

    def test_error_message_does_not_leak_raw_value(self):
        with pytest.raises(ValueError) as excinfo:
            parse_morning_time("24:00")
        assert "24:00" not in str(excinfo.value)


class TestParseTaskTime:
    def test_any_prefix_supported(self):
        assert parse_task_time("NIGHT_GREETING_TIME", "21:00") == (21, 0)
        assert parse_task_time("NIGHT_GREETING_TIME", "9:30") == (9, 30)

    def test_invalid_reports_env_name(self):
        with pytest.raises(ValueError) as excinfo:
            parse_task_time("NIGHT_GREETING_TIME", "25:00")
        assert "NIGHT_GREETING_TIME" in str(excinfo.value)
        assert "25:00" not in str(excinfo.value)


class TestWithinCatchupWindow:
    def test_before_scheduled_time_not_triggered(self):
        # 07:59 启动：等 cron，catch-up 不提前执行
        now = datetime(2025, 1, 1, 7, 59, tzinfo=TZ)
        assert not within_catchup_window(now, make_task(hour=8, minute=0))

    def test_exactly_at_scheduled_time_in_window(self):
        now = datetime(2025, 1, 1, 8, 0, tzinfo=TZ)
        assert within_catchup_window(now, make_task(hour=8, minute=0))

    def test_0810_startup_within_30_minutes(self):
        # 08:10 启动且窗口 30 分钟 → 补执行
        now = datetime(2025, 1, 1, 8, 10, tzinfo=TZ)
        assert within_catchup_window(now, make_task(hour=8, minute=0, catchup_minutes=30))

    def test_boundary_exactly_30_minutes(self):
        now = datetime(2025, 1, 1, 8, 30, tzinfo=TZ)
        assert within_catchup_window(now, make_task(catchup_minutes=30))

    def test_beyond_window_not_executed(self):
        # 10:30 启动：不突然补发“早晨任务”
        now = datetime(2025, 1, 1, 10, 30, tzinfo=TZ)
        assert not within_catchup_window(now, make_task(hour=8, minute=0, catchup_minutes=30))

    def test_catchup_zero_disables(self):
        now = datetime(2025, 1, 1, 8, 10, tzinfo=TZ)
        assert not within_catchup_window(now, make_task(catchup_minutes=0))

    def test_window_respects_bot_timezone(self):
        # 同一个 UTC 时刻，在 BOT_TIMEZONE 里 08:00 才在窗口内
        utc_now = datetime(2025, 1, 1, 0, 0, tzinfo=ZoneInfo("UTC"))
        local_now = utc_now.astimezone(TZ)  # 08:00 Asia/Shanghai
        assert local_now.hour == 8
        assert within_catchup_window(local_now, make_task(hour=8, minute=0))


class TestScheduledTaskDefaults:
    def test_morning_greeting_has_no_tools(self):
        task = make_task()
        assert task.allow_tools is False
        assert task.event_type == "morning_greeting"


class TestImportTimeValidation:
    def test_invalid_morning_time_raises_at_import(self, monkeypatch):
        """非法 MORNING_GREETING_TIME 在模块导入期抛 ValueError（bot.py 启动报错退出）。"""
        import importlib

        import services.scheduled_tasks as scheduled_tasks

        monkeypatch.setenv("SCHEDULED_TASKS_ENABLED", "true")
        monkeypatch.setenv("MORNING_GREETING_TIME", "24:00")
        with pytest.raises(ValueError):
            importlib.reload(scheduled_tasks)
        # 恢复环境并重载，保证模块与后续测试处于合法状态
        monkeypatch.undo()
        importlib.reload(scheduled_tasks)
        assert scheduled_tasks.MORNING_GREETING_TASK is None

    def test_invalid_group_ids_raise_at_import(self, monkeypatch):
        import importlib

        import services.scheduled_tasks as scheduled_tasks

        monkeypatch.setenv("SCHEDULED_TASKS_ENABLED", "true")
        monkeypatch.setenv("MORNING_GREETING_TIME", "08:00")
        monkeypatch.setenv("MORNING_GREETING_GROUP_IDS", "111,abc")
        with pytest.raises(ValueError):
            importlib.reload(scheduled_tasks)
        monkeypatch.undo()
        importlib.reload(scheduled_tasks)
        assert scheduled_tasks.MORNING_GREETING_TASK is None

    def test_disabled_by_default_in_tests(self):
        import services.scheduled_tasks as scheduled_tasks

        # conftest 显式关闭：SCHEDULED_TASKS_ENABLED=false → 不构建任何任务
        assert scheduled_tasks.SCHEDULED_TASKS_ENABLED is False
        assert scheduled_tasks.SCHEDULED_TASKS == {}
        assert scheduled_tasks.MORNING_GREETING_TASK is None

    def test_registry_builds_all_tasks_when_enabled(self, monkeypatch):
        import importlib

        import services.scheduled_tasks as scheduled_tasks

        monkeypatch.setenv("SCHEDULED_TASKS_ENABLED", "true")
        monkeypatch.setenv("MORNING_GREETING_TIME", "08:00")
        monkeypatch.setenv("MORNING_GREETING_GROUP_IDS", "111")
        monkeypatch.setenv("NIGHT_GREETING_TIME", "21:00")
        monkeypatch.setenv("NIGHT_GREETING_GROUP_IDS", "111,222")
        importlib.reload(scheduled_tasks)
        try:
            assert set(scheduled_tasks.SCHEDULED_TASKS) == {"morning_greeting", "night_greeting"}
            night = scheduled_tasks.SCHEDULED_TASKS["night_greeting"]
            assert (night.hour, night.minute) == (21, 0)
            assert night.target_group_ids == frozenset({111, 222})
            assert night.allow_tools is False
            assert scheduled_tasks.MORNING_GREETING_TASK is scheduled_tasks.SCHEDULED_TASKS["morning_greeting"]
        finally:
            monkeypatch.undo()
            importlib.reload(scheduled_tasks)

    def test_invalid_night_time_raises_at_import(self, monkeypatch):
        import importlib

        import services.scheduled_tasks as scheduled_tasks

        monkeypatch.setenv("SCHEDULED_TASKS_ENABLED", "true")
        monkeypatch.setenv("NIGHT_GREETING_TIME", "25:00")
        with pytest.raises(ValueError):
            importlib.reload(scheduled_tasks)
        monkeypatch.undo()
        importlib.reload(scheduled_tasks)
        assert scheduled_tasks.SCHEDULED_TASKS == {}
