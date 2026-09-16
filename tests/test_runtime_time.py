"""时间语义测试（v0.6）：24 小时制 / day_period 边界 / 四模式同源 runtime state。

重点断言（本版本修复的核心）：
- 02:30 → 凌晨，绝不能是下午；
- 12:30 → 中午，绝不能是凌晨；
- 13:30 → 下午；
- DIRECT / AMBIENT / SCHEDULED / POKE 四种模式看到的是同一时间语义；
- scheduled task 的 HH:MM 继续按 24 小时制解析（00:00 / 08:00 / 12:00 / 23:59 合法，
  24:00 继续非法）。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from services.prompt_builder import CurrentUser
from services.prompt_builder import ScheduledEvent
from services.prompt_builder import build_messages
from services.runtime_context import build_runtime_state
from services.runtime_context import classify_day_period
from services.runtime_context import to_12h
from services.scheduled_tasks import parse_task_time

TZ = ZoneInfo("Asia/Shanghai")


class TestClassifyDayPeriod:
    """day_period 边界：由程序集中计算，边界值与需求文档逐条对应。"""

    @pytest.mark.parametrize(
        "hour,minute,expected",
        [
            (0, 0, "凌晨"),
            (0, 30, "凌晨"),
            (2, 30, "凌晨"),
            (4, 59, "凌晨"),
            (5, 0, "早上"),
            (8, 59, "早上"),
            (9, 0, "上午"),
            (11, 59, "上午"),
            (12, 0, "中午"),
            (12, 30, "中午"),
            (12, 59, "中午"),
            (13, 0, "下午"),
            (13, 30, "下午"),
            (17, 59, "下午"),
            (18, 0, "晚上"),
            (22, 59, "晚上"),
            (23, 0, "深夜"),
            (23, 59, "深夜"),
        ],
    )
    def test_boundaries(self, hour, minute, expected):
        assert classify_day_period(hour) == expected

    def test_0230_is_not_afternoon(self):
        assert classify_day_period(2) == "凌晨"
        assert classify_day_period(2) != "下午"

    def test_1230_is_noon_not_midnight(self):
        assert classify_day_period(12) == "中午"
        assert classify_day_period(12) != "凌晨"

    def test_1330_is_afternoon(self):
        assert classify_day_period(13) == "下午"


class TestTo12h:
    def test_midnight_and_noon_conventions(self):
        # 00:00 = 12:00 AM；12:00 = 12:00 PM（明确约定，绝不混淆）
        assert to_12h(0) == (12, "AM")
        assert to_12h(12) == (12, "PM")
        assert to_12h(13) == (1, "PM")
        assert to_12h(2) == (2, "AM")
        assert to_12h(18) == (6, "PM")
        assert to_12h(23) == (11, "PM")


class TestBuildRuntimeState:
    def _now(self, hour: int, minute: int, second: int = 0) -> datetime:
        return datetime(2025, 1, 1, hour, minute, second, tzinfo=TZ)

    def test_0230_machine_readable_fields(self):
        state = build_runtime_state(self._now(2, 30, 15))
        assert "time_24h: 02:30:15" in state
        assert "hour_24: 2" in state
        assert "minute: 30" in state
        assert "day_period: 凌晨" in state
        assert "hour_12: 2" in state
        assert "meridiem: AM" in state
        assert "timezone: Asia/Shanghai" in state
        assert "datetime: 2025-01-01 02:30:15" in state
        assert "now_epoch:" in state

    def test_1230_is_noon_in_runtime_state(self):
        state = build_runtime_state(self._now(12, 30))
        assert "hour_24: 12" in state
        assert "day_period: 中午" in state
        assert "hour_12: 12" in state
        assert "meridiem: PM" in state

    def test_1330_is_afternoon_in_runtime_state(self):
        state = build_runtime_state(self._now(13, 30))
        assert "day_period: 下午" in state
        assert "hour_12: 1" in state
        assert "meridiem: PM" in state

    def test_24h_rules_are_spelled_out(self):
        state = build_runtime_state(self._now(2, 30))
        # 24 小时制规则必须显式写进 runtime state（程序告诉模型，不让模型猜）
        assert "24 小时制" in state
        assert "00:xx 表示午夜之后的凌晨" in state
        assert "02:xx 表示凌晨 2 点" in state
        assert "12:xx 表示中午 12 点" in state
        assert "13:xx 表示下午 1 点" in state
        assert "18:xx 表示晚上 6 点" in state
        assert "day_period" in state
        assert "不允许自行重新推断 AM / PM" in state


class TestAllModesShareSameRuntimeState:
    """DIRECT / AMBIENT / SCHEDULED / POKE 必须看到同一份时间语义。"""

    def _fixed_state(self) -> str:
        return build_runtime_state(datetime(2025, 1, 1, 2, 30, 15, tzinfo=TZ))

    def test_four_modes_see_same_time_semantics(self):
        runtime = self._fixed_state()
        direct = build_messages(
            CurrentUser(user_id=1, display_name="小明"),
            "stranger",
            [],
            [],
            "现在几点",
            runtime_state=runtime,
        )
        ambient = build_messages(
            None, "stranger", [], [], "", runtime_state=runtime,
            conversation_mode="ambient", ambient_context="触发片段",
        )
        scheduled = build_messages(
            None, "stranger", [], [], "", runtime_state=runtime,
            conversation_mode="scheduled",
            scheduled_event=ScheduledEvent("morning_greeting", "2025-01-01 08:00:00", "08:00"),
        )
        poke = build_messages(
            CurrentUser(user_id=1, display_name="小明"),
            "stranger",
            [],
            [],
            "",
            runtime_state=runtime,
            conversation_mode="poke",
            poke_back=True,
        )
        for label, messages in (
            ("direct", direct),
            ("ambient", ambient),
            ("scheduled", scheduled),
            ("poke", poke),
        ):
            system = messages[0]["content"]
            assert runtime in system, f"{label} 模式必须包含同一份 runtime state"
            assert "day_period: 凌晨" in system, f"{label} 模式的时段语义必须与程序一致"
            assert "time_24h: 02:30:15" in system, f"{label} 模式的时间语义必须与程序一致"

    def test_poke_mode_never_claims_web_search(self, monkeypatch):
        import services.prompt_builder as pb

        monkeypatch.setattr(pb, "TOOLS", [{"type": "function"}])
        poke = build_messages(
            CurrentUser(user_id=1, display_name="小明"),
            "stranger",
            [],
            [],
            "",
            runtime_state="RUNTIME",
            conversation_mode="poke",
        )[0]["content"]
        assert "web_search: false" in poke
        assert "web_search: true" not in poke


class TestScheduledTaskTime24h:
    """scheduled task 的 HH:MM 继续按 24 小时制解析；24:00 继续非法。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("00:00", (0, 0)),
            ("08:00", (8, 0)),
            ("12:00", (12, 0)),
            ("23:59", (23, 59)),
        ],
    )
    def test_valid_24h_times(self, raw, expected):
        assert parse_task_time("X_TIME", raw) == expected

    def test_2400_still_illegal(self):
        with pytest.raises(ValueError):
            parse_task_time("X_TIME", "24:00")

    @pytest.mark.parametrize("raw", ["24:01", "08:60", "13:5", "8点"])
    def test_other_illegal_forms(self, raw):
        with pytest.raises(ValueError):
            parse_task_time("X_TIME", raw)


class TestTimezoneResolution:
    """BOT_TIMEZONE 解析与 fallback 日志（v0.6）。"""

    def test_unset_defaults_to_asia_shanghai(self, monkeypatch):
        import services.runtime_context as rc

        monkeypatch.delenv("BOT_TIMEZONE", raising=False)
        assert rc.get_timezone() == "Asia/Shanghai"

    def test_valid_zone_used_as_is(self, monkeypatch):
        import services.runtime_context as rc

        monkeypatch.setenv("BOT_TIMEZONE", "UTC")
        assert rc.get_timezone() == "UTC"

    def test_invalid_zone_falls_back_to_default(self, monkeypatch):
        import services.runtime_context as rc

        monkeypatch.setenv("BOT_TIMEZONE", "Not/AZone")
        assert rc.get_timezone() == "Asia/Shanghai"

    def test_fallback_log_contains_configured_and_actual(self, monkeypatch):
        """发生 fallback 时日志必须同时给出 configured_timezone 与 actual_timezone，
        让“配置 Asia/Shanghai 实际跑在 UTC”这类隐蔽错误一眼可见。"""
        import services.runtime_context as rc

        logs: list[str] = []

        class FakeLogger:
            def warning(self, msg, *args, **kwargs):
                logs.append(msg.format(*args) if args else msg)

            def info(self, *args, **kwargs):
                pass

        monkeypatch.setattr(rc, "logger", FakeLogger())
        monkeypatch.setenv("BOT_TIMEZONE", "Not/AZone")
        assert rc.get_timezone() == "Asia/Shanghai"
        assert logs, "fallback 必须产生告警日志"
        assert "configured_timezone=Not/AZone" in logs[0]
        assert "actual_timezone=Asia/Shanghai" in logs[0]
        assert "fallback" in logs[0]

    def test_valid_zone_logs_configured_and_actual(self, monkeypatch):
        import services.runtime_context as rc

        logs: list[str] = []

        class FakeLogger:
            def info(self, msg, *args, **kwargs):
                logs.append(msg.format(*args) if args else msg)

            def warning(self, *args, **kwargs):
                pass

        monkeypatch.setattr(rc, "logger", FakeLogger())
        monkeypatch.setenv("BOT_TIMEZONE", "Asia/Shanghai")
        rc.get_timezone()
        assert any(
            "configured_timezone=Asia/Shanghai" in msg
            and "actual_timezone=Asia/Shanghai" in msg
            for msg in logs
        )
