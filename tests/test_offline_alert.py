"""services/offline_alert.py：断开告警状态机测试（v0.1）。

覆盖需求中的五种场景：
1. 宽限期内恢复 → 不发任何邮件；
2. 超过宽限期仍断开 → 只发一封 offline 邮件；
3. 持续断开半小时级别 → 仍然只有一封（告警去重）；
4. 告警后恢复 → 发一封 recovered 邮件并清除状态；
5. 恢复后再次掉线 → 视为新一轮故障，允许再次告警。

发送函数注入假 sender，宽限期用 0.1s，测试秒级完成。
"""

import asyncio
from datetime import datetime

import pytest

from services.offline_alert import OfflineAlertManager
from services.offline_alert import _build_offline_body
from services.offline_alert import _build_recovered_body


class FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def __call__(self, subject: str, body: str) -> bool:
        self.sent.append((subject, body))
        return True


def _make_manager() -> tuple[OfflineAlertManager, FakeSender]:
    sender = FakeSender()
    return OfflineAlertManager(grace_seconds=0.1, sender=sender), sender


class TestGracePeriod:
    async def test_recovery_within_grace_sends_nothing(self):
        manager, sender = _make_manager()
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.05)  # < grace
        await manager.on_connect("10001")
        await asyncio.sleep(0.2)  # 若计时器没被取消，这里会触发发送
        assert sender.sent == []
        await manager.shutdown()

    async def test_disconnect_beyond_grace_sends_single_offline(self):
        manager, sender = _make_manager()
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)  # > grace
        assert len(sender.sent) == 1
        assert sender.sent[0][0] == "[QQ Bot Alert] QQ Bot 离线"
        assert "OneBot connection lost." in sender.sent[0][1]
        await manager.shutdown()

    async def test_long_outage_sends_only_once(self):
        manager, sender = _make_manager()
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        # 模拟持续断开半小时：多次 disconnect 事件 + 时间流逝，都不得再发
        for _ in range(5):
            await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        assert len(sender.sent) == 1
        await manager.shutdown()

    async def test_recovery_after_alert_sends_recovered_email(self):
        manager, sender = _make_manager()
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        assert len(sender.sent) == 1
        await manager.on_connect("10001")
        assert len(sender.sent) == 2
        assert sender.sent[1][0] == "[QQ Bot Alert] QQ Bot Recovered"
        assert "Bot connection has been restored." in sender.sent[1][1]
        await manager.shutdown()

    async def test_new_round_after_recovery_alerts_again(self):
        manager, sender = _make_manager()
        # 第一轮
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        await manager.on_connect("10001")
        assert len(sender.sent) == 2
        # 第二轮：应允许再次告警
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        assert len(sender.sent) == 3
        assert sender.sent[2][0] == "[QQ Bot Alert] QQ Bot 离线"
        await manager.shutdown()

    async def test_bots_are_isolated(self):
        manager, sender = _make_manager()
        await manager.on_disconnect("10001")
        await manager.on_disconnect("10002")
        await asyncio.sleep(0.3)
        # 两个 bot 各发一封
        assert len(sender.sent) == 2
        # 只有 10001 恢复 → 只有一封 recovered
        await manager.on_connect("10001")
        assert len(sender.sent) == 3
        await manager.shutdown()


class TestEmailBodies:
    # 刻意用假号：真实机器人 QQ 号不该出现在公开仓库里
    # （推送前 scripts/scan_privacy.py 会拦下来）
    FAKE_BOT_ID = "1000000001"

    def test_offline_body_contains_required_fields(self):
        t0 = datetime(2026, 9, 12, 7, 10, 0)
        t1 = datetime(2026, 9, 12, 7, 11, 5)
        body = _build_offline_body(self.FAKE_BOT_ID, t0, t1)
        assert "QQ Bot Offline" in body
        assert f"Bot ID: {self.FAKE_BOT_ID}" in body
        assert "Disconnect Time: 2026-09-12 07:10:00" in body
        assert "Alert Time: 2026-09-12 07:11:05" in body
        assert "Offline Duration: 1m05s" in body
        assert "OneBot connection lost." in body

    def test_recovered_body_contains_required_fields(self):
        t0 = datetime(2026, 9, 12, 7, 10, 0)
        t1 = datetime(2026, 9, 12, 8, 0, 0)
        body = _build_recovered_body(self.FAKE_BOT_ID, t0, t1)
        assert "QQ Bot Recovered" in body
        assert "Recovery Time: 2026-09-12 08:00:00" in body
        assert "Offline Duration: 50m00s" in body
        assert "Bot connection has been restored." in body


# ======================================================================
# v0.9.1 回归：发送失败必须重试（2026-09-25 真实事故）
# ======================================================================

# 事故经过：09-25 14:14 连接断开 → 14:15 触发离线告警 → **发送失败** →
# 因为"没有重试 + 失败后状态被永久标记已告警"，接下来 2 天 7 小时再没有任何通知。
# 用户只收到 09-27 21:31 的恢复邮件（因为 _alerted 还在，走了 recovered 分支）。
# 下面把这些行为钉死。


class FlakySender:
    """先失败 fail_times 次，之后成功；记录全部尝试（含失败）。"""

    def __init__(self, fail_times: int = 0, raise_times: int = 0) -> None:
        self.fail_times = fail_times
        self.raise_times = raise_times
        self.attempts: list[str] = []
        self.delivered: list[tuple[str, str]] = []

    async def __call__(self, subject: str, body: str) -> bool:
        self.attempts.append(subject)
        n = len(self.attempts)
        if n <= self.raise_times:
            raise RuntimeError("SMTP 炸了")
        if n <= self.raise_times + self.fail_times:
            return False
        self.delivered.append((subject, body))
        return True


def _make_flaky(
    sender: FlakySender,
    *,
    grace: float = 0.05,
    retry_delays: tuple[int, ...] = (0, 0, 0),
    realert_seconds: int = 0,
    recorder=None,
):
    # 说明：retry/realert 的秒数在构造器里被 int() 归一，测试里用 0 表示"立刻"，
    # 因此这几条用例都在百毫秒级完成。
    return OfflineAlertManager(
        grace_seconds=grace,
        sender=sender,
        retry_delays=retry_delays,
        realert_seconds=realert_seconds,
        recorder=recorder,
    )


class TestSendRetry:
    async def test_transient_failure_is_retried_until_delivered(self):
        """偶发失败不再等于永久静默 —— 这就是 09-25 缺的那一环。"""
        sender = FlakySender(fail_times=2)
        manager = _make_flaky(sender, retry_delays=(0, 0, 0))
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.4)
        assert len(sender.attempts) == 3, "应尝试 3 次（首次 + 2 次重试）"
        assert len(sender.delivered) == 1, "最终只投递一封"
        assert sender.delivered[0][0] == "[QQ Bot Alert] QQ Bot 离线"
        await manager.shutdown()

    async def test_delivered_alert_then_recovery_sends_recovered(self):
        sender = FlakySender(fail_times=1)
        manager = _make_flaky(sender, retry_delays=(0,))
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        await manager.on_connect("10001")
        assert [s for s, _ in sender.delivered] == [
            "[QQ Bot Alert] QQ Bot 离线",
            "[QQ Bot Alert] QQ Bot Recovered",
        ]
        await manager.shutdown()

    async def test_exception_from_sender_is_retried_too(self):
        """发送函数抛异常也必须进重试，而不是静默结束。"""
        sender = FlakySender(raise_times=2)
        manager = _make_flaky(sender, retry_delays=(0, 0, 0))
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.4)
        assert len(sender.attempts) == 3
        assert len(sender.delivered) == 1
        await manager.shutdown()

    async def test_retries_exhausted_without_realert_gives_up(self):
        sender = FlakySender(fail_times=99)
        manager = _make_flaky(sender, retry_delays=(0, 0), realert_seconds=0)
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.4)
        assert len(sender.attempts) == 3, "首次 + 2 次重试后放弃"
        await asyncio.sleep(0.2)
        assert len(sender.attempts) == 3, "放弃后不应继续尝试"
        await manager.shutdown()

    async def test_realert_keeps_trying_while_still_offline(self):
        """重试耗尽且仍离线 → 周期性再告警（这里用 1 秒间隔验证）。

        时序：grace(0.05s) → 尝试#1 失败 → 重试#2 失败（delay=0）
              → 重试耗尽 → 等 1s → 尝试#3 成功
        """
        sender = FlakySender(fail_times=2)
        manager = _make_flaky(sender, retry_delays=(0,), realert_seconds=1)
        await manager.on_disconnect("10001")
        await asyncio.sleep(1.5)
        assert len(sender.attempts) == 3, f"首次 + 1 次重试 + 1 次再告警，实际 {len(sender.attempts)}"
        assert len(sender.delivered) == 1
        await manager.shutdown()

    async def test_attempt_counter_is_monotonic_in_records(self):
        """落盘记录里的 attempt 必须单调递增（便于事后复盘重试过程）。"""
        records: list[str] = []

        def recorder(kind, ok, bot_id, when, detail=""):
            records.append(f"{kind}:{ok}:{detail}")

        sender = FlakySender(fail_times=2)
        manager = _make_flaky(sender, retry_delays=(0,), realert_seconds=1, recorder=recorder)
        await manager.on_disconnect("10001")
        await asyncio.sleep(1.5)
        assert records == [
            "offline:False:attempt=1",
            "offline:False:attempt=2",
            "offline:True:attempt=3",
        ]
        await manager.shutdown()

    async def test_recovery_before_delivery_sends_no_recovery_email(self):
        """离线邮件从没送达 → 恢复时不该发"已恢复"（否则用户会莫名其妙）。"""
        sender = FlakySender(fail_times=99)
        manager = _make_flaky(sender, retry_delays=(5,))
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.2)          # 首次尝试已失败，正在等 5 秒重试
        await manager.on_connect("10001")  # 恢复 → 取消重试
        await asyncio.sleep(0.2)
        assert sender.delivered == []
        await manager.shutdown()

    async def test_first_connect_logs_no_recovery_email(self):
        sender = FlakySender()
        manager = _make_flaky(sender)
        await manager.on_connect("10001")  # 从未断开过
        assert sender.delivered == []
        await manager.shutdown()


class TestDeliveryRecord:
    async def test_recorder_sees_failure_then_success(self):
        records: list[tuple] = []

        def recorder(kind, ok, bot_id, when, detail=""):
            records.append((kind, ok, bot_id, detail))

        sender = FlakySender(fail_times=1)
        manager = _make_flaky(sender, retry_delays=(0,), recorder=recorder)
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        kinds = [(k, ok) for k, ok, _b, _d in records]
        assert kinds == [("offline", False), ("offline", True)]
        await manager.shutdown()

    async def test_recorder_exception_does_not_break_alerting(self):
        def bad_recorder(*_args, **_kwargs):
            raise RuntimeError("磁盘满了")

        sender = FlakySender()
        manager = _make_flaky(sender, retry_delays=(), recorder=bad_recorder)
        await manager.on_disconnect("10001")
        await asyncio.sleep(0.3)
        assert len(sender.delivered) == 1, "记录失败绝不能让告警失败"
        await manager.shutdown()

    def test_write_alert_record_writes_line_without_secrets(self, tmp_path):
        from services.offline_alert import write_alert_record

        log = tmp_path / "sub" / "offline_alert.log"
        write_alert_record(log, "offline", False, "10001", datetime(2026, 9, 25, 14, 15, 4), "attempt=1")
        write_alert_record(log, "offline", True, "10001", datetime(2026, 9, 25, 14, 20, 4))
        text = log.read_text(encoding="utf-8")
        assert "2026-09-25 14:15:04 | offline  | FAIL | bot=10001 | attempt=1" in text
        assert "| OK   | bot=10001" in text
        assert "PASSWORD" not in text.upper()

    def test_write_alert_record_handles_none_path(self):
        from services.offline_alert import write_alert_record

        write_alert_record(None, "offline", True, "10001", datetime.now())  # 不应抛异常

    def test_retry_delays_env_parsing(self, monkeypatch):
        from services import offline_alert as oa

        monkeypatch.setenv("OFFLINE_ALERT_RETRY_DELAYS", "10, 20 ,bad,999999,30")
        assert oa._env_retry_delays("OFFLINE_ALERT_RETRY_DELAYS", (1,)) == (10, 20, 30)
        monkeypatch.setenv("OFFLINE_ALERT_RETRY_DELAYS", "")
        assert oa._env_retry_delays("OFFLINE_ALERT_RETRY_DELAYS", (1,)) == ()
        monkeypatch.delenv("OFFLINE_ALERT_RETRY_DELAYS")
        assert oa._env_retry_delays("OFFLINE_ALERT_RETRY_DELAYS", (7,)) == (7,)
