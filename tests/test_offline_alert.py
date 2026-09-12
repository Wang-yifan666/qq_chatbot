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
    def test_offline_body_contains_required_fields(self):
        t0 = datetime(2026, 9, 12, 7, 10, 0)
        t1 = datetime(2026, 9, 12, 7, 11, 5)
        body = _build_offline_body("3780832187", t0, t1)
        assert "QQ Bot Offline" in body
        assert "Bot ID: 3780832187" in body
        assert "Disconnect Time: 2026-09-12 07:10:00" in body
        assert "Alert Time: 2026-09-12 07:11:05" in body
        assert "Offline Duration: 1m05s" in body
        assert "OneBot connection lost." in body

    def test_recovered_body_contains_required_fields(self):
        t0 = datetime(2026, 9, 12, 7, 10, 0)
        t1 = datetime(2026, 9, 12, 8, 0, 0)
        body = _build_recovered_body("3780832187", t0, t1)
        assert "QQ Bot Recovered" in body
        assert "Recovery Time: 2026-09-12 08:00:00" in body
        assert "Offline Duration: 50m00s" in body
        assert "Bot connection has been restored." in body
