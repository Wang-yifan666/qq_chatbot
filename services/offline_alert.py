"""Offline alert（v0.1）：OneBot Bot 连接断开超过宽限期 → SMTP 邮件告警。

设计原则：
- 只监听 NoneBot 的 Bot connect / disconnect 事件（OneBot 反向 WebSocket 层）。
  断开原因不判断：可能是网络抖动、NapCat 重启、NoneBot 重启或 QQ 登录失效，
  因此邮件只描述 "OneBot connection lost"，绝不断言“被腾讯风控踢下线”；
- 断开后进入宽限期（OFFLINE_ALERT_GRACE_SECONDS，默认 60 秒）：
  宽限期内恢复 → 只记日志，不告警；
  超过宽限期仍断开 → 每个 bot 每轮故障只发一封 offline 邮件（告警去重）；
- 已告警的 bot 恢复连接 → 发一封 recovered 邮件并清除告警状态；
  之后再次掉线视为新一轮故障，允许重新告警；
- SMTP 全部使用 Python 标准库（smtplib + EmailMessage + ssl），
  通过 asyncio.to_thread 执行，绝不阻塞 event loop；
  任何发送失败只记 ERROR 日志，绝不影响机器人主流程；
- 按 bot_id 分别管理状态（支持多 Bot，虽然当前只有一个 QQ 号）。

配置（.env，均无硬编码默认值之外的秘密）：
    OFFLINE_ALERT_ENABLED=true|false          总开关（默认 false）
    OFFLINE_ALERT_GRACE_SECONDS=60            宽限期（范围 5~3600，默认 60）
    SMTP_HOST=smtp.example.com                SMTP 服务器
    SMTP_PORT=465                             465=SSL；其它端口走 STARTTLS
    SMTP_USER=bot-alert@example.com           发件账号
    SMTP_PASSWORD=...                         SMTP 授权码（绝不进 Git / 日志）
    ALERT_EMAIL_TO=admin@example.com          收件人
"""

import asyncio
import os
import smtplib
import socket
import ssl
from datetime import datetime
from email.message import EmailMessage
from typing import Awaitable
from typing import Callable

from nonebot import logger

# ==========================================================================
# 配置（进程启动时解析一次；缺失关键项时 setup 阶段报错并保持关闭）
# ==========================================================================

OFFLINE_ALERT_ENABLED_DEFAULT = False
GRACE_SECONDS_DEFAULT = 60
GRACE_SECONDS_MIN = 5
GRACE_SECONDS_MAX = 3600
SMTP_TIMEOUT_SECONDS = 30.0


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("[OFFLINE ALERT] {}={} 不是合法布尔值，按 {} 处理", name, raw, default)
    return default


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[OFFLINE ALERT] {}={} 不是合法整数，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[OFFLINE ALERT] {}={} 超出范围 [{}, {}]，使用默认 {}", name, value, low, high, default)
        return default
    return value


OFFLINE_ALERT_ENABLED = _env_bool("OFFLINE_ALERT_ENABLED", OFFLINE_ALERT_ENABLED_DEFAULT)
GRACE_SECONDS = _env_int(
    "OFFLINE_ALERT_GRACE_SECONDS", GRACE_SECONDS_DEFAULT, GRACE_SECONDS_MIN, GRACE_SECONDS_MAX
)
SMTP_HOST = (os.getenv("SMTP_HOST") or "").strip()
SMTP_PORT = _env_int("SMTP_PORT", 465, 1, 65535)
SMTP_USER = (os.getenv("SMTP_USER") or "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or ""
ALERT_EMAIL_TO = (os.getenv("ALERT_EMAIL_TO") or "").strip()


def _config_complete() -> bool:
    """SMTP 配置完整性检查（不输出任何配置值，只报缺了哪一项）。"""
    missing = []
    if not SMTP_HOST:
        missing.append("SMTP_HOST")
    if not SMTP_USER:
        missing.append("SMTP_USER")
    if not SMTP_PASSWORD:
        missing.append("SMTP_PASSWORD")
    if not ALERT_EMAIL_TO:
        missing.append("ALERT_EMAIL_TO")
    if missing:
        logger.error(
            "[OFFLINE ALERT] SMTP 配置不完整（缺少 {}），离线邮件告警保持关闭；"
            "请补齐 .env 后重启",
            " / ".join(missing),
        )
        return False
    return True


# ==========================================================================
# 邮件构造与发送（同步，运行在 to_thread 里）
# ==========================================================================


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_offline_body(bot_id: str, disconnected_at: datetime, alerted_at: datetime) -> str:
    hostname = socket.gethostname()
    duration = (alerted_at - disconnected_at).total_seconds()
    return "\n".join(
        [
            "QQ Bot Offline",
            "",
            f"Bot ID: {bot_id}",
            f"Hostname: {hostname}",
            f"Disconnect Time: {_fmt(disconnected_at)}",
            f"Alert Time: {_fmt(alerted_at)}",
            f"Offline Duration: {_fmt_duration(duration)}",
            f"Grace Period: {GRACE_SECONDS}s",
            "",
            "Status:",
            "OneBot connection lost.",
            "",
            "Please check NapCat / QQ login status.",
        ]
    )


def _build_recovered_body(bot_id: str, disconnected_at: datetime, recovered_at: datetime) -> str:
    hostname = socket.gethostname()
    duration = (recovered_at - disconnected_at).total_seconds()
    return "\n".join(
        [
            "QQ Bot Recovered",
            "",
            f"Bot ID: {bot_id}",
            f"Hostname: {hostname}",
            f"Recovery Time: {_fmt(recovered_at)}",
            f"Offline Duration: {_fmt_duration(duration)}",
            "",
            "Bot connection has been restored.",
        ]
    )


def _send_email_sync(subject: str, body: str) -> None:
    """同步发送邮件（标准库；调用方负责放入 to_thread，带 timeout 与异常兜底）。

    注意：绝不把 SMTP_PASSWORD 写进任何日志 / 异常消息。
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL_TO
    msg.set_content(body)

    context = ssl.create_default_context()
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS, context=context) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.ehlo()
            if server.has_extn("starttls"):
                server.starttls(context=context)
                server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)


async def _send_email_async(subject: str, body: str) -> bool:
    """异步发送：to_thread + 异常兜底；返回是否成功（只用于日志）。"""
    try:
        await asyncio.to_thread(_send_email_sync, subject, body)
        logger.info("[OFFLINE ALERT] email sent subject={}", subject)
        return True
    except Exception as exc:
        # exc 里可能包含服务器返回信息，但绝不包含密码；仍只输出异常类型与摘要。
        logger.error(
            "[OFFLINE ALERT] email send failed subject={} error={}: {}",
            subject,
            type(exc).__name__,
            str(exc)[:200],
        )
        return False


# ==========================================================================
# 状态机（按 bot_id 隔离；发送函数可注入，便于测试）
# ==========================================================================


class OfflineAlertManager:
    """OneBot 断开告警状态机：宽限期 → 单次 offline 邮件 → 恢复 → recovered 邮件。"""

    def __init__(
        self,
        grace_seconds: int,
        sender: Callable[[str, str], Awaitable[bool]],
    ) -> None:
        self.grace_seconds = grace_seconds
        self._sender = sender
        self._pending: dict[str, asyncio.Task] = {}
        self._alerted: set[str] = set()
        self._disconnect_times: dict[str, datetime] = {}

    async def on_disconnect(self, bot_id: str) -> None:
        now = datetime.now()
        # 已告警的 bot 继续断开：保持首轮告警状态，不重复发信、不重置计时。
        if bot_id in self._alerted:
            logger.info("[OFFLINE ALERT] bot {} still offline (already alerted)", bot_id)
            return
        if bot_id in self._pending:
            # 宽限期内再次断开（理论上不应发生）：保持首次断开时间与计时器。
            logger.info("[OFFLINE ALERT] bot {} disconnected again during grace period", bot_id)
            return
        self._disconnect_times[bot_id] = now
        logger.info(
            "[OFFLINE ALERT] bot {} disconnected; waiting {} seconds before alert",
            bot_id,
            self.grace_seconds,
        )
        self._pending[bot_id] = asyncio.create_task(self._grace_elapsed(bot_id, now))

    async def on_connect(self, bot_id: str) -> None:
        now = datetime.now()
        pending = self._pending.pop(bot_id, None)
        if pending is not None:
            pending.cancel()
            self._disconnect_times.pop(bot_id, None)
            logger.info("[OFFLINE ALERT] bot {} recovered during grace period, no alert", bot_id)
            return
        if bot_id in self._alerted:
            disconnected_at = self._disconnect_times.pop(bot_id, now)
            self._alerted.discard(bot_id)
            logger.info("[OFFLINE ALERT] bot {} recovered after alert; sending recovery email", bot_id)
            await self._sender(
                "[QQ Bot Alert] QQ Bot Recovered",
                _build_recovered_body(bot_id, disconnected_at, now),
            )
            return
        # 首次连接 / 无告警状态的普通重连：只记日志。
        logger.info("[OFFLINE ALERT] bot {} connected", bot_id)

    async def _grace_elapsed(self, bot_id: str, disconnected_at: datetime) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
        except asyncio.CancelledError:
            return
        self._pending.pop(bot_id, None)
        self._alerted.add(bot_id)
        alerted_at = datetime.now()
        logger.info(
            "[OFFLINE ALERT] bot {} still offline after {}s, sending offline email",
            bot_id,
            self.grace_seconds,
        )
        await self._sender(
            "[QQ Bot Alert] QQ Bot 离线",
            _build_offline_body(bot_id, disconnected_at, alerted_at),
        )

    async def shutdown(self) -> None:
        """进程退出：取消所有宽限期计时任务。"""
        for task in list(self._pending.values()):
            task.cancel()
        self._pending.clear()


# ==========================================================================
# 全局单例与 NoneBot 注册（bot.py 调用）
# ==========================================================================

_manager: OfflineAlertManager | None = None


def _hook_bot_id(bot) -> str:
    return str(getattr(bot, "self_id", "") or "unknown")


async def _on_bot_disconnect(bot) -> None:
    if _manager is not None:
        await _manager.on_disconnect(_hook_bot_id(bot))


async def _on_bot_connect(bot) -> None:
    if _manager is not None:
        await _manager.on_connect(_hook_bot_id(bot))


async def _on_shutdown() -> None:
    if _manager is not None:
        await _manager.shutdown()


def setup_offline_alert(driver) -> None:
    """在 bot.py 启动阶段调用：注册 Bot connect / disconnect 告警钩子。

    - OFFLINE_ALERT_ENABLED=false → 只记日志，不注册任何钩子；
    - SMTP 配置不完整 → ERROR 日志，不注册（fail-closed，绝不半开）。
    """
    global _manager
    if not OFFLINE_ALERT_ENABLED:
        logger.info("[OFFLINE ALERT] OFFLINE_ALERT_ENABLED=false，离线邮件告警未启用")
        return
    if not _config_complete():
        return
    _manager = OfflineAlertManager(GRACE_SECONDS, _send_email_async)
    driver.on_bot_connect(_on_bot_connect)
    driver.on_bot_disconnect(_on_bot_disconnect)
    driver.on_shutdown(_on_shutdown)
    logger.info(
        "[OFFLINE ALERT] 已启用：grace={}s host={} port={} to={}",
        GRACE_SECONDS,
        SMTP_HOST,
        SMTP_PORT,
        ALERT_EMAIL_TO,
    )
