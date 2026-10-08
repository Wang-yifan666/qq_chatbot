"""Offline alert（v0.1 → v0.9.1）：OneBot Bot 连接断开超过宽限期 → SMTP 邮件告警。

设计原则：
- 只监听 NoneBot 的 Bot connect / disconnect 事件（OneBot 反向 WebSocket 层）。
  断开原因不判断：可能是网络抖动、NapCat 重启、NoneBot 重启或 QQ 登录失效，
  因此邮件只描述 "OneBot connection lost"，绝不断言“被腾讯风控踢下线”；
- 断开后进入宽限期（OFFLINE_ALERT_GRACE_SECONDS，默认 60 秒）：
  宽限期内恢复 → 只记日志，不告警；
  超过宽限期仍断开 → 每个 bot 每轮故障发一封 offline 邮件（告警去重）；
- **发送失败必须重试**（v0.9.1，真实事故驱动）：
  2026-09-25 14:15 的离线告警只发了一次、失败了，此后 2 天 7 小时里
  再没有任何通知——因为没有重试，且失败后状态被永久标记为"已告警"。
  现在：失败按 OFFLINE_ALERT_RETRY_DELAYS 重试；重试耗尽后若仍离线，
  每 OFFLINE_ALERT_REALERT_SECONDS 再告警一次，直到送达或恢复连接；
- **投递结果落盘**（OFFLINE_ALERT_LOG_FILE）：同一次事故里 journal 被
  另一个服务刷爆、日志被轮转删除，告警到底发没发、为什么失败完全查不到。
  现在每次尝试都追加一行到独立文件（不含任何凭据），journal 没了也能追溯；
- 已告警的 bot 恢复连接 → 发一封 recovered 邮件并清除告警状态；
  之后再次掉线视为新一轮故障，允许重新告警；
  （**只有离线邮件真正送达过才发恢复邮件**——否则用户只收到一封"已恢复"会很困惑）
- SMTP 全部使用 Python 标准库（smtplib + EmailMessage + ssl），
  通过 asyncio.to_thread 执行，绝不阻塞 event loop；
  任何发送失败只记 ERROR 日志，绝不影响机器人主流程；
- 按 bot_id 分别管理状态（支持多 Bot，虽然当前只有一个 QQ 号）。

配置（.env，均无硬编码默认值之外的秘密）：
    OFFLINE_ALERT_ENABLED=true|false             总开关（默认 false）
    OFFLINE_ALERT_GRACE_SECONDS=60               宽限期（范围 5~3600，默认 60）
    OFFLINE_ALERT_RETRY_DELAYS=60,300,900        发送失败后的重试间隔（秒，最多 5 个）
    OFFLINE_ALERT_REALERT_SECONDS=1800           重试耗尽后仍离线时的再告警间隔
                                                 （0~86400，0 = 不再告警）
    OFFLINE_ALERT_LOG_FILE=data/offline_alert.log 投递记录落盘路径（空 = 不落盘）
    SMTP_HOST=smtp.example.com                   SMTP 服务器
    SMTP_PORT=465                                465=SSL；其它端口走 STARTTLS
    SMTP_USER=bot-alert@example.com              发件账号
    SMTP_PASSWORD=...                            SMTP 授权码（绝不进 Git / 日志）
    ALERT_EMAIL_TO=admin@example.com             收件人
"""

import asyncio
import os
import smtplib
import socket
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
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

# v0.9.1：发送失败后的重试间隔（秒）与"重试耗尽后仍离线"的再告警间隔。
RETRY_DELAYS_DEFAULT = (60, 300, 900)
RETRY_DELAYS_MAX_COUNT = 5
REALERT_SECONDS_DEFAULT = 1800
REALERT_SECONDS_MAX = 86400
ALERT_LOG_FILE_DEFAULT = "data/offline_alert.log"
# 落盘文件的滚动上限（超过就截断重写，避免无限增长）
ALERT_LOG_MAX_BYTES = 64 * 1024

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _env_retry_delays(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """解析逗号分隔的重试间隔（秒）：非法项剔除、超范围回落、最多 5 个。

    允许显式配置成空字符串（= 不重试，退回旧行为）。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return ()
    delays: list[int] = []
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            logger.warning("[OFFLINE ALERT] {} 里的 {} 不是整数，已忽略", name, part)
            continue
        if not (1 <= value <= 86400):
            logger.warning("[OFFLINE ALERT] {} 里的 {} 超出 1~86400，已忽略", name, value)
            continue
        delays.append(value)
    if not delays and raw:
        logger.warning("[OFFLINE ALERT] {}={} 没有可用项，改为不重试", name, raw)
        return ()
    return tuple(delays[:RETRY_DELAYS_MAX_COUNT])


def _resolve_log_file(raw: str) -> Path | None:
    """投递记录文件路径（相对路径按项目根解析）；空字符串 = 不落盘。"""
    text = (raw or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else _PROJECT_ROOT / path


OFFLINE_ALERT_ENABLED = _env_bool("OFFLINE_ALERT_ENABLED", OFFLINE_ALERT_ENABLED_DEFAULT)
GRACE_SECONDS = _env_int(
    "OFFLINE_ALERT_GRACE_SECONDS", GRACE_SECONDS_DEFAULT, GRACE_SECONDS_MIN, GRACE_SECONDS_MAX
)
RETRY_DELAYS = _env_retry_delays("OFFLINE_ALERT_RETRY_DELAYS", RETRY_DELAYS_DEFAULT)
REALERT_SECONDS = _env_int(
    "OFFLINE_ALERT_REALERT_SECONDS", REALERT_SECONDS_DEFAULT, 0, REALERT_SECONDS_MAX
)
ALERT_LOG_FILE = _resolve_log_file(
    os.getenv("OFFLINE_ALERT_LOG_FILE", ALERT_LOG_FILE_DEFAULT)
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


def write_alert_record(
    path: Path | None,
    kind: str,
    ok: bool,
    bot_id: str,
    when: datetime,
    detail: str = "",
) -> None:
    """把一次告警投递结果追加到独立文件（**绝不包含任何凭据 / 邮件正文**）。

    为什么需要它：2026-09-25 的事故里，journal 被另一个服务刷爆、qq-bot 的日志
    被轮转删除，导致"告警到底发没发、为什么失败"完全无法追溯。
    这个文件与 journal 无关，只要磁盘还在就查得到。

    约定：任何写盘失败都被吞掉（告警是旁路能力，绝不能因此影响主流程）。
    """
    if path is None:
        return
    try:
        line = (
            f"{when.strftime('%Y-%m-%d %H:%M:%S')} | {kind:<8} | "
            f"{'OK  ' if ok else 'FAIL'} | bot={bot_id}"
            + (f" | {detail}" if detail else "")
            + "\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > ALERT_LOG_MAX_BYTES:
            # 简单滚动：保留后半段，避免无限增长
            tail = path.read_bytes()[-ALERT_LOG_MAX_BYTES // 2 :]
            path.write_bytes(b"# (truncated)\n" + tail)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:  # pragma: no cover - 取决于磁盘状态
        logger.warning(
            "[OFFLINE ALERT] 投递记录落盘失败（忽略）：{}: {}",
            type(exc).__name__,
            exc,
        )


# ==========================================================================
# 状态机（按 bot_id 隔离；发送函数可注入，便于测试）
# ==========================================================================


@dataclass
class _Outage:
    """一轮故障的运行时状态（on_connect 时整体丢弃）。"""

    disconnected_at: datetime
    alerted: bool = False          # 离线邮件**真正送达**过
    task: asyncio.Task | None = None


class OfflineAlertManager:
    """OneBot 断开告警状态机。

    v0.9.1 起的行为：
        宽限期 → 发送失败则按 retry_delays 重试 →
        重试耗尽仍失败且仍离线 → 每 realert_seconds 再试一次 →
        送达（或恢复连接）为止。

    去重语义保持不变：**同一轮故障内，成功送达的离线邮件只有一封**
    （on_disconnect 重复触发不会重新开一轮）。
    """

    def __init__(
        self,
        grace_seconds: int,
        sender: Callable[[str, str], Awaitable[bool]],
        *,
        retry_delays: tuple[int, ...] = RETRY_DELAYS_DEFAULT,
        realert_seconds: int = REALERT_SECONDS_DEFAULT,
        recorder: Callable[[str, bool, str, datetime, str], None] | None = None,
    ) -> None:
        self.grace_seconds = grace_seconds
        self.retry_delays = tuple(retry_delays or ())
        self.realert_seconds = max(0, int(realert_seconds))
        self._sender = sender
        self._recorder = recorder
        self._outages: dict[str, _Outage] = {}

    def _record(self, kind: str, ok: bool, bot_id: str, when: datetime, detail: str = "") -> None:
        if self._recorder is None:
            return
        try:
            self._recorder(kind, ok, bot_id, when, detail)
        except Exception:  # pragma: no cover - 记录器自身异常绝不能影响告警
            logger.warning("[OFFLINE ALERT] 投递记录写入失败（忽略）")

    async def on_disconnect(self, bot_id: str) -> None:
        if bot_id in self._outages:
            # 同一轮故障内的重复断开事件：保持首次断开时间与正在跑的重试循环。
            logger.info("[OFFLINE ALERT] bot {} still offline (already tracked)", bot_id)
            return
        now = datetime.now()
        logger.info(
            "[OFFLINE ALERT] bot {} disconnected; waiting {} seconds before alert",
            bot_id,
            self.grace_seconds,
        )
        outage = _Outage(disconnected_at=now)
        outage.task = asyncio.create_task(self._run_outage(bot_id, now))
        self._outages[bot_id] = outage

    async def on_connect(self, bot_id: str) -> None:
        now = datetime.now()
        outage = self._outages.pop(bot_id, None)
        if outage is None:
            # 首次连接 / 无告警状态的普通重连：只记日志。
            logger.info("[OFFLINE ALERT] bot {} connected", bot_id)
            return
        if outage.task is not None:
            outage.task.cancel()
        if not outage.alerted:
            # 离线邮件从未送达过 → 不发恢复邮件（否则用户会收到一封"已恢复"而莫名其妙）。
            logger.info(
                "[OFFLINE ALERT] bot {} recovered before offline email was delivered; "
                "no recovery email",
                bot_id,
            )
            return
        logger.info("[OFFLINE ALERT] bot {} recovered after alert; sending recovery email", bot_id)
        ok = await self._sender(
            "[QQ Bot Alert] QQ Bot Recovered",
            _build_recovered_body(bot_id, outage.disconnected_at, now),
        )
        self._record("recovered", bool(ok), bot_id, now)

    async def _run_outage(self, bot_id: str, disconnected_at: datetime) -> None:
        """宽限期 → 首次告警 → 失败重试 → 仍离线则周期性再告警；送达或取消为止。"""
        try:
            await asyncio.sleep(self.grace_seconds)
        except asyncio.CancelledError:
            return

        delays = list(self.retry_delays)
        retry_index = 0   # 已消耗的重试次数（决定下一次等多久）
        total = 0         # 总尝试次数（只用于日志与落盘，单调递增）
        while True:
            total += 1
            when = datetime.now()
            if total == 1:
                logger.info(
                    "[OFFLINE ALERT] bot {} still offline after {}s, sending offline email",
                    bot_id,
                    self.grace_seconds,
                )
            else:
                logger.info("[OFFLINE ALERT] bot {} offline email attempt #{}", bot_id, total)
            try:
                ok = bool(
                    await self._sender(
                        "[QQ Bot Alert] QQ Bot 离线",
                        _build_offline_body(bot_id, disconnected_at, when),
                    )
                )
            except Exception as exc:  # 发送函数自身抛异常也要进入重试，而不是静默结束
                ok = False
                logger.error(
                    "[OFFLINE ALERT] offline email raised {}: {}",
                    type(exc).__name__,
                    str(exc)[:200],
                )
            outage = self._outages.get(bot_id)
            if outage is None:
                return  # 发送期间已恢复并清理
            if ok:
                outage.alerted = True
                self._record("offline", True, bot_id, when, detail=f"attempt={total}")
                logger.info(
                    "[OFFLINE ALERT] bot {} offline email delivered (attempt {})",
                    bot_id,
                    total,
                )
                return
            self._record("offline", False, bot_id, when, detail=f"attempt={total}")
            logger.error(
                "[OFFLINE ALERT] bot {} offline email FAILED (attempt {}/{})",
                bot_id,
                total,
                1 + len(delays),
            )

            if retry_index < len(delays):
                wait = delays[retry_index]
                retry_index += 1
            elif self.realert_seconds > 0:
                wait = self.realert_seconds
                logger.warning(
                    "[OFFLINE ALERT] bot {} 离线邮件重试耗尽，{}s 后再试一次",
                    bot_id,
                    wait,
                )
            else:
                logger.error(
                    "[OFFLINE ALERT] bot {} 离线邮件重试耗尽且未配置再告警，放弃本轮",
                    bot_id,
                )
                return
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                return

    async def shutdown(self) -> None:
        """进程退出：取消所有正在跑的告警 / 重试任务。"""
        for outage in list(self._outages.values()):
            if outage.task is not None:
                outage.task.cancel()
        self._outages.clear()


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

    def _recorder(kind: str, ok: bool, bot_id: str, when: datetime, detail: str = "") -> None:
        write_alert_record(ALERT_LOG_FILE, kind, ok, bot_id, when, detail)

    _manager = OfflineAlertManager(
        GRACE_SECONDS,
        _send_email_async,
        retry_delays=RETRY_DELAYS,
        realert_seconds=REALERT_SECONDS,
        recorder=_recorder,
    )
    driver.on_bot_connect(_on_bot_connect)
    driver.on_bot_disconnect(_on_bot_disconnect)
    driver.on_shutdown(_on_shutdown)
    logger.info(
        "[OFFLINE ALERT] 已启用：grace={}s retry={} realert={}s log={} host={} port={} to={}",
        GRACE_SECONDS,
        list(RETRY_DELAYS) or "无",
        REALERT_SECONDS,
        ALERT_LOG_FILE or "不落盘",
        SMTP_HOST,
        SMTP_PORT,
        ALERT_EMAIL_TO,
    )
