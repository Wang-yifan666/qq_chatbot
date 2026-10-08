#!/usr/bin/env python3
"""树莓派侧独立看门狗（v1）：**保证**机器人在线状态被检测到，并邮件通知。

## 为什么需要它（两次真实事故）

| 事故 | bot 内置告警为什么没救到你 |
| --- | --- |
| 2026-09-21 | NapCat 登录失效，bot 进程起来了但**从未连接** → 没有 disconnect 事件 → 告警根本没被激活 |
| 2026-09-25 | 告警触发了、发出去了、**发送失败** → 当时没有重试 → 静默 2 天 7 小时 |

结论：**"保证能通知到"不能建立在被监控对象自己的代码上。**
本脚本由 systemd timer 独立驱动，只依赖标准库，不 import 项目的任何模块；
即使 bot 进程挂了、卡死、或根本没启动，它照样能发现并发邮件。

正文一律**纯文本**，不含任何 markdown 记号（`**粗体**`、`# 标题` 在纯文本邮件里
只会显示成字面星号/井号）；每封都带一个 `.txt` 日志附件（看门狗日志、napcat /
qq-bot 服务日志、NetworkManager 网络事件、bot 侧投递记录），正文里再放一小段摘要。

## 判据（四条全满足才算在线）

1. `qq-bot.service` active；
2. 树莓派本身能访问外网（TCP 探测固定 IP，见 `INTERNET_PROBES`）；
3. NapCat WebUI `CheckLoginStatus` 返回 `isLogin: true`（QQ 已登录）；
4. 本地 8080 端口存在 **ESTAB** 连接（OneBot 反向 WebSocket 链路在）。

任一条不满足 → 判定离线，并给出 `reason`（决定要不要重启、邮件里怎么写）。

第 2 条是 2026-09-28 加的：那天树莓派 WiFi 17:46 掉线 3 小时 47 分，
QQ 因此失联。如果只看 NapCat，会报成"QQ 未登录 → 去扫码"，
把人引向完全错误的方向（扫码根本没用，是网络问题）。

## 处置策略（区别对待，不做无用功）

| reason | 重启？ | 为什么 |
| --- | --- | --- |
| `bot_service_down` | 重启 qq-bot | 进程没了，重启有效 |
| `no_internet` | **不重启** | 断网时重启服务没有任何意义，只会白刷日志 |
| `napcat_unreachable` | 重启 napcat | NapCat 卡死，重启有效 |
| `onebot_link_missing` | 重启 qq-bot | QQ 已登录但链路断了，重建连接有效 |
| `qq_not_logged_in` | **不重启** | 需要人工扫码；重启只会让二维码失效（09-21 空转 639 次就是这么来的） |

## 邮件策略（少而准，绝不再刷屏）

一次掉线**最多两封**：离线 1 封 + 恢复 1 封（都带日志附件）。

- 离线：达到 `WATCHDOG_ALERT_AFTER_SECONDS` 后发**一封**；
  只有"还没送达"才会重试（同一封，非发出去不可），送达后不再重复；
- 想要周期性提醒才把 `WATCHDOG_REALERT_SECONDS` 设成 >0（默认 0 = 不重复）；
  2026-09-29 的教训：默认 30 分钟一封把用户邮箱刷爆，人只能关机躲它；
- 恢复：发一封总结（含完整日志附件）。如果离线期间那封**根本没送出去**
  （断网时邮件发不出去），恢复这封会明确写成"补报"并说明原因，绝不静默；
- 恢复邮件自己发失败 → 保留状态，下一个 tick 继续重试；
- 每次尝试都追加到 `WATCHDOG_LOG_FILE`，独立于 journald——
  09-25 那次就是因为另一个服务刷爆 journal，连"到底发没发"都查不到。

## 用法

    python3 scripts/pi_watchdog.py            # 正常巡检一次（systemd timer 调用）
    python3 scripts/pi_watchdog.py --dry-run  # 只打印判定与计划动作，不改任何状态
    python3 scripts/pi_watchdog.py --test-email  # 只测发信链路
    python3 scripts/pi_watchdog.py --status   # 打印当前状态文件
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULTS = {
    "WATCHDOG_ENABLED": "true",
    "WATCHDOG_ALERT_AFTER_SECONDS": "300",
    # 0 = 一次掉线**只发一封**离线告警，恢复时再发一封（默认）。
    # 设成 >0 才会在仍然离线时每隔这么久重复提醒一次。
    # 默认 0 是 2026-09-29 的教训：30 分钟一封把用户邮箱刷爆，只能关机。
    "WATCHDOG_REALERT_SECONDS": "0",
    "WATCHDOG_RESTART_COOLDOWN_SECONDS": "1800",
    "WATCHDOG_LOG_FILE": "data/watchdog.log",
    "WATCHDOG_STATE_FILE": "data/watchdog_state.json",
    "WATCHDOG_WEBUI_URL": "http://127.0.0.1:6099",
    # 邮件里写给人看的 WebUI 地址。留空则用 WATCHDOG_WEBUI_URL。
    # 单独一个变量是为了**不在代码里硬编码内网 IP**（仓库是公开的），
    # 真实地址只放在树莓派本地 .env（已 gitignore）。
    "WATCHDOG_WEBUI_HINT_URL": "",
    "WATCHDOG_WEBUI_TOKEN": "",
    "WATCHDOG_ONEBOT_PORT": "8080",
    "WATCHDOG_BOT_SERVICE": "qq-bot",
    "WATCHDOG_NAPCAT_SERVICE": "napcat",
    "SMTP_TIMEOUT_SECONDS": "30",
}
LOG_MAX_BYTES = 256 * 1024

# 外网可达性探测目标：**直接用 IP**，避免"探测本身依赖 DNS"的循环依赖。
# 全部失败才判定为断网（单个目标抖动不会误报）。
INTERNET_PROBES = (("223.5.5.5", 443), ("119.29.29.29", 443), ("1.1.1.1", 443))
INTERNET_PROBE_TIMEOUT = 4.0

# 邮件正文里附带的日志行数 / 附件里每类日志的行数上限
BODY_LOG_LINES = 12
ATTACH_LOG_LINES = 60
ATTACH_MAX_BYTES = 96 * 1024

REASON_LABELS = {
    "ok": "在线",
    "no_internet": "树莓派没有外网",
    "bot_service_down": "qq-bot 服务未运行",
    "napcat_unreachable": "NapCat 无响应（WebUI 打不开）",
    "qq_not_logged_in": "QQ 未登录（需要在 NapCat WebUI 扫码）",
    "onebot_link_missing": "QQ 已登录但 OneBot 链路断开",
    "watchdog_disabled": "看门狗已关闭",
}


# ==========================================================================
# 配置
# ==========================================================================


def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """极简 .env 解析（不依赖 python-dotenv，保持看门狗零依赖）。"""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@dataclass
class Config:
    enabled: bool
    alert_after: int
    realert: int
    restart_cooldown: int
    log_file: Path
    state_file: Path
    webui_url: str
    webui_hint_url: str
    webui_token: str
    onebot_port: int
    bot_service: str
    napcat_service: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    alert_to: str
    smtp_timeout: float

    @property
    def can_email(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password and self.alert_to)


def get_config(env: dict[str, str]) -> Config:
    def val(key: str) -> str:
        return (env.get(key) or DEFAULTS.get(key) or "").strip()

    def num(key: str, fallback: int) -> int:
        try:
            return int(val(key) or fallback)
        except ValueError:
            return fallback

    def path_of(key: str) -> Path:
        raw = val(key) or DEFAULTS[key]
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    return Config(
        enabled=(val("WATCHDOG_ENABLED") or "true").lower() not in ("0", "false", "no", "off"),
        alert_after=num("WATCHDOG_ALERT_AFTER_SECONDS", 300),
        realert=num("WATCHDOG_REALERT_SECONDS", 0),
        restart_cooldown=num("WATCHDOG_RESTART_COOLDOWN_SECONDS", 1800),
        log_file=path_of("WATCHDOG_LOG_FILE"),
        state_file=path_of("WATCHDOG_STATE_FILE"),
        webui_url=val("WATCHDOG_WEBUI_URL") or "http://127.0.0.1:6099",
        webui_hint_url=val("WATCHDOG_WEBUI_HINT_URL"),
        webui_token=val("WATCHDOG_WEBUI_TOKEN"),
        onebot_port=num("WATCHDOG_ONEBOT_PORT", 8080),
        bot_service=val("WATCHDOG_BOT_SERVICE") or "qq-bot",
        napcat_service=val("WATCHDOG_NAPCAT_SERVICE") or "napcat",
        smtp_host=val("SMTP_HOST"),
        smtp_port=num("SMTP_PORT", 465),
        smtp_user=val("SMTP_USER"),
        smtp_password=env.get("SMTP_PASSWORD") or "",
        alert_to=val("ALERT_EMAIL_TO"),
        smtp_timeout=float(num("SMTP_TIMEOUT_SECONDS", 30)),
    )


# ==========================================================================
# 判据（纯函数，便于测试）
# ==========================================================================


@dataclass
class Status:
    online: bool
    reason: str
    detail: str = ""


def classify(
    *,
    bot_service_active: bool,
    internet_ok: bool,
    napcat_login: bool | None,
    onebot_link: bool,
    bot_service: str = "qq-bot",
    onebot_port: int = 8080,
) -> Status:
    """把信号映射成 (是否在线, 原因)。

    napcat_login=None 表示 WebUI 不可达（NapCat 卡死 / 没起来）。
    顺序很重要：先看 bot 服务，再看**外网**，再看 NapCat，最后看链路——
    这样 reason 指向的是**最根本**的那一环。

    `internet_ok` 放在 NapCat 之前是 2026-09-28 的真实教训：
    那天树莓派 WiFi 在 17:46 掉了 3 小时 47 分，QQ 因此失联，
    但如果只看 NapCat 就会报成"QQ 未登录 → 去扫码"，
    把用户引向完全错误的方向（扫码根本没用，是网络问题）。
    """
    if not bot_service_active:
        return Status(False, "bot_service_down", f"{bot_service} 服务未运行")
    if not internet_ok:
        return Status(False, "no_internet", "树莓派无法访问外网（探测目标全部不可达）")
    if napcat_login is None:
        return Status(False, "napcat_unreachable", "NapCat WebUI 无法访问")
    if not napcat_login:
        return Status(False, "qq_not_logged_in", "QQ 未登录")
    if not onebot_link:
        return Status(
            False, "onebot_link_missing", f"本地 {onebot_port} 端口上没有 ESTAB 连接"
        )
    return Status(True, "ok")


# 需要重启的动作映射。刻意不含 no_internet / qq_not_logged_in：
# 断网重启服务没用；登录失效重启只会让二维码失效（09-21 空转了 639 次）。
RESTART_FOR_REASON = {
    "bot_service_down": "bot",
    "napcat_unreachable": "napcat",
    "onebot_link_missing": "bot",
}


def plan_actions(
    state: dict,
    status: Status,
    cfg: Config,
    now: float,
) -> list[str]:
    """根据当前状态与历史状态，决定这一轮要做什么（纯函数）。

    返回动作列表，可能包含：
        "alert_offline" / "alert_recovered" / "restart_bot" / "restart_napcat"
    """
    actions: list[str] = []
    was_offline = bool(state.get("offline_since"))

    if status.online:
        if was_offline:
            offline_since = float(state.get("offline_since") or now)
            offline_for = now - offline_since
            delivered = bool(state.get("alert_delivered"))
            # 两种情况都要在恢复时说一声：
            #   1. 离线告警成功发过了 → 这是"恢复通知"；
            #   2. 离线告警**一次都没送出去**（断网时邮件根本发不出去）→ 这是补报。
            # 只判断 (1) 是 09-25 静默 2.5 天的同一个坑：没送出去就永远不吭声。
            if delivered or offline_for >= cfg.alert_after:
                actions.append("alert_recovered")
        return actions

    offline_since = float(state.get("offline_since") or now)
    offline_for = now - offline_since

    # 1) 是否该发（或重发）离线告警
    if offline_for >= cfg.alert_after:
        delivered = bool(state.get("alert_delivered"))
        last_delivered_at = float(state.get("last_alert_delivered_at") or 0)
        if not delivered:
            # 还没送达过 → 每个 tick 都重试，直到成功。
            # 这不是"多发"，而是"同一封非发出去不可"（09-25 静默 2.5 天的根因）。
            actions.append("alert_offline")
        elif cfg.realert > 0 and now - last_delivered_at >= cfg.realert:
            # 只有显式开启了重复提醒（WATCHDOG_REALERT_SECONDS > 0）才补发。
            # 默认 0 = 一次掉线只发一封：2026-09-29 用户被 30 分钟一封刷爆邮箱，
            # 只能关机，明确要求"掉线只发 1 封、恢复再发 1 封"。
            actions.append("alert_offline")

    # 2) 是否该重启（带冷却，避免 09-21 那种 639 次空转）
    target = RESTART_FOR_REASON.get(status.reason)
    if target:
        last = state.get(f"last_restart_{target}_at")
        # 从未重启过 → 立即允许；重启过 → 必须过了冷却期。
        # 刻意不用 `or 0` 比较：那会让判断依赖"当前时间戳足够大"，是个隐患。
        if last is None or now - float(last) >= cfg.restart_cooldown:
            actions.append(f"restart_{target}")
    return actions


# ==========================================================================
# 采集（唯一有副作用的部分）
# ==========================================================================


def _run(cmd: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_bot_service(service: str) -> bool:
    try:
        return _run(["systemctl", "is-active", "--quiet", service]).returncode == 0
    except Exception:
        return False


def check_napcat_login(cfg: Config) -> bool | None:
    """True=已登录 / False=未登录 / None=WebUI 不可达。"""
    if not cfg.webui_token:
        return None
    import hashlib

    digest = hashlib.sha256(f"{cfg.webui_token}.napcat".encode()).hexdigest()
    try:
        req = urllib.request.Request(
            f"{cfg.webui_url}/api/auth/login",
            data=json.dumps({"hash": digest}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            credential = json.loads(resp.read()).get("data", {}).get("Credential", "")
        if not credential:
            return None
        req = urllib.request.Request(
            f"{cfg.webui_url}/api/QQLogin/CheckLoginStatus",
            data=b"{}",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {credential}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read()).get("data", {})
        return bool(data.get("isLogin"))
    except Exception:
        return None


def check_onebot_link(port: int) -> bool:
    """本地 port 上是否存在 ESTAB 连接（OneBot 反向 WS 链路）。"""
    try:
        out = _run(["ss", "-tn"], timeout=8).stdout
    except Exception:
        return False
    needle = f"127.0.0.1:{port}"
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "ESTAB" and parts[3] == needle:
            return True
    return False


def check_internet(timeout: float = INTERNET_PROBE_TIMEOUT) -> bool:
    """树莓派能否访问外网：对所有探测目标做 TCP 连接，**全部失败**才算断网。

    刻意用 IP 而不是域名：断网时 DNS 通常也一起挂，
    用域名探测会把"DNS 挂了"和"网络挂了"混在一起。
    """
    for host, port in INTERNET_PROBES:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            continue
    return False


# ==========================================================================
# 副作用：日志 / 状态 / 邮件 / 重启
# ==========================================================================


def append_log(cfg: Config, message: str) -> None:
    try:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        if cfg.log_file.is_file() and cfg.log_file.stat().st_size > LOG_MAX_BYTES:
            tail = cfg.log_file.read_bytes()[-LOG_MAX_BYTES // 2 :]
            cfg.log_file.write_bytes(b"# (truncated)\n" + tail)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with cfg.log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} [watchdog] {message}\n")
    except Exception:
        pass


def load_state(cfg: Config) -> dict:
    try:
        if cfg.state_file.is_file():
            return json.loads(cfg.state_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_state(cfg: Config, state: dict) -> None:
    try:
        cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def send_email(
    cfg: Config,
    subject: str,
    body: str,
    *,
    attachment: tuple[str, str] | None = None,
) -> bool:
    """发信。body 是**纯文本**（绝不含 markdown 记号）。

    attachment=(文件名, 文本内容) 时，作为 UTF-8 文本附件附上。
    """
    if not cfg.can_email:
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = cfg.smtp_user
        msg["To"] = cfg.alert_to
        msg.set_content(body)
        if attachment is not None:
            filename, text = attachment
            # 传 str（不是 bytes）：str 走 set_text_content，附件头里才会带
            # charset="utf-8" + base64，中文日志在邮件客户端里才不是乱码。
            # 注意 set_text_content **不接受 maintype**（传了会 TypeError），
            # 而 bytes 分支（set_bytes_content）**不接受 charset** —— 两种拼法都踩过。
            msg.add_attachment(text, subtype="plain", filename=filename)
        context = ssl.create_default_context()
        if cfg.smtp_port == 465:
            with smtplib.SMTP_SSL(
                cfg.smtp_host, cfg.smtp_port, timeout=cfg.smtp_timeout, context=context
            ) as server:
                server.login(cfg.smtp_user, cfg.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=cfg.smtp_timeout) as server:
                server.ehlo()
                if server.has_extn("starttls"):
                    server.starttls(context=context)
                    server.ehlo()
                server.login(cfg.smtp_user, cfg.smtp_password)
                server.send_message(msg)
        return True
    except Exception as exc:
        append_log(cfg, f"email FAILED subject={subject} error={type(exc).__name__}: {str(exc)[:160]}")
        return False


# ==========================================================================
# 掉线日志采集（邮件正文给摘要，附件给完整）
# ==========================================================================

# journal 里的噪音：与故障无关且极长，采集时必须滤掉
_LOG_NOISE = (
    "RealtimeKit",
    "PipeWire",
    "xdg-desktop",
    "locale resources",
    "dri3 extension",
    "InitializeSandbox",
    "gpu_memory_buffer",
)


def _journal(
    units: list[str] | None,
    since: float | None,
    lines: int,
    until: float | None = None,
) -> list[str]:
    """读 journal（**不需要 sudo**：看门狗用户已在 adm 组）。

    units=None 表示全系统日志；给列表则按单元过滤。
    until 给的是**窗口起点那一端**：`since ~ until` 用来精确抓故障发生的那一刻。
    任何失败都返回空列表，绝不让采集拖垮告警。
    """
    cmd = ["journalctl", "--no-pager", "-n", str(lines), "-o", "short-iso"]
    for unit in units or []:
        cmd += ["-u", unit]
    if since:
        cmd += ["--since", datetime.fromtimestamp(since).strftime("%Y-%m-%d %H:%M:%S")]
    if until:
        cmd += ["--until", datetime.fromtimestamp(until).strftime("%Y-%m-%d %H:%M:%S")]
    try:
        out = _run(cmd, timeout=20).stdout
    except Exception:
        return []
    kept = []
    for line in out.splitlines():
        if line.startswith("-- "):  # journalctl 的占位行：-- No entries -- / -- Reboot --
            continue
        if any(noise in line for noise in _LOG_NOISE):
            continue
        if "[▄" in line or "█" in line:  # 二维码艺术字
            continue
        kept.append(line)
    return kept


# 网络层单元。09-28 的离线根因（WiFi 掉线）只出现在这些单元里，
# napcat / qq-bot 的日志里只有一句"账号离线"。
NETWORK_UNITS = ("NetworkManager", "wpa_supplicant", "systemd-networkd", "dhcpcd")
# 掉线起点往后多久算"故障发生现场"。
# 取 600s（而不是更长）是刻意的：journalctl -n 取的是窗口**尾部**，
# 窗口越长，尾巴离真正的故障时刻就越远。
NETWORK_ONSET_WINDOW = 600.0
# 兜底关键字：万一网络由别的组件管（netplan / iwd / connman…）
_NETWORK_KEYWORDS = (
    "NetworkManager",
    "wpa_supplicant",
    "dhcp",
    "carrier",
    "link is not ready",
    "wlan0",
    "eth0",
    "default via",
    "no internet",
    "Temporary failure in name resolution",
)


def _merge_chrono(onset: list[str], tail: list[str], lines: int) -> list[str]:
    """合并"故障现场"与"最近"两段，各自按时间戳排序去重。

    short-iso 以时间戳开头，取前 25 字符（`2026-09-28T21:33:01+08:00`）即可比较；
    sorted 是稳定排序，同一秒内保持 journalctl 给出的原始顺序。
    """
    half = max(1, lines // 2)
    onset = sorted(dict.fromkeys(onset), key=lambda ln: ln[:25])[-half:]
    tail = sorted(dict.fromkeys(tail), key=lambda ln: ln[:25])[-half:]
    if not onset or not tail:
        return onset or tail
    seen = set(onset)
    tail_only = [ln for ln in tail if ln not in seen]
    if not tail_only:
        return onset  # 离线时间很短，两段本来就是重合的
    # 中间那段（可能几小时）省略，只留"怎么坏的"和"怎么好的"
    return onset + ["…（中间省略）…"] + tail_only


def collect_network_logs(since: float, lines: int = ATTACH_LOG_LINES) -> list[str]:
    """采集网络层日志（掉线最常见、也最容易被漏掉的根因所在）。

    两个坑，都是实测踩出来的：

    1. **必须按单元直查**，不能"取全系统最后 N 行再过滤"。本机 journal 很吵
       （etest / qq-bot 刷屏），17:46 的掉线事件在当天就已经被挤出窗口。
    2. **必须同时取故障起点**，不能只取最近 N 行。09-28 离线 3 小时 47 分，
       只取尾部的话抓到的全是 21:33 的"恢复"，真正的原因（17:46 掉线）看不到。
    """
    half = max(1, lines // 2)
    onset: list[str] = []
    tail: list[str] = []
    for unit in NETWORK_UNITS:
        onset += _journal([unit], since, half, until=since + NETWORK_ONSET_WINDOW)
        tail += _journal([unit], since, half)
    if onset or tail:
        return _merge_chrono(onset, tail, lines)

    # 兜底：单元名不对时退回全系统关键字过滤（窗口给大一些）
    return [
        line
        for line in _journal(None, since, 3000)
        if any(key in line for key in _NETWORK_KEYWORDS)
    ][-lines:]


def _tail_file(path: Path, lines: int) -> list[str]:
    try:
        if not path.is_file():
            return []
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return content[-lines:]
    except Exception:
        return []


def collect_logs(cfg: Config, offline_since: float) -> tuple[str, str, str]:
    """返回 (正文用的日志摘要, 附件用的完整日志文本, 附件文件名)。

    采集范围刻意覆盖网络层——2026-09-28 的事故根因是 WiFi 掉线，
    napcat/qq-bot 的日志里只有"账号离线"，真正的原因在 NetworkManager 里。
    """
    since = offline_since - 300  # 往前多看 5 分钟，抓住故障前的最后一刻

    sections: list[tuple[str, list[str]]] = [
        ("看门狗日志（最近 %d 行）" % ATTACH_LOG_LINES, _tail_file(cfg.log_file, ATTACH_LOG_LINES)),
        ("napcat 服务日志", _journal([cfg.napcat_service], since, ATTACH_LOG_LINES)),
        ("qq-bot 服务日志", _journal([cfg.bot_service], since, ATTACH_LOG_LINES)),
        ("网络事件（NetworkManager / WiFi / DHCP）", collect_network_logs(since)),
        ("bot 侧告警投递记录", _tail_file(cfg.log_file.parent / "offline_alert.log", 20)),
    ]

    header = [
        "qq_chatbot 掉线现场日志",
        f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"主机: {_hostname()}",
        f"离线起点: {datetime.fromtimestamp(offline_since):%Y-%m-%d %H:%M:%S}",
        "说明: 纯文本附件，由树莓派独立看门狗自动采集；不含任何凭据。",
        "=" * 64,
        "",
    ]
    parts = list(header)
    for title, lines in sections:
        if not lines:
            continue
        parts.append(f"----- {title} -----")
        parts.extend(lines)
        parts.append("")
    full = "\n".join(parts)
    if len(full.encode("utf-8")) > ATTACH_MAX_BYTES:
        full = full.encode("utf-8")[:ATTACH_MAX_BYTES].decode("utf-8", "ignore") + "\n…（已截断）\n"

    # 正文摘要：看门狗自己最近几行 + 网络事件最后几行（最常指向根因）
    brief_lines: list[str] = []
    brief_lines += sections[0][1][-BODY_LOG_LINES:]
    net_lines = sections[3][1][-6:]
    if net_lines:
        brief_lines.append("")
        brief_lines.append("网络事件（最后 6 条）:")
        brief_lines += net_lines
    brief = "\n".join(brief_lines) if brief_lines else "（未采集到日志）"

    filename = f"offline-logs-{datetime.now():%Y%m%d-%H%M%S}.txt"
    return brief, full, filename


def restart_service(cfg: Config, which: str) -> bool:
    """重启服务。需要非交互 sudo（见部署时的 /etc/sudoers.d/qq-watchdog）。

    刻意不带 `--no-block`：oneshot 等它做完更可靠，而 sudoers 里也只授权
    这两条**精确的命令行**，不带额外参数。
    """
    service = cfg.bot_service if which == "bot" else cfg.napcat_service
    try:
        result = _run(["sudo", "-n", "systemctl", "restart", service], timeout=30)
        ok = result.returncode == 0
        if not ok and result.stderr:
            append_log(cfg, f"restart {service} stderr: {result.stderr.strip()[:120]}")
    except Exception as exc:
        ok = False
        append_log(cfg, f"restart {service} raised {type(exc).__name__}")
    append_log(cfg, f"restart {service}: {'OK' if ok else 'FAILED'}")
    return ok


def fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days} 天")
    if hours:
        parts.append(f"{hours} 小时")
    parts.append(f"{minutes} 分")
    return " ".join(parts)


def build_offline_body(
    cfg: Config,
    status: Status,
    offline_for: float,
    hint: str,
    log_brief: str = "",
) -> str:
    """离线告警正文：**纯文本，不含任何 markdown 记号**。

    纯文本邮件里 `**加粗**`、`# 标题` 只会显示成字面星号/井号，很难看，
    所以正文一律使用「」和中划线做视觉分层。
    """
    lines = [
        "QQ Bot 离线告警（由树莓派独立看门狗检测）",
        "=" * 46,
        "",
        f"主机        : {_hostname()}",
        f"检测时间    : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"已离线      : {fmt_duration(offline_for)}",
        f"原因        : {REASON_LABELS.get(status.reason, status.reason)}",
        f"原因代码    : {status.reason}",
        f"判定明细    : {status.detail or '-'}",
        "",
        f"「该怎么办」",
        hint,
        "",
        "=" * 46,
        "最近日志（完整日志见附件）",
        "-" * 46,
        log_brief or "（未采集到日志）",
        "",
        "=" * 46,
        "说明：这条告警来自独立于 Bot 进程的看门狗，",
        "即使 Bot 崩溃、卡死或从未启动，它也会照常通知。",
    ]
    return "\n".join(lines)


def build_recovered_body(
    cfg: Config,
    offline_for: float,
    log_brief: str = "",
    *,
    missed: bool = False,
) -> str:
    """恢复通知正文（同样纯文本）。

    missed=True 表示离线期间的告警邮件**始终没能送出**（多半是树莓派当时
    没有外网），这封就是恢复后的补报 —— 必须说清楚，否则用户会以为
    "离线期间没发生过事"。
    """
    lines = [
        "QQ Bot 已恢复",
        "=" * 46,
        "",
        f"主机        : {_hostname()}",
        f"恢复时间    : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"离线时长    : {fmt_duration(offline_for)}",
        "",
    ]
    if missed:
        lines += [
            "注意：这次离线期间，看门狗没能把告警发出来",
            "      （最常见的原因就是树莓派当时没有外网，邮件发不出去），",
            "      所以这封是恢复后的补报。",
            "",
        ]
    lines.append("机器人连接已恢复。")
    if log_brief:
        lines += ["", "=" * 46, "最近日志（完整日志见附件）", "-" * 46, log_brief]
    return "\n".join(lines)


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def webui_hint_url(cfg: Config) -> str:
    """邮件里写给人看的 WebUI 地址。

    单独开一个变量，是为了**不在公开仓库里硬编码内网 IP**：
    真实可达地址（例如 `http://<树莓派局域网IP>:6099/webui`）只写在
    树莓派本地的 `.env` 里，代码里最多只有 127.0.0.1。
    """
    base = (cfg.webui_hint_url or cfg.webui_url or "").rstrip("/")
    if not base:
        return "（未配置 WATCHDOG_WEBUI_URL）"
    return base if base.endswith("/webui") else f"{base}/webui"


def build_hint(cfg: Config, reason: str) -> str:
    """把 HINTS 模板里的 {webui} 换成真实地址；模板坏了也不能影响发信。"""
    template = HINTS.get(reason, DEFAULT_HINT)
    try:
        return template.format(webui=webui_hint_url(cfg))
    except (KeyError, IndexError, ValueError):
        return template


HINTS = {
    # 邮件正文一律纯文本：不用 markdown 记号（**粗体**、# 标题在纯文本邮件里
    # 只会显示成字面星号/井号）。用「」和短横线做视觉分层。
    "no_internet": (
        "树莓派本身访问不了外网，QQ 因此失联——这不是 QQ 或扫码的问题。\n"
        "请检查：\n"
        "  1. WiFi 是否掉线（wlan0 / 路由器；默认路由走 wlan0）\n"
        "  2. 网线这条备路（eth0，走的是电脑热点，通常也没外网）\n"
        "  3. 网络恢复后 QQ 会自动重连，看门狗会再发一封恢复邮件。\n"
        "看门狗不会重启任何服务——断网时重启没有意义。"
    ),
    "qq_not_logged_in": (
        "QQ 登录态已失效，需要在 NapCat WebUI 扫码登录：\n"
        "  {webui}\n"
        "看门狗刻意不会重启 NapCat —— 重启只会让二维码失效。"
    ),
    "napcat_unreachable": "NapCat 无响应，看门狗已尝试重启 napcat 服务。",
    "bot_service_down": "qq-bot 服务未运行，看门狗已尝试重启。",
    "onebot_link_missing": "QQ 已登录但 OneBot 链路断开，看门狗已尝试重启 qq-bot。",
}
DEFAULT_HINT = "请登录树莓派检查 qq-bot / napcat 状态。"


# ==========================================================================
# 主流程
# ==========================================================================


def run_once(cfg: Config, *, dry_run: bool = False) -> int:
    now = time.time()
    state = load_state(cfg)

    bot_active = check_bot_service(cfg.bot_service)
    napcat_login = check_napcat_login(cfg)
    link = check_onebot_link(cfg.onebot_port)
    internet_ok = check_internet()
    status = classify(
        bot_service_active=bot_active,
        internet_ok=internet_ok,
        napcat_login=napcat_login,
        onebot_link=link,
        bot_service=cfg.bot_service,
        onebot_port=cfg.onebot_port,
    )

    if not status.online:
        # 补上更具体的判定明细
        status.detail = status.detail or (
            f"bot={bot_active} net={internet_ok} login={napcat_login} link={link}"
        )

    actions = plan_actions(state, status, cfg, now)

    summary = (
        f"{'在线' if status.online else '离线'} reason={status.reason} "
        f"bot={int(bot_active)} net={int(internet_ok)} login={napcat_login} link={int(link)} "
        f"actions={actions or ['none']}"
    )
    print(f"[watchdog] {summary}")
    if dry_run:
        return 0

    append_log(cfg, summary)

    if status.online:
        if "alert_recovered" not in actions:
            # 短暂抖动，或已经说清楚了：状态清零，不留尾巴
            save_state(cfg, {"online": True})
            return 0

        offline_since = float(state.get("offline_since") or now)
        offline_for = now - offline_since
        delivered = bool(state.get("alert_delivered"))
        brief, full, filename = collect_logs(cfg, offline_since)
        subject = (
            "[QQ Bot Alert] QQ Bot 已恢复"
            if delivered
            else "[QQ Bot Alert] QQ Bot 已恢复（离线告警此前未能送出）"
        )
        ok = send_email(
            cfg,
            subject,
            build_recovered_body(cfg, offline_for, brief, missed=not delivered),
            attachment=(filename, full),
        )
        append_log(
            cfg,
            f"recovered email: {'OK' if ok else 'FAILED'} "
            f"(offline_for={int(offline_for)}s, missed_alert={not delivered}, 附日志 {filename})",
        )
        if ok:
            save_state(cfg, {"online": True})
        else:
            # 恢复邮件本身也没送出去（发信链路有问题）→ **保留**离线状态，
            # 下一个 tick 继续重试，绝不静默丢掉"这段时间掉过线"这件事。
            state["online"] = True
            save_state(cfg, state)
            append_log(cfg, "恢复邮件未送出，保留状态待下一轮重试")
        return 0

    # ---- 离线分支 ----
    if not state.get("offline_since"):
        state["offline_since"] = now
        state["alert_delivered"] = False
        state.pop("last_alert_delivered_at", None)
    offline_for = now - float(state["offline_since"])

    for action in actions:
        if action == "restart_bot":
            restart_service(cfg, "bot")
            state["last_restart_bot_at"] = now
        elif action == "restart_napcat":
            restart_service(cfg, "napcat")
            state["last_restart_napcat_at"] = now
        elif action == "alert_offline":
            state["last_alert_attempt_at"] = now
            hint = build_hint(cfg, status.reason)
            brief, full, filename = collect_logs(cfg, float(state["offline_since"]))
            ok = send_email(
                cfg,
                f"[QQ Bot Alert] QQ Bot 离线（{REASON_LABELS.get(status.reason, status.reason)}）",
                build_offline_body(cfg, status, offline_for, hint, brief),
                attachment=(filename, full),
            )
            append_log(
                cfg,
                f"offline email: {'OK' if ok else 'FAILED'} "
                f"(offline_for={int(offline_for)}s, 附日志 {filename})",
            )
            if ok:
                state["alert_delivered"] = True
                state["last_alert_delivered_at"] = now
            else:
                state["alert_delivered"] = False

    state["online"] = False
    state["reason"] = status.reason
    state["alert_delivered"] = bool(state.get("alert_delivered"))
    save_state(cfg, state)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="树莓派独立看门狗")
    parser.add_argument("--dry-run", action="store_true", help="只打印判定，不改状态、不发信、不重启")
    parser.add_argument("--test-email", action="store_true", help="只测发信链路")
    parser.add_argument("--status", action="store_true", help="打印状态文件")
    args = parser.parse_args(argv)

    env = load_env_file()
    # 进程环境变量优先于 .env 文件（与 python-dotenv / 12-factor 的惯例一致），
    # 这样临时覆盖（测试、模拟离线）不需要改 .env，也让 systemd 的
    # Environment= 指令可以生效。
    env = {**env, **os.environ}
    cfg = get_config(env)

    if args.status:
        print(json.dumps(load_state(cfg), ensure_ascii=False, indent=2))
        return 0

    if args.test_email:
        if not cfg.can_email:
            print("[watchdog] SMTP 配置不完整，无法测试")
            return 1
        ok = send_email(
            cfg,
            "[QQ Bot Alert] 看门狗发信测试",
            f"这是一封来自独立看门狗的测试邮件。\n时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"主机: {_hostname()}",
        )
        print(f"[watchdog] test email: {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    if not cfg.enabled and not args.dry_run:
        append_log(cfg, "WATCHDOG_ENABLED=false，跳过")
        return 0
    return run_once(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
