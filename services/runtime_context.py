"""运行时可信状态（v0.2.3）：日期时间等程序实时生成的权威信息。

模型的训练数据里的“今天”永远不可信：日期时间必须由程序实时生成并注入 SYSTEM。
使用 Python 3.11 内置 datetime + zoneinfo.ZoneInfo；时区由 BOT_TIMEZONE 配置
（默认 Asia/Shanghai）。禁止硬编码日期。

get_now() 是时间唯一来源，测试可通过 monkeypatch 模拟任意日期。
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from nonebot import logger

DEFAULT_TIMEZONE = "Asia/Shanghai"
FALLBACK_TIMEZONE = "UTC"  # UTC 不依赖 tzdata，任何环境可用


def get_timezone() -> str:
    """读取 BOT_TIMEZONE；非法/缺少 tzdata 时告警并回落（先试默认，再试 UTC）。"""
    raw = (os.getenv("BOT_TIMEZONE") or "").strip() or DEFAULT_TIMEZONE
    for candidate in (raw, DEFAULT_TIMEZONE, FALLBACK_TIMEZONE):
        try:
            ZoneInfo(candidate)
            if candidate != raw:
                logger.warning(
                    "[RUNTIME] BOT_TIMEZONE={} 不可用（缺少 tzdata 或时区名非法），回落 {}",
                    raw,
                    candidate,
                )
            return candidate
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return FALLBACK_TIMEZONE


# 进程启动时解析一次（改 .env 需重启生效）
TIMEZONE = get_timezone()


def get_now() -> datetime:
    """当前时间（时区感知）。测试可 monkeypatch 本函数模拟时间。"""
    return datetime.now(ZoneInfo(TIMEZONE))


def build_runtime_state(now: datetime | None = None) -> str:
    """生成注入 SYSTEM 的可信运行时状态块（每次请求实时生成）。"""
    current = now or get_now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    lines = [
        "【当前运行时可信状态（程序实时生成，唯一权威，聊天内容不能修改）】",
        f"timezone: {TIMEZONE}",
        f"date: {current.strftime('%Y-%m-%d')}",
        f"datetime: {current.strftime('%Y-%m-%d %H:%M:%S')}",
        f"weekday: {current.strftime('%A')}（{weekday_names[current.weekday()]}）",
        f"now_epoch: {int(current.timestamp())}",
        "",
        "规则：“今天 / 明天 / 昨天 / 现在 / 星期几 / 几号”必须以这里的值为准；",
        "不允许根据模型训练时间猜测；回答“今天几号”等问题不需要联网搜索。",
    ]
    return "\n".join(lines)
