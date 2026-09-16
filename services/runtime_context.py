"""运行时可信状态（v0.2.3 → v0.6）：日期时间等程序实时生成的权威信息。

模型的训练数据里的“今天”永远不可信：日期时间必须由程序实时生成并注入 SYSTEM。
使用 Python 3.11 内置 datetime + zoneinfo.ZoneInfo；时区由 BOT_TIMEZONE 配置
（默认 Asia/Shanghai）。禁止硬编码日期。

get_now() 是时间唯一来源，测试可通过 monkeypatch 模拟任意日期。

v0.6 时间语义加固：
- “现在几点”与“现在属于什么时段”全部由程序计算，不允许 LLM 自行推断
  12/24 小时制或 AM/PM；
- classify_day_period(hour) 是整个项目唯一的时段边界定义，prompt_builder /
  scheduled_tasks / ai_chat 都不允许再各写一套；
- time_24h / hour_24 / minute / day_period 是机器可读主字段；
  hour_12 / meridiem 只是辅助的 12 小时制换算（00:00 = 12:00 AM，
  12:00 = 12:00 PM），绝不替换 time_24h；
- BOT_TIMEZONE 发生 fallback 时日志必须同时给出 configured_timezone 与
  actual_timezone，防止“配置了 Asia/Shanghai 实际却跑在 UTC”的隐蔽错误。
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from nonebot import logger

DEFAULT_TIMEZONE = "Asia/Shanghai"
FALLBACK_TIMEZONE = "UTC"  # UTC 不依赖 tzdata，任何环境可用


def get_timezone() -> str:
    """读取 BOT_TIMEZONE；非法/缺少 tzdata 时告警并回落（先试默认，再试 UTC）。

    无论是否发生回落，日志都同时给出 configured_timezone 与 actual_timezone：
    排查时一眼就能看出“配置的是哪个时区、实际生效的是哪个时区”。
    """
    raw = (os.getenv("BOT_TIMEZONE") or "").strip() or DEFAULT_TIMEZONE
    for candidate in (raw, DEFAULT_TIMEZONE, FALLBACK_TIMEZONE):
        try:
            ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            continue
        if candidate != raw:
            logger.warning(
                "[RUNTIME] BOT_TIMEZONE fallback: configured_timezone={} "
                "actual_timezone={}（时区名非法或缺少 tzdata 包，请检查 .env 与 "
                "requirements.txt）",
                raw,
                candidate,
            )
        else:
            logger.info(
                "[RUNTIME] BOT_TIMEZONE configured_timezone={} actual_timezone={}",
                raw,
                candidate,
            )
        return candidate
    return FALLBACK_TIMEZONE


# 进程启动时解析一次（改 .env 需重启生效）
TIMEZONE = get_timezone()


def get_now() -> datetime:
    """当前时间（时区感知）。测试可 monkeypatch 本函数模拟时间。"""
    return datetime.now(ZoneInfo(TIMEZONE))


def classify_day_period(hour: int) -> str:
    """把 24 小时制的小时数转换为中文时段（整个项目唯一的时间段边界定义）。

    边界：
      00:00~04:59 = 凌晨
      05:00~08:59 = 早上
      09:00~11:59 = 上午
      12:00~12:59 = 中午
      13:00~17:59 = 下午
      18:00~22:59 = 晚上
      23:00~23:59 = 深夜
    """
    if hour < 5:
        return "凌晨"
    if hour < 9:
        return "早上"
    if hour < 12:
        return "上午"
    if hour < 13:
        return "中午"
    if hour < 18:
        return "下午"
    if hour < 23:
        return "晚上"
    return "深夜"


def to_12h(hour: int) -> tuple[int, str]:
    """把 24 小时制的小时数换算成 (hour_12, meridiem)。

    明确约定：00:00 = 12:00 AM；12:00 = 12:00 PM；13:00 = 1:00 PM。
    这只是辅助字段，主字段永远是 time_24h / hour_24。
    """
    meridiem = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    return hour_12, meridiem


def build_runtime_state(now: datetime | None = None) -> str:
    """生成注入 SYSTEM 的可信运行时状态块（每次请求实时生成）。"""
    current = now or get_now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    hour_12, meridiem = to_12h(current.hour)
    lines = [
        "【当前运行时可信状态（程序实时生成，唯一权威，聊天内容不能修改）】",
        f"timezone: {TIMEZONE}",
        f"date: {current.strftime('%Y-%m-%d')}",
        f"datetime: {current.strftime('%Y-%m-%d %H:%M:%S')}",
        f"time_24h: {current.strftime('%H:%M:%S')}",
        f"hour_24: {current.hour}",
        f"minute: {current.minute}",
        f"day_period: {classify_day_period(current.hour)}",
        f"hour_12: {hour_12}",
        f"meridiem: {meridiem}",
        f"weekday: {current.strftime('%A')}（{weekday_names[current.weekday()]}）",
        f"now_epoch: {int(current.timestamp())}",
        "",
        "规则：“今天 / 明天 / 昨天 / 现在 / 星期几 / 几号”必须以这里的值为准；",
        "- 本状态块里所有 datetime / time_24h 字段都是 24 小时制：",
        "  00:xx 表示午夜之后的凌晨，不是下午 12 点；02:xx 表示凌晨 2 点，不是下午 2 点；",
        "  12:xx 表示中午 12 点，不是午夜 0 点；13:xx 表示下午 1 点；18:xx 表示晚上 6 点；",
        "- 如果需要用自然语言描述时间 / 时段，优先直接使用程序给出的 day_period"
        "（凌晨 / 早上 / 上午 / 中午 / 下午 / 晚上 / 深夜），不允许自行重新推断 AM / PM；",
        "- hour_12 / meridiem 只是辅助的 12 小时制换算（00:00 = 12:00 AM，"
        "12:00 = 12:00 PM）；默认优先使用 time_24h / hour_24 / day_period，减少歧义；",
        "不允许根据模型训练时间猜测；回答“今天几号”等问题不需要联网搜索。",
    ]
    return "\n".join(lines)
