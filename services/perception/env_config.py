"""环境变量解析助手（v0.7）：整个感知层共用的“解析 + 范围校验 + 安全回落”。

为什么不复用各模块里已有的私有 _env_int（vision.py / context_serializer.py /
poke.py 各有一份）：
- 本次新增的 REPLY_* / FORWARD_* / FILE_* / *_MAX_* 配置项超过 15 个，
  再复制 15 份解析函数就是明显的重复代码；
- 本模块只有“读环境变量 → 返回安全值”这一件事，没有任何业务语义，
  不会成为第二套 config system（真正的配置语义仍然写在各自模块顶部）。

失败语义（与项目既有约定一致）：未配置 → 默认值；非法 / 超范围 →
WARNING 日志 + 默认值，绝不在 import 期抛异常。
"""

import os

from nonebot import logger


def env_bool(name: str, default: bool, log_tag: str) -> bool:
    """解析布尔型环境变量；非法值安全回落默认值。"""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("[{}] {}={} 不是合法布尔值，使用默认 {}", log_tag, name, raw, default)
    return default


def env_int(
    name: str,
    default: int,
    low: int,
    high: int,
    log_tag: str,
) -> int:
    """解析整数型环境变量并做范围校验；非法 / 超范围安全回落默认值。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[{}] {}={} 不是合法整数，使用默认 {}", log_tag, name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning(
            "[{}] {}={} 超出范围 [{}, {}]，使用默认 {}",
            log_tag,
            name,
            value,
            low,
            high,
            default,
        )
        return default
    return value
