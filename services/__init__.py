"""业务服务包：DeepSeek / 智谱 GLM API 封装等。"""

import os


def redact_secrets(text: str) -> str:
    """把已知的密钥（API Key、接入令牌）从文本中替换成 ***。

    用于写日志前的兜底清洗，防止任何异常信息意外把密钥带进日志。
    """
    for env_name in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "ONEBOT_ACCESS_TOKEN"):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "***")
    return text


# ===== 日志隐私策略（v0.3.1） =====
# 生产默认日志不得记录聊天正文 / Personal Memory value / 长期记忆正文 /
# 搜索 query / Persona 原始台词。默认只记录长度等 metadata。
# LOG_MESSAGE_CONTENT=true 是显式开发调试开关，默认 false；非法布尔值安全回落 false。
#
# 刻意做成“每次调用实时读环境变量”而不是模块级常量：
# 本模块会被 services/ 下多个模块在 import 期使用，模块级读取会重新引入
# “load_dotenv 之前 import”的顺序陷阱（见 bot.py 的 bootstrap 说明）。

_LOG_MESSAGE_CONTENT_TRUE_VALUES = ("1", "true", "yes", "on")

# 即便开启 LOG_MESSAGE_CONTENT，正文写日志也最多保留这么多字符（含省略号）
LOG_CONTENT_MAX_CHARS = 120


def log_message_content_enabled() -> bool:
    """LOG_MESSAGE_CONTENT 开关：默认 false；任何非法值都安全回落 false。"""
    raw = (os.getenv("LOG_MESSAGE_CONTENT") or "").strip().lower()
    return raw in _LOG_MESSAGE_CONTENT_TRUE_VALUES


def safe_log_text(text: str, max_chars: int = LOG_CONTENT_MAX_CHARS) -> str:
    """聊天正文类内容写日志前的第二层清洗：redact_secrets + 折叠空白 + 截断。

    这只是兜底：正常 INFO 日志应优先只记长度（question_chars），
    只有 LOG_MESSAGE_CONTENT=true 时才把本函数的输出写进日志。
    """
    cleaned = redact_secrets(str(text or ""))
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"
    return cleaned
