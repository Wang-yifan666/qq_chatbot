"""DeepSeek API 客户端封装。

本模块只负责一件事：把用户问题发给 DeepSeek，返回文本回答。

要点：
- 使用 OpenAI 官方 SDK 的 AsyncOpenAI（异步客户端），不会阻塞
  NoneBot2 的 asyncio 事件循环；
- API Key 从环境变量 DEEPSEEK_API_KEY 读取，严禁硬编码；
- 模型名从环境变量 DEEPSEEK_MODEL 读取，未配置时使用默认值 deepseek-v4-flash；
- 客户端懒加载：首次调用时才创建。这样当 .env 里 AI_PROVIDER=zhipu
  （即本次不选用 DeepSeek）时，缺少 DEEPSEEK_API_KEY 也不会影响 Bot 启动；
- 当前版本每次请求都是独立对话，不携带任何聊天历史。
"""

import os

from nonebot import logger
from openai import AsyncOpenAI

from services import redact_secrets

# DeepSeek 兼容 OpenAI 接口，只需把 base_url 指向 DeepSeek
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 单次请求超时（秒），包含建立连接 + 等待回复。
# DeepSeek 偶发响应较慢，60 秒比较稳妥；如觉得太久可改小。
DEEPSEEK_TIMEOUT = 60.0

# 系统提示词：让 AI 用中文清晰、准确地回答
SYSTEM_PROMPT = "你是一个QQ群里的AI助手，请使用中文清晰、准确地回答用户问题。"

# 模型名从环境变量读取，不散落在业务代码里；未配置时用默认值
# DeepSeek 当前支持的模型名：deepseek-v4-flash / deepseek-v4-pro（deepseek-chat 为兼容别名）
model = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """懒加载获取全局唯一的异步客户端（首次调用时才创建，复用底层连接池）。"""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            # bot.py 启动时已做同样的检查，这里再兜底一次，
            # 避免直接抛出难以理解的 SDK 异常
            raise RuntimeError("缺少 DEEPSEEK_API_KEY：请在 .env 中填写后重启")
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=DEEPSEEK_TIMEOUT,
            max_retries=2,  # 网络抖动时自动重试最多 2 次（含首次共 3 次尝试）
        )
    return _client


async def ask_deepseek(question: str) -> str | None:
    """向 DeepSeek 提问。

    成功返回回答文本；任何失败（超时、网络错误、Key 错误、
    返回为空等）都返回 None，由调用方决定如何提示用户。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        # 异步调用，不阻塞事件循环
        response = await _get_client().chat.completions.create(
            model=model,
            messages=messages,
        )

        if not response.choices:
            logger.warning("[AI CHAT] DeepSeek 返回为空（没有 choices）")
            return None

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            logger.warning("[AI CHAT] DeepSeek 返回内容为空")
            return None

        return answer.strip()

    except Exception as exc:  # 兜底捕获所有异常，保证 Bot 进程不崩溃
        # 只记录异常类型和简要说明；先经 redact_secrets 清洗，绝不把 API Key 带进日志
        # 注意：NoneBot2 使用 loguru，占位符是 {} 风格而不是 %s
        logger.error(
            "[AI CHAT] DeepSeek error: {}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None
