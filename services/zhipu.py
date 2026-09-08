"""智谱 GLM API 客户端封装。

与 DeepSeek 一样，智谱开放平台也提供 OpenAI 兼容接口，只需更换 base_url。

本模块只负责一件事：把用户问题发给 GLM 模型，返回文本回答。

要点：
- 使用 OpenAI 官方 SDK 的 AsyncOpenAI（异步客户端），不阻塞事件循环；
- API Key 从环境变量 ZHIPU_API_KEY 读取，严禁硬编码；
- 模型名从环境变量 ZHIPU_MODEL 读取，未配置时使用默认值 glm-4.7-flash；
- 客户端懒加载：首次调用时才创建。这样当 .env 里 AI_PROVIDER=deepseek
  （即本次不选用智谱）时，缺少 ZHIPU_API_KEY 也不会影响 Bot 启动；
- 每次请求都是独立对话，不携带任何聊天历史。
"""

import os

from nonebot import logger
from openai import AsyncOpenAI

from services import redact_secrets

# 智谱开放平台 OpenAI 兼容接口地址（文档：https://open.bigmodel.cn）
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 单次请求超时（秒），包含建立连接 + 等待回复
GLM_TIMEOUT = 60.0

# 系统提示词：让 AI 用中文清晰、准确地回答
SYSTEM_PROMPT = "你是一个QQ群里的AI助手，请使用中文清晰、准确地回答用户问题。"

# 模型名从环境变量读取，未配置时用默认值 glm-4.7-flash
model = os.getenv("ZHIPU_MODEL") or "glm-4.7-flash"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """懒加载获取全局唯一的异步客户端（首次调用时才创建，复用底层连接池）。"""
    global _client
    if _client is None:
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            # bot.py 启动时已做同样的检查，这里再兜底一次
            raise RuntimeError("缺少 ZHIPU_API_KEY：请在 .env 中填写后重启")
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=GLM_BASE_URL,
            timeout=GLM_TIMEOUT,
            max_retries=2,  # 网络抖动时自动重试最多 2 次（含首次共 3 次尝试）
        )
    return _client


async def ask_glm(question: str) -> str | None:
    """向智谱 GLM 提问。

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
            logger.warning("[AI CHAT] GLM 返回为空（没有 choices）")
            return None

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            logger.warning("[AI CHAT] GLM 返回内容为空")
            return None

        return answer.strip()

    except Exception as exc:  # 兜底捕获所有异常，保证 Bot 进程不崩溃
        # 只记录异常类型和简要说明；先经 redact_secrets 清洗，绝不把 API Key 带进日志
        logger.error(
            "[AI CHAT] GLM error: {}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None
