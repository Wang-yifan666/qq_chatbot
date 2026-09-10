"""DeepSeek API 客户端封装（纯 LLM Transport 层）。

本模块只负责一件事：把调用方构造好的 messages 发给 DeepSeek，返回文本回答。
不负责人格、上下文、QQ 用户信息、群聊格式化（这些统一由
services/prompt_builder.py 负责，保证主备服务商收到完全相同的 Prompt）。

要点：
- 使用 OpenAI 官方 SDK 的 AsyncOpenAI（异步客户端），不会阻塞
  NoneBot2 的 asyncio 事件循环；
- API Key 从环境变量 DEEPSEEK_API_KEY 读取，严禁硬编码；
- 模型名从环境变量 DEEPSEEK_MODEL 读取，未配置时使用默认值 deepseek-flash
  （V4.1 Flash，原生支持 text + image 多模态）；
- 客户端懒加载：首次调用时才创建。这样当 .env 里 AI_PROVIDER=zhipu
  （即本次不选用 DeepSeek）时，缺少 DEEPSEEK_API_KEY 也不会影响 Bot 启动；
- Transport 层对 content 不做任何处理：字符串 content 与 multimodal list
  content（含 image_url block）都原样交给 SDK，绝不 str() 化、绝不 json.dumps。
"""

import os

from nonebot import logger
from openai import AsyncOpenAI

from services import redact_secrets
from services.tool_orchestrator import RawCompletion

# DeepSeek 兼容 OpenAI 接口，只需把 base_url 指向 DeepSeek
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 单次请求超时（秒），包含建立连接 + 等待回复。
# DeepSeek 偶发响应较慢，60 秒比较稳妥；如觉得太久可改小。
DEEPSEEK_TIMEOUT = 60.0

# 默认模型名从环境变量读取，不散落在业务代码里；未配置时用默认值
# DeepSeek 当前推荐模型名：deepseek-flash（V4.1 Flash，支持 text + image）；
# deepseek-chat（V4 Pro 的兼容别名，text-only）。
# 注意：deepseek-v4-flash-vision-exp 属于上一代 Vision Exp，仅作兼容 alias 保留，
# 不要作为新功能的模型名。
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-flash"

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


async def call_deepseek(
    messages: list[dict],
    model: str | None = None,
    tools: list[dict] | None = None,
) -> RawCompletion | None:
    """原始调用：返回内容 + 可能的 tool_calls（供 Tool Orchestrator 使用）。

    tools 为 OpenAI 兼容工具定义列表；None 表示不启用工具。
    messages 的 content 可能是 str 或 multimodal list（含 image_url block）：
    Transport 层一律原样透传，绝不做 str() / json.dumps 转换。
    任何失败返回 None（异常只记日志，Bot 不崩溃）。
    """
    selected_model = model or DEFAULT_MODEL
    try:
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        response = await _get_client().chat.completions.create(
            model=selected_model,
            messages=messages,
            **kwargs,
        )

        if not response.choices:
            logger.warning("[AI CHAT] DeepSeek 返回为空（没有 choices）")
            return None

        message = response.choices[0].message
        content = (message.content or "").strip() or None
        tool_calls = [
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments or "{}",
            }
            for call in (message.tool_calls or [])
        ]
        if not content and not tool_calls:
            logger.warning("[AI CHAT] DeepSeek 返回内容为空")
            return None
        return RawCompletion(content=content, tool_calls=tool_calls)

    except Exception as exc:  # 兜底捕获所有异常，保证 Bot 进程不崩溃
        # 只记录异常类型和简要说明；先经 redact_secrets 清洗，绝不把 API Key 带进日志
        # 注意：NoneBot2 使用 loguru，占位符是 {} 风格而不是 %s
        logger.error(
            "[AI CHAT] DeepSeek error: {}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None


async def ask_deepseek(
    messages: list[dict],
    model: str | None = None,
) -> str | None:
    """把构造好的 messages 发给 DeepSeek（无工具路径，兼容旧调用）。

    model：本次调用使用的模型名；None 时使用默认模型（DEEPSEEK_MODEL，
    未配置则 deepseek-flash）。同服务商双模型降级时由调用方传入。
    """
    raw = await call_deepseek(messages, model=model)
    return raw.content if raw else None
