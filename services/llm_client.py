"""统一 LLM 调用层（v0.4）：DIRECT / AMBIENT / SCHEDULED 三种模式共用。

只有这里决定按 AI_PROVIDER / AI_FALLBACK 选择服务商与模型；
聊天触发（ai_chat / ambient）与定时触发（scheduled）不再各自复制 Provider 逻辑。
主备降级使用完全相同的 messages；tools 参数由各模式自己决定
（DIRECT 按 WEB_SEARCH_ENABLED，SCHEDULED / AMBIENT 默认 None = 无工具）。
"""

import os

from nonebot import logger

from services.deepseek import ask_deepseek
from services.deepseek import call_deepseek
from services.tool_orchestrator import WEB_SEARCH_TOOL_SCHEMA
from services.tool_orchestrator import run_with_tools
from services.web_search import WEB_SEARCH_ENABLED
from services.zhipu import ask_glm
from services.zhipu import call_glm

# 服务商名 → 对应的调用函数（两家都是 OpenAI 兼容接口，返回格式一致）
PROVIDER_HANDLERS = {
    "deepseek": ask_deepseek,
    "zhipu": ask_glm,
}

# 服务商名 → 原始调用（返回 content + tool_calls，供 Tool Orchestrator 使用）
PROVIDER_RAW_HANDLERS = {
    "deepseek": call_deepseek,
    "zhipu": call_glm,
}

# 主服务商与备用服务商（在 .env 中配置，修改后需重启生效）。
# 支持两种降级：
# - 跨服务商：AI_PROVIDER=deepseek + AI_FALLBACK=zhipu（或反之）；
# - 同服务商双模型：AI_PROVIDER=AI_FALLBACK=deepseek，
#   主模型 = AI_MODEL（或 DEEPSEEK_MODEL 默认），备用模型 = AI_FALLBACK_MODEL。
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").strip().lower()
AI_FALLBACK = (os.getenv("AI_FALLBACK", "") or "").strip().lower() or None
# 主模型覆盖（留空 = 用服务商默认模型 DEEPSEEK_MODEL / ZHIPU_MODEL）
AI_MODEL = (os.getenv("AI_MODEL", "") or "").strip() or None
# 备用模型覆盖（留空 = 用备用服务商默认模型；同服务商降级时必填）
AI_FALLBACK_MODEL = (os.getenv("AI_FALLBACK_MODEL", "") or "").strip() or None

# DIRECT 模式可用的工具（由程序根据 WEB_SEARCH_ENABLED 决定，聊天内容不能修改）
TOOLS = [WEB_SEARCH_TOOL_SCHEMA] if WEB_SEARCH_ENABLED else None


async def ask(
    provider: str,
    messages: list[dict[str, str]],
    model: str | None = None,
) -> str | None:
    """按服务商名调用对应的 AI 服务；model=None 时使用服务商默认模型。"""
    return await PROVIDER_HANDLERS[provider](messages, model)


async def ask_raw(
    provider: str,
    messages: list[dict],
    model: str | None = None,
    tools: list[dict] | None = None,
):
    """按服务商名调用原始接口（返回 RawCompletion | None，供工具编排使用）。"""
    return await PROVIDER_RAW_HANDLERS[provider](messages, model, tools)


async def ask_with_fallback(
    messages: list[dict[str, str]],
    tools: list[dict] | None = None,
) -> tuple[str | None, str]:
    """主 Provider（主模型）→ 失败用完全相同的 messages 降级备用（备用模型）。

    返回 (answer, 实际使用的 provider)。同服务商双模型降级时，
    主备都指向同一 provider，但分别使用 AI_MODEL 与 AI_FALLBACK_MODEL。
    tools 不为 None 时启用工具编排（主备各自独立编排，工具输出不可信 DATA）。
    """
    used_provider = AI_PROVIDER
    if tools:
        answer = await run_with_tools(
            lambda msgs, t=None: ask_raw(AI_PROVIDER, msgs, AI_MODEL, t),
            messages,
            tools,
        )
    else:
        answer = await ask(AI_PROVIDER, messages, AI_MODEL)

    if not answer and AI_FALLBACK:
        logger.warning(
            "[AI CHAT] 主服务商 {} 调用失败，降级到 {} 重试"
            "（主模型 {}，备用模型 {}）",
            AI_PROVIDER,
            AI_FALLBACK,
            AI_MODEL or "默认",
            AI_FALLBACK_MODEL or "默认",
        )
        used_provider = AI_FALLBACK
        if tools:
            answer = await run_with_tools(
                lambda msgs, t=None: ask_raw(AI_FALLBACK, msgs, AI_FALLBACK_MODEL, t),
                messages,
                tools,
            )
        else:
            answer = await ask(AI_FALLBACK, messages, AI_FALLBACK_MODEL)
    return answer, used_provider
