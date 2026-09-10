"""统一 LLM 调用层（v0.4 → v0.5）：DIRECT / AMBIENT / SCHEDULED 三种模式共用。

只有这里决定按 AI_PROVIDER / AI_FALLBACK 选择服务商与模型；
聊天触发（ai_chat / ambient）与定时触发（scheduled）不再各自复制 Provider 逻辑。
主备降级使用完全相同的 messages；tools 参数由各模式自己决定
（DIRECT 按 WEB_SEARCH_ENABLED，SCHEDULED / AMBIENT 默认 None = 无工具）。

v0.5 capability-aware fallback：
- multimodal 请求（含 image_url block）必须 `require_vision=True`；
- 只有官方明确验证支持图片输入的模型才会接收视觉请求
  （VISION_CAPABLE_MODELS = {deepseek-flash}），其余一律视为 text-only；
- 视觉请求绝不发给 text-only 候选（不硬发等它 400），也绝不偷偷删图降级；
- 所有 vision-capable 候选都失败 → 返回 (None, ...)，由调用方给统一提示；
- 纯文本请求的 fallback 行为与 v0.4 完全一致。
"""

import os

from nonebot import logger

from services.deepseek import DEFAULT_MODEL as DEEPSEEK_DEFAULT_MODEL
from services.deepseek import ask_deepseek
from services.deepseek import call_deepseek
from services.tool_orchestrator import WEB_SEARCH_TOOL_SCHEMA
from services.tool_orchestrator import run_with_tools
from services.web_search import WEB_SEARCH_ENABLED
from services.zhipu import DEFAULT_MODEL as GLM_DEFAULT_MODEL
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

# ===== 视觉能力表（v0.5） =====
# 只有官方明确验证支持 image input 的模型才标记 True；其他 provider / model
# 一律视为 text-only（不猜能力）。deepseek-chat / glm-* 默认 text-only。
VISION_CAPABLE_MODELS = frozenset({"deepseek-flash"})


def effective_model(provider: str, model: str | None) -> str:
    """返回候选实际使用的模型名（model 显式传入优先，否则服务商默认模型）。"""
    if model:
        return model.strip().lower()
    if provider == "deepseek":
        return DEEPSEEK_DEFAULT_MODEL.strip().lower()
    return GLM_DEFAULT_MODEL.strip().lower()


def provider_supports_vision(provider: str, model: str | None) -> bool:
    """该 (provider, model) 候选是否被官方验证支持 image input。"""
    if provider != "deepseek":
        return False
    return effective_model(provider, model) in VISION_CAPABLE_MODELS


def _candidates(require_vision: bool) -> list[tuple[str, str | None]]:
    """主备候选列表；require_vision 时只保留 vision-capable 的候选。

    纯文本请求返回完整候选列表（与 v0.4 行为一致）。
    """
    candidates: list[tuple[str, str | None]] = [(AI_PROVIDER, AI_MODEL)]
    if AI_FALLBACK:
        candidates.append((AI_FALLBACK, AI_FALLBACK_MODEL))
    if not require_vision:
        return candidates
    capable = [
        (provider, model)
        for provider, model in candidates
        if provider_supports_vision(provider, model)
    ]
    if not capable:
        logger.warning(
            "[AI CHAT] 视觉请求没有任何 vision-capable 候选（候选={}），本次不调用模型",
            [(provider, model or "默认") for provider, model in candidates],
        )
    return capable


async def ask(
    provider: str,
    messages: list[dict],
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
    messages: list[dict],
    tools: list[dict] | None = None,
    require_vision: bool = False,
) -> tuple[str | None, str]:
    """主 Provider → 失败降级备用（capability-aware）。

    返回 (answer, 实际使用的 provider)。
    - tools 不为 None 时启用工具编排（主备各自独立编排，工具输出不可信 DATA）；
    - require_vision=True 时只尝试 vision-capable 的候选；没有可用候选直接
      返回 (None, AI_PROVIDER)，由调用方决定提示语（绝不硬发 text-only 模型，
      也绝不偷偷删图）；
    - 纯文本请求（require_vision=False）行为与 v0.4 完全一致。
    """
    candidates = _candidates(require_vision)
    last_provider = AI_PROVIDER
    for index, (provider, model) in enumerate(candidates):
        if require_vision and not provider_supports_vision(provider, model):
            logger.info(
                "[AI CHAT] 视觉请求跳过 text-only 候选 provider={} model={}",
                provider,
                model or "默认",
            )
            continue
        if tools:
            answer = await run_with_tools(
                lambda msgs, t=None: ask_raw(provider, msgs, model, t),
                messages,
                tools,
            )
        else:
            answer = await ask(provider, messages, model)
        last_provider = provider
        if answer:
            return answer, provider
        if index == 0:
            logger.warning(
                "[AI CHAT] 主服务商 {} 调用失败，尝试降级重试（主模型 {}）",
                provider,
                model or "默认",
            )
    return None, last_provider
