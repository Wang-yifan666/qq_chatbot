"""AI 聊天插件：群里 @机器人 → 提取问题 → AI 模型 → 群回复。

流程：
1. 只处理 QQ 群消息（GroupMessageEvent）；
2. 只有 @机器人 才触发（to_me 规则）；
3. 提取 @ 之后的纯文本问题（去掉 QQ 的 CQ Code）；
4. 根据 .env 配置选择服务商：
   - AI_PROVIDER：主服务商（deepseek | zhipu）
   - AI_FALLBACK：备用服务商（deepseek | zhipu | 留空=不降级）
   主服务商调用失败（超时/限流/Key 错误/网络错误/返回为空）时，
   自动降级到备用服务商再试一次；
5. 把回答发回当前群。
"""

import os

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.rule import to_me

from services.deepseek import ask_deepseek
from services.zhipu import ask_glm

# 服务商名 → 对应的调用函数（两家都是 OpenAI 兼容接口，返回格式一致）
PROVIDER_HANDLERS = {
    "deepseek": ask_deepseek,
    "zhipu": ask_glm,
}

# 主服务商与备用服务商（在 .env 中配置，修改后需重启生效）
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").strip().lower()
AI_FALLBACK = (os.getenv("AI_FALLBACK", "") or "").strip().lower() or None

# 消息事件匹配器：
# - rule=to_me()：只有 @机器人（或回复机器人）的消息才进入本处理器
# - priority=10：优先级（数字越小越先执行）
# - block=True：处理完后不再交给后续低优先级处理器
chat = on_message(rule=to_me(), priority=10, block=True)


async def _ask(provider: str, question: str) -> str | None:
    """按服务商名调用对应的 AI 服务，成功返回回答文本，失败返回 None。"""
    return await PROVIDER_HANDLERS[provider](question)


@chat.handle()
async def handle(event: GroupMessageEvent):
    # get_plaintext() 只保留纯文本，自动去掉 @ 本体和所有 CQ Code。
    # 例如 "[CQ:at,qq=xxx] 什么是STM32的DMA？" -> "什么是STM32的DMA？"
    question = event.get_plaintext().strip()

    # 收到 @ 消息的基础日志（严禁打印 API Key）
    # 注意：NoneBot2 使用 loguru，占位符是 {} 风格而不是 %s
    logger.info(
        "[AI CHAT] provider={} group_id={} user_id={} question={}",
        AI_PROVIDER,
        event.group_id,
        event.user_id,
        question,
    )

    # 只 @ 了机器人、后面没有问题：直接提示，不调用 API
    if not question:
        await chat.finish("有什么想问我的？")

    # 先调用主服务商
    used_provider = AI_PROVIDER
    answer = await _ask(AI_PROVIDER, question)

    # 主服务商失败且配置了备用服务商：自动降级再试一次
    if not answer and AI_FALLBACK:
        logger.warning(
            "[AI CHAT] 主服务商 {} 调用失败，降级到 {} 重试",
            AI_PROVIDER,
            AI_FALLBACK,
        )
        used_provider = AI_FALLBACK
        answer = await _ask(AI_FALLBACK, question)

    if not answer:
        # 两个服务商都失败时，不把任何异常细节或 API Key 发到群里
        await chat.finish("AI 服务暂时不可用，请稍后再试。")

    logger.info("[AI CHAT] reply success (provider={})", used_provider)
    await chat.finish(answer)
