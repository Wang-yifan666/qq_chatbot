"""AI 聊天插件（v0.2）：群里 @机器人 → 短期上下文 + 固定人格 → AI 模型 → 群回复。

处理顺序（同一群内的 @ 处理通过 per-group asyncio.Lock 串行）：
1. 只处理 QQ 群消息（GroupMessageEvent），只有 @机器人 才触发（to_me 规则）；
2. 提取纯文本问题；问题为空时保持 v0.1 行为：直接回复提示，不调用 API；
3. 先读取该群最近 CONTEXT_MESSAGE_LIMIT 条历史（旧 Context）；
4. 保存当前用户问题（role=user）；
5. 用「固定人格 + 旧 Context + 当前问题」只构造一次 messages；
6. 主服务商失败时，用完全相同的 messages 降级到备用服务商；
7. 成功后在 chat.finish() 之前保存机器人回答（role=assistant）
   —— finish 会结束当前 Handler，保存代码绝不能写在 finish 之后。

群消息入库分工（与 plugins/context_recorder.py 配合）：
- 普通非 @ 群消息 → context_recorder（priority=20）入库；
- @机器人 的消息 → 本插件（priority=10, block=True）拦截，由本插件自己保存，
  保证每条消息最多保存一次。
"""

import asyncio
import os

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.rule import to_me

from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import add_message
from services.context_store import get_recent_messages
from services.deepseek import ask_deepseek
from services.prompt_builder import BOT_NAME
from services.prompt_builder import build_messages
from services.prompt_builder import sender_display_name
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
# - priority=10：比 context_recorder 的 20 更先执行
# - block=True：处理完后不再交给后续低优先级处理器（context_recorder 不重复保存）
chat = on_message(rule=to_me(), priority=10, block=True)

# 同一群的 @ 处理串行（不同群互不阻塞）。
# 两个 @ 问题同时到达时，若完全并行，问题 2 读 Context 时可能还没看到问题 1 的
# 机器人回答；per-group 锁保证同群处理顺序稳定。锁按需创建，dict 操作无 await、
# 在事件循环内原子，不需要额外的保护锁。
_group_locks: dict[int, asyncio.Lock] = {}


def _get_group_lock(group_id: int) -> asyncio.Lock:
    """获取某个群专用的锁；首次访问时创建。"""
    lock = _group_locks.get(group_id)
    if lock is None:
        lock = _group_locks[group_id] = asyncio.Lock()
    return lock


async def _ask(provider: str, messages: list[dict[str, str]]) -> str | None:
    """按服务商名调用对应的 AI 服务，成功返回回答文本，失败返回 None。"""
    return await PROVIDER_HANDLERS[provider](messages)


@chat.handle()
async def handle(event: GroupMessageEvent):
    # get_plaintext() 只保留纯文本，自动去掉 @ 本体和所有 CQ Code。
    question = event.get_plaintext().strip()

    # 收到 @ 消息的基础日志（严禁打印 API Key）
    logger.info(
        "[AI CHAT] provider={} group_id={} user_id={} question={}",
        AI_PROVIDER,
        event.group_id,
        event.user_id,
        question,
    )

    # 只 @ 了机器人、后面没有问题：保持 v0.1 行为，直接提示，不调用 API
    if not question:
        await chat.finish("有什么想问我的？")

    answer = await _answer(event, question)
    await chat.finish(answer)


async def _answer(event: GroupMessageEvent, question: str) -> str:
    """读旧 Context → 保存当前问题 → 构造 Prompt → 调模型 → 保存回答。

    整个流程持本群专用锁执行：同一群的 @ 问题串行处理，保证后到的问题
    一定能读到先到问题保存的机器人回答；不同群锁相互独立，互不阻塞。
    返回要发给群里的最终文本（成功回答 / 统一的服务不可用提示）。
    """
    async with _get_group_lock(event.group_id):
        nickname = sender_display_name(event)

        # 1. 先读旧 Context（读完才保存当前问题，避免当前问题在 Prompt 中出现两遍）
        history = await get_recent_messages(event.group_id, CONTEXT_MESSAGE_LIMIT)

        # 2. 保存当前用户问题；失败只记日志（context_store 内部处理），不影响本轮回答
        await add_message(
            group_id=event.group_id,
            user_id=event.user_id,
            nickname=nickname,
            role="user",
            content=question,
        )

        # 3. 只构造一次 messages；主备服务商共用，保证人格与上下文完全一致
        messages = build_messages(
            history=history,
            asker_nickname=nickname,
            asker_user_id=event.user_id,
            question=question,
        )

        # 4. 先调主服务商
        used_provider = AI_PROVIDER
        answer = await _ask(AI_PROVIDER, messages)

        # 主服务商失败且配置了备用服务商：用完全相同的 messages 自动降级重试
        if not answer and AI_FALLBACK:
            logger.warning(
                "[AI CHAT] 主服务商 {} 调用失败，降级到 {} 重试",
                AI_PROVIDER,
                AI_FALLBACK,
            )
            used_provider = AI_FALLBACK
            answer = await _ask(AI_FALLBACK, messages)

        if not answer:
            # 两个服务商都失败时，不把任何异常细节或 API Key 发到群里
            return "AI 服务暂时不可用，请稍后再试。"

        # 5. chat.finish() 会结束当前 Handler，因此必须先把回答写入数据库再回复
        await add_message(
            group_id=event.group_id,
            user_id=event.self_id,
            nickname=BOT_NAME,
            role="assistant",
            content=answer,
        )

        logger.info("[AI CHAT] reply success (provider={})", used_provider)
        return answer
