"""AI 聊天插件（v0.2.2）：群里 @机器人 → 用户状态 + 短期上下文 + 人格 → AI 模型 → 群回复。

处理顺序（同一群内的 @ 处理通过 per-group asyncio.Lock 串行）：
1. 只处理 QQ 群消息（GroupMessageEvent），只有 @机器人 才触发（to_me 规则）；
2. 提取纯文本问题；问题为空时保持 v0.1 行为：直接回复提示，不调用 API；
3. 用户身份 upsert（user_id 稳定身份，nickname 只是显示名）；
4. 先读取该群最近 CONTEXT_MESSAGE_LIMIT 条历史（旧 Context）；
5. 保存当前用户问题（role=user）；
6. 读取该用户本群长期记忆（user_id + group_id 双重隔离）；
7. 计算有效关系（close 为运行时派生状态，唯一来源 CLOSE_USER_ID）；
8. 从最近群聊提取参与者，构造 Relationship Context（亲近倾向，多人偏向）；
9. Mini-RAG：检索本群个人资料（memory_retriever，失败降级为无记忆对话）；
10. 用「人格 + 可信状态 + 亲近倾向 + Personal Memory + 旧 Context + 当前问题」
    只构造一次 messages；
11. 主服务商失败时，用完全相同的 messages 降级到备用服务商；
12. 成功后在 chat.finish() 之前：保存机器人回答（role=assistant）
    —— finish 会结束当前 Handler，保存代码绝不能写在 finish 之后；
13. 有效互动计数原子 +1（@ 且成功得到回答才计数；普通群聊不加关系进度）；
14. 后台异步尝试长期记忆提取（不阻塞回复；失败只记日志）。

群消息入库分工（与 plugins/context_recorder.py 配合）：
- 普通非 @ 群消息 → context_recorder（priority=20）入库（同时 upsert 用户）；
- @机器人 的消息 → 本插件（priority=10, block=True）拦截，由本插件自己保存，
  保证每条消息最多保存一次。
"""

import asyncio
import os

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.rule import to_me

from services import redact_secrets
from services.affection_store import collect_participant_ids
from services.affection_store import get_relationship_context
from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import add_message
from services.context_store import get_recent_messages
from services.deepseek import ask_deepseek
from services.memory_extractor import extract_memories
from services.memory_retriever import MEMORY_MAX_CHARS
from services.memory_retriever import MEMORY_TOP_K
from services.memory_retriever import format_memory_context
from services.memory_retriever import retrieve_memories
from services.memory_store import USER_MEMORY_LIMIT
from services.memory_store import add_memory
from services.memory_store import get_user_memories
from services.prompt_builder import BOT_NAME
from services.prompt_builder import CurrentUser
from services.prompt_builder import build_messages
from services.prompt_builder import sender_display_name
from services.relationship_service import get_effective_relationship
from services.relationship_service import record_direct_interaction
from services.user_store import upsert_user
from services.zhipu import ask_glm

# 服务商名 → 对应的调用函数（两家都是 OpenAI 兼容接口，返回格式一致）
PROVIDER_HANDLERS = {
    "deepseek": ask_deepseek,
    "zhipu": ask_glm,
}

# 主服务商与备用服务商（在 .env 中配置，修改后需重启生效）
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").strip().lower()
AI_FALLBACK = (os.getenv("AI_FALLBACK", "") or "").strip().lower() or None

# 记忆提取的最小问题长度（太短的寒暄不值得多花一次 LLM 调用）
_MEMORY_EXTRACT_MIN_LEN = 6
# 记忆提取单次调用超时（秒），超时放弃，不影响回复
_MEMORY_EXTRACT_TIMEOUT = 20.0

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


async def _ask_with_fallback(
    messages: list[dict[str, str]],
) -> tuple[str | None, str]:
    """主 Provider → 失败用完全相同的 messages 降级备用；返回 (answer, 实际 provider)。"""
    used_provider = AI_PROVIDER
    answer = await _ask(AI_PROVIDER, messages)
    if not answer and AI_FALLBACK:
        logger.warning(
            "[AI CHAT] 主服务商 {} 调用失败，降级到 {} 重试",
            AI_PROVIDER,
            AI_FALLBACK,
        )
        used_provider = AI_FALLBACK
        answer = await _ask(AI_FALLBACK, messages)
    return answer, used_provider


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
    """用户状态 → 旧 Context → 保存问题 → 记忆/关系 → 构造 Prompt → 调模型 → 保存。

    整个流程持本群专用锁执行：同一群的 @ 问题串行处理；不同群锁相互独立，互不阻塞。
    返回要发给群里的最终文本（成功回答 / 统一的服务不可用提示）。
    """
    async with _get_group_lock(event.group_id):
        user_id = event.user_id
        group_id = event.group_id
        nickname = sender_display_name(event)

        # 1. 用户身份：user_id 是稳定身份，nickname 只是显示名（可更新）
        await upsert_user(user_id, nickname)

        # 2. 先读旧 Context（读完才保存当前问题，避免当前问题在 Prompt 中出现两遍）
        history = await get_recent_messages(group_id, CONTEXT_MESSAGE_LIMIT)

        # 3. 保存当前用户问题；失败只记日志（store 内部处理），不影响本轮回答
        await add_message(
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            role="user",
            content=question,
        )

        # 4. 该用户在本群的长期记忆（user_id + group_id 双重隔离）
        memories = await get_user_memories(user_id, group_id, USER_MEMORY_LIMIT)

        # 5. 有效关系：close 是运行时派生状态（唯一来源 CLOSE_USER_ID），
        #    数据库里永远只有 base_level
        relationship = await get_effective_relationship(user_id)

        # 6. Relationship Context：从最近群聊中提取参与者，
        #    按亲近倾向（affection）排序后注入 Prompt（多人场景下的隐式人格偏置）。
        #    数据库不可用时返回空块，降级为无偏向的普通对话。
        participant_ids = collect_participant_ids(history, user_id)
        relationship_context = await get_relationship_context(
            group_id, participant_ids, user_id
        )

        # 7. Mini-RAG：检索本群个人资料（增强能力）。
        #    记忆库故障时记录 [MEMORY] retrieve failed 并降级为无 Memory 的普通对话，
        #    绝不让 Memory 数据库故障导致聊天功能整体不可用。
        try:
            retrieved = await retrieve_memories(group_id, user_id, question, MEMORY_TOP_K)
            memory_context = format_memory_context(retrieved, MEMORY_MAX_CHARS) or None
            if retrieved:
                logger.info(
                    "[RAG] group_id={} user_id={} query={} retrieved={}",
                    group_id,
                    user_id,
                    question,
                    len(retrieved),
                )
        except Exception as exc:
            logger.error(
                "[MEMORY] retrieve failed: {}: {}",
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
            memory_context = None

        # 8. 只构造一次 messages；主备服务商共用，
        #    人格 / 身份 / 关系 / 亲近倾向 / 记忆 / Personal Memory / 上下文完全一致
        messages = build_messages(
            current_user=CurrentUser(user_id=user_id, display_name=nickname),
            relationship=relationship,
            memories=memories,
            history=history,
            question=question,
            personal_memory_context=memory_context,
            relationship_context=relationship_context,
        )

        # 9. 主备调用（同一 messages）
        answer, used_provider = await _ask_with_fallback(messages)

        if not answer:
            # 两个服务商都失败时，不把任何异常细节或 API Key 发到群里
            return "AI 服务暂时不可用，请稍后再试。"

        # 10. chat.finish() 会结束当前 Handler，因此必须先把回答写入数据库再回复
        await add_message(
            group_id=group_id,
            user_id=event.self_id,
            nickname=BOT_NAME,
            role="assistant",
            content=answer,
        )

        # 11. 有效互动计数原子 +1，并按阈值重算 base_level（close 用户同样计数）
        await record_direct_interaction(user_id)

        # 12. 后台异步提取长期记忆（不阻塞回复；失败只记日志，绝不影响主回答）
        if len(question) >= _MEMORY_EXTRACT_MIN_LEN:
            asyncio.create_task(
                _extract_memories_in_background(user_id, group_id, nickname, question)
            )

        logger.info("[AI CHAT] reply success (provider={})", used_provider)
        return answer


async def _extract_memories_in_background(
    user_id: int,
    group_id: int,
    display_name: str,
    question: str,
) -> None:
    """后台任务：LLM 提取长期记忆并入库。

    与主回复解耦：创建任务后立即返回，回复先发出；提取失败只记 ERROR 日志，
    任何异常都不能影响已经发出的回复。
    """

    async def _extract_ask(messages: list[dict[str, str]]) -> str | None:
        text, _ = await _ask_with_fallback(messages)
        return text

    try:
        drafts = await asyncio.wait_for(
            extract_memories(
                question=question,
                display_name=display_name,
                ask_fn=_extract_ask,
            ),
            timeout=_MEMORY_EXTRACT_TIMEOUT,
        )
    except Exception as exc:
        logger.error(
            "[MEMORY] 记忆提取失败（不影响回复）：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return

    saved = 0
    for draft in drafts:
        ok = await add_memory(
            user_id=user_id,
            group_id=group_id,
            memory_type=draft.memory_type,
            content=draft.content,
            importance=draft.importance,
            source_message_id=None,
        )
        if ok:
            saved += 1
    if saved:
        logger.info(
            "[MEMORY] user_id={} group_id={} 新增 {} 条长期记忆",
            user_id,
            group_id,
            saved,
        )
