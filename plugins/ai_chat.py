"""AI 聊天插件（v0.3.1）：群里 @机器人 → 用户状态 + 短期上下文 + 人格 → AI 模型 → 群回复。

处理顺序（同一群内的 @ 处理通过 per-group asyncio.Lock 串行）：
0. 群访问白名单（fail-closed）：未授权群直接丢弃——不读取/不记录问题正文、
   不回复、不调用 AI、不触发 fallback、不落库、不启动任何后台任务；
1. 只处理 QQ 群消息（GroupMessageEvent），只有 @机器人 才触发（to_me 规则）；
2. 提取纯文本问题与 image segment（v0.5 DIRECT Vision）：
   问题与图片都为空时保持 v0.1 行为（回复提示，不调用 API）；
   有图片时正常进入 AI pipeline（图片只进最后一个 user 消息，
   Context 只存文字占位符）；
3. 用户身份 upsert（user_id 稳定身份，nickname 只是显示名）；
4. 先读取该群最近 CONTEXT_MESSAGE_LIMIT 条历史（旧 Context）；
5. 保存当前用户问题（role=user）；
6. 读取该用户本群长期记忆（user_id + group_id 双重隔离）；
7. 计算有效关系（close 为运行时派生状态，唯一来源 CLOSE_USER_ID）；
8. 从最近群聊提取参与者，构造 Relationship Context（亲近倾向，多人偏向）；
9. Mini-RAG：检索本群个人资料（memory_retriever，失败降级为无记忆对话）；
10. Persona RAG：检索夜子人格语料参考（本地 NumPy 索引，to_thread 执行，
    失败降级为无参考，绝不影响主链路）；
11. 用「人格 + 可信状态 + 亲近倾向 + Personal Memory + Persona RAG + 旧 Context + 当前问题」
    只构造一次 messages；
12. 主服务商失败时，用完全相同的 messages 降级到备用服务商；
13. 成功后在 chat.finish() 之前：保存机器人回答（role=assistant）
    —— finish 会结束当前 Handler，保存代码绝不能写在 finish 之后；
14. 有效互动计数原子 +1（@ 且成功得到回答才计数；普通群聊不加关系进度）；
15. 后台异步尝试长期记忆提取（不阻塞回复；失败只记日志）。

群消息入库分工（与 plugins/context_recorder.py 配合）：
- 普通非 @ 群消息 → context_recorder（priority=20）入库（同时 upsert 用户）；
- @机器人 的消息 → 本插件（priority=10, block=True）拦截，由本插件自己保存，
  保证每条消息最多保存一次；
- 两个插件都在处理器最前面执行同一个群访问白名单检查（services/group_access.py）：
  未授权群的任何消息（无论是否 @机器人）都直接丢弃，不写 messages / users /
  relationships / user_memories，也不产生任何 AI 调用。
"""

import asyncio
import re

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent

from services import log_message_content_enabled
from services import redact_secrets
from services import safe_log_text
from services import persona_rag
from services import knowledge_rag
from services.affection_store import collect_participant_ids
from services.affection_store import affection_level
from services.affection_store import get_affection
from services.affection_store import get_relationship_context
from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import add_message
from services.context_store import get_recent_messages
from services.context_arbitration import should_inject_personal_memory
from services.group_access import is_group_allowed
from services.group_conversation import cancel_pending_ambient
from services.group_conversation import get_group_conversation_state
from services.interaction_profile import build_interaction_profile
from services.interaction_profile import build_profile_block
from services.trigger_intensity import TRIGGER_EMOTIONAL_DISCLOSURE
from services.trigger_intensity import TRIGGER_NONE
from services.trigger_intensity import assess_trigger
from services.llm_client import AI_PROVIDER
from services.llm_client import TOOLS
from services.llm_client import ask_with_fallback
from services.memory_extractor import extract_memories
from services.memory_retriever import MEMORY_MAX_CHARS
from services.memory_retriever import MEMORY_TOP_K
from services.memory_retriever import format_memory_context
from services.memory_retriever import retrieve_memories
from services.memory_store import USER_MEMORY_LIMIT
from services.memory_store import add_memory
from services.memory_store import get_user_memories
from services.perception.content import FileContent
from services.perception.content import ForwardContent
from services.perception.content import ImageContent
from services.perception.message_resolver import MessageResolver
from services.perception.message_resolver import ResolvedConversation
from services.perception.multimodal_builder import ConversationContent
from services.perception.multimodal_builder import file_summary
from services.perception.multimodal_builder import forward_summary
from services.perception.multimodal_builder import reply_summary
from services.prompt_builder import BOT_NAME
from services.prompt_builder import CurrentUser
from services.prompt_builder import DirectConversationContent
from services.prompt_builder import build_messages
from services.prompt_builder import sender_display_name
from services.relationship_service import get_effective_relationship
from services.relationship_service import record_direct_interaction
from services.reply_splitter import SPLIT_REPLY_DELAY_MS
from services.reply_splitter import SPLIT_REPLY_ENABLED
from services.reply_splitter import split_reply
from services.runtime_context import TIMEZONE
from services.runtime_context import get_now
from services.user_store import upsert_user
from services.vision import VISION_ALL_FAILED_REPLY
from services.vision import VISION_DISABLED_REPLY
from services.vision import VISION_ENABLED
from services.vision import VISION_READ_FAILED_REPLY
from services.vision import build_normalized_context_text

# 记忆提取的最小问题长度（太短的寒暄不值得多花一次 LLM 调用）
_MEMORY_EXTRACT_MIN_LEN = 6
# 记忆提取单次调用超时（秒），超时放弃，不影响回复
_MEMORY_EXTRACT_TIMEOUT = 20.0


async def _rule_direct_mention(event: GroupMessageEvent) -> bool:
    """DIRECT 触发规则：@机器人 或 回复机器人。

    为什么不用 to_me()：NapCat v4.18.19 在「图片段在前」的群消息里会漏发
    OneBot 事件的 to_me 字段（实测：image+at+text 的原始 JSON 没有 to_me 键），
    nonebot-adapter-onebot 对缺失字段按 False 处理，导致带图 @ 无法触发。
    这里在 to_me 缺失时兜底检查消息里的 at 段，同时保留回复机器人语义。
    """
    if event.to_me:
        return True
    self_id = str(event.self_id)
    if any(
        getattr(seg, "type", None) == "at"
        and str((getattr(seg, "data", None) or {}).get("qq", "")) == self_id
        for seg in event.message
    ):
        return True
    if event.reply is not None and event.reply.sender.user_id == event.self_id:
        return True
    return False


# 消息事件匹配器：
# - rule=_rule_direct_mention：@机器人（或回复机器人）；含 NapCat 漏发 to_me 的兜底
# - priority=10：比 context_recorder 的 20 更先执行
# - block=True：处理完后不再交给后续低优先级处理器（context_recorder 不重复保存）
chat = on_message(rule=_rule_direct_mention, priority=10, block=True)

# 同一群的 @ 处理串行（不同群互不阻塞）：锁来自 services/group_conversation.py，
# DIRECT / AMBIENT / SCHEDULED 三种模式共用同一把 per-group 锁。
# 两个 @ 问题同时到达时，若完全并行，问题 2 读 Context 时可能还没看到问题 1 的
# 机器人回答；per-group 锁保证同群处理顺序稳定。


@chat.handle()
async def handle(event: GroupMessageEvent, bot: Bot):
    """DIRECT 处理器（v0.7）：授权 → Message Resolver → Conversation Pipeline。

    bot 由 NoneBot2 依赖注入（OneBot V11 Bot），只用于 Reply / Forward 的
    get_msg / get_forward_msg 只读查询；插件层不直接发送消息
    （发送仍走 matcher 的 finish / send）。
    """
    # 0. 群访问白名单（fail-closed）：必须在读取 / 输出用户问题内容之前判断。
    #    未授权群直接结束处理：不回复任何内容（包括「有什么想问我的？」）、
    #    不调用 AI / fallback、不 upsert 用户、不读写 Context / Memory、
    #    不增加关系计数、不启动 memory extractor，不产生任何 API 费用。
    #    本匹配器 block=True：即使这里直接 return，事件也不会再传播到
    #    context_recorder（priority=20），因此未授权消息不会入库。
    #    日志只输出群号，绝不输出该群的聊天正文。
    if not is_group_allowed(event.group_id):
        logger.info(
            "[GROUP ACCESS] ignored unauthorized group group_id={}",
            event.group_id,
        )
        return

    # 0.5 DIRECT 优先：这是授权群的真正 direct interaction（matcher rule 已保证
    #     to_me），立即取消该群 pending 的 AMBIENT 等待任务——
    #     不让旧 timer 到点后白跑一次 decision LLM，最后才被冷却/锁挡掉。
    cancel_pending_ambient(event.group_id)

    # get_plaintext() 只保留纯文本，自动去掉 @ 本体和所有 CQ Code。
    question = event.get_plaintext().strip()

    # 0.55 \ping 快速在线自检：不调用 AI、零费用、不进关系计数。
    #     回复当前 BOT_TIMEZONE 时间，确认「QQ 在线 + 机器人进程存活 + 时钟正确」。
    #     v0.6：时间标签与真实时间源一致——Asia/Shanghai 显示「北京时间」，
    #     其它时区显示真实配置名（绝不硬编码北京时间）。
    if re.fullmatch(r"[\\/]ping", question, flags=re.IGNORECASE):
        now = get_now()
        tz_label = "北京时间" if TIMEZONE == "Asia/Shanghai" else TIMEZONE
        reply = f"在。{now.strftime('%H:%M:%S')}（{tz_label}）"
        await add_message(
            group_id=event.group_id,
            user_id=event.self_id,
            nickname=BOT_NAME,
            role="assistant",
            content=reply,
        )
        await chat.finish(reply)

    # 0.6 统一消息解析（v0.7）：白名单已通过才读取任何消息内容 / 图片 URL / 文件。
    #     Message Resolver 负责 text / image / reply / forward / file 的全部解释，
    #     插件层不再针对这些类型写 if/else 分支；日志只记数量统计，
    #     绝不输出图片 URL / Base64 / CDN token / 文件正文 / 转发正文。
    resolver = MessageResolver(bot)
    try:
        resolved = await resolver.resolve_event(event)
        conversation = await resolver.build_conversation(resolved)
    except Exception as exc:
        # 解析层任何意外都降级为“纯文本 + 无多媒体”，绝不因感知失败而不回复。
        logger.error(
            "[RESOLVER] 解析失败，降级为纯文本（{}）：{}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        resolver.cleanup()
        resolved = None
        conversation = None

    question = event.get_plaintext().strip()
    # 归一化后的文本（保持原始顺序，含引用 / 转发 / 文件的结构化占位）：
    # 这才是交给模型的“用户消息文本”；question 保持“用户真正打出来的字”，
    # 用于长期记忆提取（绝不从引用 / 转发 / 文件正文里提取记忆）。
    normalized_text = question
    if resolved is not None:
        normalized = resolved.message.text()
        if normalized:
            normalized_text = normalized
    image_total = resolved.image_total if resolved is not None else 0
    accepted_images = resolved.image_accepted if resolved is not None else 0
    logger.info(
        "[VISION] group_id={} user_id={} enabled={} images_total={} accepted={}",
        event.group_id,
        event.user_id,
        VISION_ENABLED,
        image_total,
        accepted_images,
    )

    # 收到 @ 消息的基础日志（隐私：默认只记长度，绝不默认打印问题正文；严禁打印 API Key）
    if log_message_content_enabled():
        logger.info(
            "[AI CHAT] provider={} group_id={} user_id={} question_chars={} question={}",
            AI_PROVIDER,
            event.group_id,
            event.user_id,
            len(question),
            safe_log_text(question),
        )
    else:
        logger.info(
            "[AI CHAT] provider={} group_id={} user_id={} question_chars={}",
            AI_PROVIDER,
            event.group_id,
            event.user_id,
            len(question),
        )

    # 只 @ 了机器人、没有任何正文也没有任何图片：保持 v0.1 行为，直接提示，不调用 API。
    # 与其它模式一致：机器人实际发出的这句话也要写进 Context（只写一次），
    # 并且让 has_recent_bot_message 生效——DIRECT 刚结束时 AMBIENT 不会马上插话。
    if resolved is None or resolved.message.is_empty():
        resolver.cleanup()
        await add_message(
            group_id=event.group_id,
            user_id=event.self_id,
            nickname=BOT_NAME,
            role="assistant",
            content="有什么想问我的？",
        )
        await chat.finish("有什么想问我的？")

    # 只有图片、没有文字，但一张图都不可用（视觉关闭 / 全部被拒）：
    # 给稳定、明确的降级回复，而不是假装没收到。
    if not question and image_total > 0 and accepted_images == 0:
        resolver.cleanup()
        reply = VISION_DISABLED_REPLY if not VISION_ENABLED else VISION_READ_FAILED_REPLY
        await add_message(
            group_id=event.group_id,
            user_id=event.self_id,
            nickname=BOT_NAME,
            role="assistant",
            content=reply,
        )
        await chat.finish(reply)

    # 其余情况（有文字、或文字+图、或纯图片且图片可用、或引用/转发/文件）
    # 都进入同一条 AI pipeline。
    try:
        answer = await _answer(
            event,
            normalized_text,
            conversation=conversation,
            resolved=resolved,
        )
    finally:
        # 图片类文件下载到本机临时目录：无论成功失败都必须删除（绝不残留）。
        resolver.cleanup()

    # 多自然段拆成多条 QQ 消息（防刷屏）：前 N-1 条用 send，最后一条用 finish。
    # 注意：SQLite 里的 assistant 回答始终只保存完整原始 answer 一次（在 _answer 内）。
    if SPLIT_REPLY_ENABLED:
        parts = split_reply(answer)
        for part in parts[:-1]:
            await chat.send(part)
            if SPLIT_REPLY_DELAY_MS > 0:
                await asyncio.sleep(SPLIT_REPLY_DELAY_MS / 1000)
        await chat.finish(parts[-1])
    else:
        await chat.finish(answer)


async def _answer(
    event: GroupMessageEvent,
    normalized_text: str,
    conversation: ConversationContent | None = None,
    resolved: ResolvedConversation | None = None,
) -> str:
    """用户状态 → 旧 Context → 保存问题 → 记忆/关系 → 构造 Prompt → 调模型 → 保存。

    整个流程持本群专用锁执行（与 AMBIENT / SCHEDULED 共用同一把锁）：
    同一群的 @ 问题串行处理；不同群锁独立，互不阻塞。
    返回要发给群里的最终文本（成功回答 / 统一的服务不可用提示）。

    normalized_text：v0.7 归一化后的用户消息文本（含引用 / 转发 / 文件占位）。
    conversation / resolved：v0.7 统一 Message Resolver 的输出
    （有序 multimodal blocks + 引用 / 转发 / 文件 DATA + 程序资源提示）。
    后两者为 None 时退化为纯文本请求，行为与旧版本一致。
    """
    # 记忆提取 / 个人资料检索 / Persona RAG 只用“用户真正打出来的字”：
    # 引用、转发、文件正文属于不可信数据，既不进记忆，也不该影响检索相关性。
    plain_question = _extract_plain_question(normalized_text)
    async with get_group_conversation_state(event.group_id).lock:
        user_id = event.user_id
        group_id = event.group_id
        nickname = sender_display_name(event)

        # 1. 用户身份：user_id 是稳定身份，nickname 只是显示名（可更新）
        await upsert_user(user_id, nickname)

        # 2. 先读旧 Context（读完才保存当前问题，避免当前问题在 Prompt 中出现两遍）
        history = await get_recent_messages(group_id, CONTEXT_MESSAGE_LIMIT)

        # 3. 保存当前用户问题；失败只记日志（store 内部处理），不影响本轮回答。
        #    图片 / 文件 / 转发只以结构化短占位入库：
        #    绝不写 URL / Base64 / CDN token / 文件正文 / 文件临时路径 / 转发正文。
        await add_message(
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            role="user",
            content=_context_placeholder(normalized_text, resolved),
        )

        # 4. 该用户在本群的长期记忆（user_id + group_id 双重隔离）
        memories = await get_user_memories(user_id, group_id, USER_MEMORY_LIMIT)

        # 5. 有效关系：close 是运行时派生状态（唯一来源 CLOSE_USER_ID），
        #    数据库里永远只有 base_level
        relationship = await get_effective_relationship(user_id)

        # 5.5 Interaction Profile（v0.8）：relationship × affection → 确定性社交画像。
        #     affection 只在程序层被转换成语义等级（very_close / normal / …），
        #     裸数值 0~100 永远不进 Prompt。失败一律回落 normal（普通），
        #     画像生成是纯函数，不会让任何一轮聊天因为数据库问题而失败。
        affection = "normal"
        try:
            score = await get_affection(group_id, user_id)
            affection = affection_level(score)
        except Exception as exc:
            logger.error(
                "[PROFILE] 读取亲近倾向失败（按 normal 处理）group_id={} user_id={}: {}: {}",
                group_id,
                user_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
        interaction_profile = build_interaction_profile(relationship, affection)
        if log_message_content_enabled():
            logger.info(
                "[PROFILE] group_id={} user_id={} relationship={} affection={} profile={}",
                group_id,
                user_id,
                relationship,
                affection,
                interaction_profile.as_dict(),
            )
        else:
            logger.info(
                "[PROFILE] group_id={} user_id={} relationship={} affection={}",
                group_id,
                user_id,
                relationship,
                affection,
            )

        # 5.6 Trigger Intensity（v0.9）：这一轮的事件强度。
        #     画像回答"这个人被允许靠近多少"（耐心），强度回答
        #     "刚才这件事值不值得真的动情绪"——包括书 / 故事引起的兴趣、
        #     对方的疲惫与失败、自尊受刺激、越界与连续纠缠。
        #     传入 history：文字纠缠（同一句话反复刷）只能从历史里看出来。
        trigger = assess_trigger(
            plain_question,
            interaction_profile,
            history=history,
            user_id=user_id,
        )
        if log_message_content_enabled():
            logger.info(
                "[TRIGGER] group_id={} user_id={} category={} intensity={} ceiling={}",
                group_id,
                user_id,
                trigger.category,
                trigger.intensity,
                trigger.ceiling,
            )
        else:
            logger.info(
                "[TRIGGER] group_id={} user_id={} category={} intensity={}",
                group_id,
                user_id,
                trigger.category,
                trigger.intensity,
            )
        # 6. Relationship Context：从最近群聊中提取参与者，
        #    按亲近倾向（affection）排序后注入 Prompt（多人场景下的隐式人格偏置）。
        #    数据库不可用时返回空块，降级为无偏向的普通对话。
        participant_ids = collect_participant_ids(history, user_id)
        relationship_context = await get_relationship_context(
            group_id, participant_ids, user_id
        )

        # 7. Mini-RAG：检索本群个人资料（增强能力）。
        #    v0.8 上下文仲裁：先做“这一轮是否值得动用个人背景”的确定性前置判断——
        #    “在吗 / 嗯 / 谢谢 / 我去睡了”这类消息不注入任何个人资料，
        #    避免把长期记忆变成每轮都要汇报的待办清单（记得 ≠ 必须提）。
        #    v0.9 例外：**披露疲惫 / 难受 / 失败**的一轮必须带上背景——
        #    这类话本身往往就是寒暄式短句（“累死了”“我输了”），
        #    如果按寒暄静默处理，夜子就失去了“在该关心的时候关心”的能力。
        #    记忆库故障时记录 [MEMORY] retrieve failed 并降级为无 Memory 的普通对话，
        #    绝不让 Memory 数据库故障导致聊天功能整体不可用。
        emotional_turn = trigger.category == TRIGGER_EMOTIONAL_DISCLOSURE
        use_personal_memory = should_inject_personal_memory(plain_question) or emotional_turn
        if not use_personal_memory:
            logger.info(
                "[RAG] 本轮消息无实质内容（寒暄/应答），跳过个人资料注入 group_id={} user_id={}",
                group_id,
                user_id,
            )
        elif emotional_turn and not should_inject_personal_memory(plain_question):
            logger.info(
                "[RAG] 本轮是情绪披露，例外注入个人资料 group_id={} user_id={}",
                group_id,
                user_id,
            )
        try:
            retrieved = (
                await retrieve_memories(group_id, user_id, plain_question, MEMORY_TOP_K)
                if use_personal_memory
                else []
            )
            memory_context = format_memory_context(retrieved, MEMORY_MAX_CHARS) or None
            if retrieved:
                if log_message_content_enabled():
                    logger.info(
                        "[RAG] group_id={} user_id={} query_chars={} query={} retrieved={}",
                        group_id,
                        user_id,
                        len(plain_question),
                        safe_log_text(plain_question),
                        len(retrieved),
                    )
                else:
                    logger.info(
                        "[RAG] group_id={} user_id={} query_chars={} retrieved={}",
                        group_id,
                        user_id,
                        len(plain_question),
                        len(retrieved),
                    )
        except Exception as exc:
            logger.error(
                "[MEMORY] retrieve failed: {}: {}",
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
            memory_context = None

        # 7.5 Persona RAG：检索夜子人格语料参考（风格参考，不是记忆/事实）。
        #      embedding 推理是同步 CPU 计算，用 to_thread 避免阻塞事件循环；
        #      任何失败（模型缺失 / 索引缺失 / 维度不匹配等）都降级为无参考，
        #      绝不让 Persona RAG 故障使 Bot 掉线。
        persona_refs = []
        if persona_rag.PERSONA_RAG_ENABLED:
            try:
                persona_refs = await asyncio.to_thread(
                    persona_rag.retrieve, plain_question, relationship, history
                )
                if persona_refs:
                    logger.info(
                        "[PERSONA RAG] group_id={} user_id={} relationship={} refs={}",
                        group_id,
                        user_id,
                        relationship,
                        len(persona_refs),
                    )
            except Exception as exc:
                logger.exception(
                    "[PERSONA RAG] retrieval failed（降级为无参考，Bot 正常回答）：{}: {}",
                    type(exc).__name__,
                    redact_secrets(str(exc)),
                )
                persona_refs = []

        # 7.6 知识库 RAG（v0.9）：按当前问题检索本地资料，作为**参考资料**注入。
        #      与 Persona RAG 是两套独立数据：这里检索的是"外部资料"（文档/手册），
        #      用于回答"关于这份资料"的问题；查不到就完全不注入（空块）。
        #      embedding 是同步 CPU 计算 → to_thread；任何失败都降级为无参考资料。
        knowledge_block = ""
        if knowledge_rag.KNOWLEDGE_RAG_ENABLED and plain_question:
            try:
                knowledge_chunks = await asyncio.to_thread(
                    knowledge_rag.retrieve, plain_question
                )
                if knowledge_chunks:
                    knowledge_block = knowledge_rag.build_knowledge_block(knowledge_chunks)
                    logger.info(
                        "[KNOWLEDGE] group_id={} user_id={} refs={} top_score={:.3f} sources={}",
                        group_id,
                        user_id,
                        len(knowledge_chunks),
                        knowledge_chunks[0].score,
                        sorted({c.source_file for c in knowledge_chunks}),
                    )
            except Exception as exc:
                logger.exception(
                    "[KNOWLEDGE] retrieval failed（降级为无参考资料，Bot 正常回答）：{}: {}",
                    type(exc).__name__,
                    redact_secrets(str(exc)),
                )
                knowledge_block = ""

        # 8. 只构造一次 messages；主备服务商共用，
        #    人格 / 身份 / 关系 / 亲近倾向 / 记忆 / Personal Memory / Persona RAG /
        #    上下文完全一致。
        #    纯图片（plain_question 为空）时，文本块表达程序事实，而不是替用户编问题：
        #    让 Persona Core 决定夜子自然怎么回应（梗图/截图/表情包各有各的回应）。
        #    v0.7：感知层结果通过 DirectConversationContent 传入，
        #    Prompt Builder 只负责分信任区，绝不重新解析 QQ 消息。
        prompt_question = (
            normalized_text
            if normalized_text
            else "用户只发送了图片，没有附加文字。"
        )
        messages = build_messages(
            current_user=CurrentUser(user_id=user_id, display_name=nickname),
            relationship=relationship,
            memories=memories,
            history=history,
            question=prompt_question,
            personal_memory_context=memory_context,
            relationship_context=relationship_context,
            persona_refs=persona_refs,
            conversation_content=_prompt_content(normalized_text, conversation, resolved),
            interaction_profile=interaction_profile,
            trigger=trigger,
            knowledge_block=knowledge_block,
        )

        # 9. 主备调用（capability-aware：含图片时 require_vision=True，
        #     绝不把图片请求发给 text-only 候选；纯文本行为与 v0.4 一致）
        has_images = bool(conversation and conversation.has_any_image)
        answer, used_provider = await ask_with_fallback(
            messages, TOOLS, require_vision=has_images
        )

        if not answer:
            # 视觉请求所有可用候选都失败 / 纯文本主备都失败：
            # 不把任何异常细节或 API Key 发到群里
            if has_images:
                return VISION_ALL_FAILED_REPLY
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
        #     只用“用户真正打出来的字”，绝不从引用 / 转发 / 文件正文里提取记忆。
        if len(plain_question) >= _MEMORY_EXTRACT_MIN_LEN:
            asyncio.create_task(
                _extract_memories_in_background(user_id, group_id, nickname, plain_question)
            )

        logger.info("[AI CHAT] reply success (provider={})", used_provider)
        return answer


def _extract_plain_question(question: str) -> str:
    """从归一化文本里取“用户真正打出来的字”（用于记忆提取 / RAG 查询）。

    归一化文本可能带有引用、转发、文件的结构化占位；这些内容属于不可信数据，
    绝不能变成用户的长期记忆，也不应影响个人资料检索的相关性。
    """
    lines: list[str] = []
    for line in (question or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # 程序生成的占位行以 [ 开头（[附带 N 张图片] / [发送文件: ...] 等）
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        if stripped.startswith("〖"):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _context_placeholder(
    question: str,
    resolved: ResolvedConversation | None,
) -> str:
    """SQLite 落库占位符（绝不写 URL / Base64 / 文件正文 / 转发正文）。

    形如：

        这个是谁？
        [回复了一条包含图片的消息（1 张图片）]
        [发送了图片文件: a.png]

    没有可写内容时返回空字符串（调用方不会走到这里，因为空消息已提前分支）。
    """
    if resolved is None:
        return question
    file_notes: list[str] = []
    forward_notes: list[str] = []
    image_count = 0
    for item in resolved.message.items:
        if isinstance(item, ImageContent):
            image_count += 1
        elif isinstance(item, FileContent):
            file_notes.append(file_summary(item))
        elif isinstance(item, ForwardContent):
            forward_notes.append(forward_summary(item))
    return build_normalized_context_text(
        question,
        image_count=image_count,
        file_notes=file_notes,
        forward_notes=forward_notes,
        reply_note=reply_summary(resolved.reply),
    )


def _prompt_content(
    normalized_text: str,
    conversation: ConversationContent | None,
    resolved: ResolvedConversation | None,
) -> DirectConversationContent:
    """把感知层输出适配成 Prompt Builder 的 DirectConversationContent。

    conversation 为 None（解析层异常降级）时仍然返回一个纯文本载体，
    保证 messages 结构一致；唯一区别是没有 image block。
    """
    image_total = resolved.image_total if resolved is not None else 0
    if conversation is None:
        return DirectConversationContent(
            text=normalized_text,
            item_blocks=(),
            image_count=image_total,
            has_any_image=False,
            empty=not normalized_text,
        )
    text = conversation.data_text or normalized_text
    return DirectConversationContent(
        text=text,
        item_blocks=tuple(conversation.items),
        reply_block=conversation.reply_text,
        forward_block=conversation.forward_block,
        file_block=conversation.file_block,
        notice=conversation.notice,
        image_count=image_total,
        has_any_image=conversation.has_any_image,
        empty=not text,
    )


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
        text, _ = await ask_with_fallback(messages)
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
