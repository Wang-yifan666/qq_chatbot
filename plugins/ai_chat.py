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

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent

from services import log_message_content_enabled
from services import redact_secrets
from services import safe_log_text
from services import persona_rag
from services.affection_store import collect_participant_ids
from services.affection_store import get_relationship_context
from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import add_message
from services.context_store import get_recent_messages
from services.group_access import is_group_allowed
from services.group_conversation import cancel_pending_ambient
from services.group_conversation import get_group_conversation_state
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
from services.prompt_builder import BOT_NAME
from services.prompt_builder import CurrentUser
from services.prompt_builder import build_messages
from services.prompt_builder import sender_display_name
from services.relationship_service import get_effective_relationship
from services.relationship_service import record_direct_interaction
from services.reply_splitter import SPLIT_REPLY_DELAY_MS
from services.reply_splitter import SPLIT_REPLY_ENABLED
from services.reply_splitter import split_reply
from services.user_store import upsert_user
from services.vision import VISION_ALL_FAILED_REPLY
from services.vision import VISION_DISABLED_REPLY
from services.vision import VISION_ENABLED
from services.vision import VISION_READ_FAILED_REPLY
from services.vision import VisionImage
from services.vision import attach_images_to_last_user_message
from services.vision import convert_unsupported_images
from services.vision import build_context_text
from services.vision import extract_images

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
async def handle(event: GroupMessageEvent):
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

    # 0.6 图片提取（v0.5 DIRECT Vision）：白名单已通过才允许读取 image segment。
    #     日志只记数量统计，绝不输出图片 URL / Base64 / CDN token。
    extraction = extract_images(event)
    accepted_images: list[VisionImage] = extraction.images if VISION_ENABLED else []
    dropped = extraction.rejected + (len(extraction.images) if not VISION_ENABLED else 0)
    # 0.61 格式兼容：DeepSeek 不收的格式（如 BMP）内存中转成 JPEG data URL。
    #      转换失败保留原图（外链直传仍可能成功），不计入 dropped。
    if VISION_ENABLED and accepted_images:
        accepted_images, _convert_failed = await convert_unsupported_images(accepted_images)
    logger.info(
        "[VISION] group_id={} user_id={} enabled={} images_total={} accepted={} rejected={}",
        event.group_id,
        event.user_id,
        VISION_ENABLED,
        extraction.total,
        len(accepted_images),
        dropped,
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
    if not question and extraction.total == 0:
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
    if not question and extraction.total > 0 and not accepted_images:
        reply = VISION_DISABLED_REPLY if not VISION_ENABLED else VISION_READ_FAILED_REPLY
        await add_message(
            group_id=event.group_id,
            user_id=event.self_id,
            nickname=BOT_NAME,
            role="assistant",
            content=reply,
        )
        await chat.finish(reply)

    # 其余情况（有文字、或文字+图、或纯图片且图片可用）都进入 AI pipeline。
    answer = await _answer(
        event,
        question,
        images=accepted_images,
        image_total=extraction.total,
    )

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
    question: str,
    images: list[VisionImage] | None = None,
    image_total: int = 0,
) -> str:
    """用户状态 → 旧 Context → 保存问题 → 记忆/关系 → 构造 Prompt → 调模型 → 保存。

    整个流程持本群专用锁执行（与 AMBIENT / SCHEDULED 共用同一把锁）：
    同一群的 @ 问题串行处理；不同群锁相互独立，互不阻塞。
    返回要发给群里的最终文本（成功回答 / 统一的服务不可用提示）。

    images：通过校验、要交给模型的图片（None/[] = 纯文本请求，行为与 v0.4 一致）；
    image_total：消息里的图片总数（写入 Context 的文字占位符用，绝不写 URL）。
    """
    images = images or []
    async with get_group_conversation_state(event.group_id).lock:
        user_id = event.user_id
        group_id = event.group_id
        nickname = sender_display_name(event)

        # 1. 用户身份：user_id 是稳定身份，nickname 只是显示名（可更新）
        await upsert_user(user_id, nickname)

        # 2. 先读旧 Context（读完才保存当前问题，避免当前问题在 Prompt 中出现两遍）
        history = await get_recent_messages(group_id, CONTEXT_MESSAGE_LIMIT)

        # 3. 保存当前用户问题；失败只记日志（store 内部处理），不影响本轮回答。
        #    图片只以文字占位符入库（[附带 N 张图片]），绝不写 URL / Base64。
        await add_message(
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            role="user",
            content=build_context_text(question, image_total),
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
                if log_message_content_enabled():
                    logger.info(
                        "[RAG] group_id={} user_id={} query_chars={} query={} retrieved={}",
                        group_id,
                        user_id,
                        len(question),
                        safe_log_text(question),
                        len(retrieved),
                    )
                else:
                    logger.info(
                        "[RAG] group_id={} user_id={} query_chars={} retrieved={}",
                        group_id,
                        user_id,
                        len(question),
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
                    persona_rag.retrieve, question, relationship, history
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

        # 8. 只构造一次 messages；主备服务商共用，
        #    人格 / 身份 / 关系 / 亲近倾向 / 记忆 / Personal Memory / Persona RAG /
        #    上下文完全一致。
        #    纯图片（question 为空）时，文本块表达程序事实，而不是替用户编问题：
        #    让 Persona Core 决定夜子自然怎么回应（梗图/截图/表情包各有各的回应）。
        prompt_question = question if question else "用户只发送了图片，没有附加文字。"
        messages = build_messages(
            current_user=CurrentUser(user_id=user_id, display_name=nickname),
            relationship=relationship,
            memories=memories,
            history=history,
            question=prompt_question,
            personal_memory_context=memory_context,
            relationship_context=relationship_context,
            persona_refs=persona_refs,
        )

        # 8.5 视觉：沿用同一套 messages，只把最后一个 user 消息变成 multimodal
        #     （图片只进 user content，绝不进 system / assistant / tool / 历史 DATA）。
        if images:
            messages = attach_images_to_last_user_message(messages, images)

        # 9. 主备调用（capability-aware：含图片时 require_vision=True，
        #     绝不把图片请求发给 text-only 候选；纯文本行为与 v0.4 一致）
        answer, used_provider = await ask_with_fallback(
            messages, TOOLS, require_vision=bool(images)
        )

        if not answer:
            # 视觉请求所有可用候选都失败 / 纯文本主备都失败：
            # 不把任何异常细节或 API Key 发到群里
            if images:
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
