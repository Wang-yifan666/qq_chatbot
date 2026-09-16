"""POKE 互动（v0.6.1）：群聊“戳一戳 / 拍一拍”→ 防刷 → 统一生成管线。

conversation_mode=poke 是第四种 interaction（direct / ambient / scheduled / poke）：

    QQ Notice Event（notice_type=notify, sub_type=poke）
      → plugins/poke.py 门禁：群白名单（fail-closed，最先）→ POKE_ENABLED
        → 只处理群聊 poke → target_id == self_id（只处理“有人戳机器人”）
      → services/poke.py.on_group_poke()
        → cancel_pending_ambient（poke 是明确针对机器人的互动）
        → per-group 锁（与 DIRECT / AMBIENT / SCHEDULED 同一把，同群串行、异群并行）
        → 防刷 cooldown（用户级 + 群级，内存态）
        → relationship / Relationship Context（affection）/ 最近群聊 Context /
          可信 runtime time / Persona RAG
        → POKE generation（conversation_mode=poke，唯一 Persona Core）
        → 回应：最多 1 条短文本 + 最多 1 次 group_poke 戳回
        → Context（结构化文字占位：互动事件 / 戳回动作）

v0.6.1 能力模型（把「收到戳」与「主动戳回」正式拆开）：
- 收到戳（inbound）与短文本回复是核心能力：只要 POKE_ENABLED 就工作；
- 主动戳回（poke_back）是可选能力，依赖 NapCat PacketBackend 发包能力，
  可能因 QQ build × 架构 × NapCat 版本组合而动态不可用；
  PacketBackend 不可用时：文字回复照常、戳回自动消失、日志干净、
  Context 不撒谎（绝不写“戳回了”）、不会持续打失败 API（services/poke_sender.py
  内置三态熔断，TTL 默认 30 分钟，恢复后可自动探测回来）。

第一版边界（v0.6）：
- 只处理群聊 poke（私聊 poke 忽略）；
- 不调用 web_search / Vision / memory extractor：poke 本身没有值得提取的长期事实；
- 不调用 record_direct_interaction()：连续戳机器人不能刷关系等级
  （poke 对关系的影响留给未来单独设计限频计数）；
- poke_back 由程序按独立限频决定（POKE_POKE_BACK_ENABLED +
  POKE_POKE_BACK_COOLDOWN_SECONDS），LLM 只负责生成一句短文本，
  不做 action decision、没有任意工具调用：解析失败 / 模型失败一律 fail-safe
  （宁可不回文字，也绝不执行未知 action）。
"""

import asyncio
import os
import time

from nonebot import logger

from services import redact_secrets
from services.affection_store import collect_participant_ids
from services.affection_store import get_relationship_context
from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import add_message
from services.context_store import get_recent_messages
from services.group_conversation import cancel_pending_ambient
from services.group_conversation import get_group_conversation_state
from services.llm_client import ask_with_fallback
from services.persona_rag import PERSONA_RAG_ENABLED
from services.persona_rag import retrieve as persona_rag_retrieve
from services.poke_sender import send_group_poke
from services.proactive_sender import get_onebot_bot
from services.proactive_sender import save_assistant_message
from services.proactive_sender import send_group_message
from services.prompt_builder import CurrentUser
from services.prompt_builder import build_messages
from services.relationship_service import get_effective_relationship
from services.user_store import get_user
from services.user_store import upsert_user

# ===== 配置（.env；改 .env 需重启生效） =====

POKE_USER_COOLDOWN_SECONDS_DEFAULT = 10.0
POKE_GROUP_COOLDOWN_SECONDS_DEFAULT = 3.0
POKE_POKE_BACK_COOLDOWN_SECONDS_DEFAULT = 60.0
POKE_MAX_REPLY_CHARS_DEFAULT = 60


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("[POKE] {}={} 不是合法布尔值，按 {} 处理", name, raw, default)
    return default


def _env_float(name: str, default: float, low: float, high: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("[POKE] {}={} 不是合法数字，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[POKE] {}={} 超出范围 [{}, {}]，使用默认 {}", name, value, low, high, default)
        return default
    return value


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[POKE] {}={} 不是合法整数，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[POKE] {}={} 超出范围 [{}, {}]，使用默认 {}", name, value, low, high, default)
        return default
    return value


# 进程启动时解析一次
POKE_ENABLED = _env_bool("POKE_ENABLED", default=True)
POKE_USER_COOLDOWN_SECONDS = _env_float(
    "POKE_USER_COOLDOWN_SECONDS",
    POKE_USER_COOLDOWN_SECONDS_DEFAULT,
    1.0,
    3600.0,
)
POKE_GROUP_COOLDOWN_SECONDS = _env_float(
    "POKE_GROUP_COOLDOWN_SECONDS",
    POKE_GROUP_COOLDOWN_SECONDS_DEFAULT,
    0.0,
    3600.0,
)
POKE_POKE_BACK_ENABLED = _env_bool("POKE_POKE_BACK_ENABLED", default=True)
POKE_POKE_BACK_COOLDOWN_SECONDS = _env_float(
    "POKE_POKE_BACK_COOLDOWN_SECONDS",
    POKE_POKE_BACK_COOLDOWN_SECONDS_DEFAULT,
    0.0,
    86400.0,
)
POKE_MAX_REPLY_CHARS = _env_int(
    "POKE_MAX_REPLY_CHARS", POKE_MAX_REPLY_CHARS_DEFAULT, 10, 200
)

# ===== 防刷 / 限频状态（内存态，与 AMBIENT 的频率状态同一策略） =====
# 键：user → (group_id, user_id)；group → group_id。只在通过冷却时写入。

_user_last_accepted: dict[tuple[int, int], float] = {}
_group_last_accepted: dict[int, float] = {}
_poke_back_last: dict[tuple[int, int], float] = {}


def reset_poke_state() -> None:
    """清空全部 poke 内存状态（测试隔离 / 排查用）。"""
    _user_last_accepted.clear()
    _group_last_accepted.clear()
    _poke_back_last.clear()


def _monotonic() -> float:
    """当前单调时钟（独立函数，便于测试注入）。"""
    return time.monotonic()


def _check_cooldowns(group_id: int, user_id: int, now: float) -> str | None:
    """返回命中的冷却类型（'user' / 'group'），未命中返回 None。

    用户级冷却优先：同一个用户在 POKE_USER_COOLDOWN_SECONDS 内连续戳 → 直接忽略；
    群级冷却对全群成员生效：不同用户共用 POKE_GROUP_COOLDOWN_SECONDS 的群级节奏。
    """
    last_user = _user_last_accepted.get((group_id, user_id))
    if (
        last_user is not None
        and POKE_USER_COOLDOWN_SECONDS > 0
        and now - last_user < POKE_USER_COOLDOWN_SECONDS
    ):
        return "user"
    last_group = _group_last_accepted.get(group_id)
    if (
        last_group is not None
        and POKE_GROUP_COOLDOWN_SECONDS > 0
        and now - last_group < POKE_GROUP_COOLDOWN_SECONDS
    ):
        return "group"
    return None


def _record_accepted(group_id: int, user_id: int, now: float) -> None:
    """记录一次通过冷却的合法 poke（作为用户级与群级冷却的起点）。"""
    _user_last_accepted[(group_id, user_id)] = now
    _group_last_accepted[group_id] = now


def _poke_back_allowed(group_id: int, user_id: int, now: float) -> bool:
    """程序决定本次是否戳回：开关 + 独立限频（与 LLM 完全无关）。

    默认同一用户在 POKE_POKE_BACK_COOLDOWN_SECONDS 内最多被戳回一次，
    防止“你戳我我戳你”式连续 poke 互动刷屏。
    """
    if not POKE_POKE_BACK_ENABLED:
        return False
    last = _poke_back_last.get((group_id, user_id))
    if last is None:
        return True
    if POKE_POKE_BACK_COOLDOWN_SECONDS <= 0:
        return True
    return now - last >= POKE_POKE_BACK_COOLDOWN_SECONDS


def _note_poke_back(group_id: int, user_id: int, now: float) -> None:
    _poke_back_last[(group_id, user_id)] = now


# ===== 回复收敛：poke 的回应必须非常短 =====

_POKE_SENTENCE_END = "。！？!?；;…"

# Context 结构化文字占位（DATA，不是 SYSTEM；绝不保存 CQ Code / raw_info / 原始事件 JSON）
POKE_EVENT_CONTEXT_PLACEHOLDER = "[互动事件：该用户戳了机器人一下]"
POKE_BACK_CONTEXT_PLACEHOLDER = "[互动动作：机器人戳回了该用户]"

# Persona RAG 的检索 seed（只是“检索什么风格”的 query，绝不是输出文案）
POKE_QUERY_TEXT = "被群里认识的人戳了一下（拍一拍），夜子怎么自然反应"


def clean_poke_reply(text: str | None, max_chars: int | None = None) -> str:
    """把 LLM 输出收敛为一条短文本；空 / 纯空白 → ''（当作没有文字回复）。

    超长时优先在句末标点处截断（在 max_chars 往前找），找不到则硬截断：
    保证 poke 永远只产生一句短文本，绝不把 DIRECT 式长回答发出去。
    """
    if not text:
        return ""
    limit = max_chars if max_chars is not None else POKE_MAX_REPLY_CHARS
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    cut = limit
    for index in range(limit - 1, max(0, limit - 40), -1):
        if cleaned[index] in _POKE_SENTENCE_END:
            cut = index + 1
            break
    return cleaned[:cut].strip()


# ===== 主流程 =====


async def on_group_poke(group_id: int, user_id: int) -> str:
    """处理一次合法群聊 poke（绝不抛出）。返回本次 action：

    - 'ignore'：被冷却拦截（cooldown_hit=user/group）；
    - 'none'：没有产生任何回应（无 Bot 连接 / LLM 失败且未戳回 / 全部发送失败）；
    - 'text' / 'poke_back' / 'text_and_poke'：实际执行了的回应组合。

    与 DIRECT / AMBIENT / SCHEDULED 共用同一把 per-group 锁：同群串行、异群并行。
    """
    # 1. poke 属于明确针对机器人的互动：立即取消该群 pending 的 AMBIENT 等待任务。
    cancel_pending_ambient(group_id)

    state = get_group_conversation_state(group_id)
    async with state.lock:
        # 2. 防刷冷却（内存态）：冷却期内直接忽略，不调用 LLM、不读不写任何数据。
        now = _monotonic()
        cooldown_hit = _check_cooldowns(group_id, user_id, now)
        if cooldown_hit is not None:
            logger.info(
                "[POKE] cooldown_hit={} group_id={} user_id={} action=ignore",
                cooldown_hit,
                group_id,
                user_id,
            )
            return "ignore"
        _record_accepted(group_id, user_id, now)

        # 3. Bot 可用性检查放在 LLM 之前：没有 OneBot 连接就不花模型费用。
        bot = get_onebot_bot()
        if bot is None:
            logger.warning("[POKE] 无可用 OneBot 连接，跳过回应 group_id={} user_id={}", group_id, user_id)
            return "none"

        # 4. 显示名与用户身份：poke 事件没有昵称字段，优先用 users 表的最近显示名，
        #    查不到才用 QQ 号；upsert 只刷新 last_seen_at，绝不把 QQ 号覆盖好昵称。
        display_name = str(user_id)
        user = await get_user(user_id)
        if user is not None and user.latest_nickname:
            display_name = user.latest_nickname
        await upsert_user(user_id, display_name)  # 失败只记日志，不影响 poke

        # 5. 最近群聊 Context（DB 不可用 → []，允许“没有上下文”）。
        history = await get_recent_messages(group_id, CONTEXT_MESSAGE_LIMIT)

        # 6. poke 事件进入 Context：结构化文字占位（DATA，不是 SYSTEM），
        #    绝不保存 CQ Code / raw_info / 完整原始事件 JSON；失败只记日志。
        try:
            await add_message(
                group_id=group_id,
                user_id=user_id,
                nickname=display_name,
                role="user",
                content=POKE_EVENT_CONTEXT_PLACEHOLDER,
            )
        except Exception as exc:
            logger.error(
                "[POKE] 事件写入 Context 失败（不影响回应）group_id={} user_id={}: {}: {}",
                group_id,
                user_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )

        # 7. 有效关系（close 由 CLOSE_USER_ID 派生）+ Relationship Context（affection）。
        #    任何失败都降级为 stranger / 空块，绝不阻塞 poke。
        relationship = "stranger"
        try:
            relationship = await get_effective_relationship(user_id)
        except Exception as exc:
            logger.error(
                "[POKE] 读取关系失败（按 stranger 处理）group_id={} user_id={}: {}: {}",
                group_id,
                user_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
        relationship_context = ""
        try:
            participant_ids = collect_participant_ids(history, user_id)
            relationship_context = await get_relationship_context(
                group_id, participant_ids, user_id
            )
        except Exception as exc:
            logger.error(
                "[POKE] Relationship Context 失败（降级为空）group_id={} user_id={}: {}: {}",
                group_id,
                user_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )

        # 8. Persona RAG 风格参考（可选，失败 → 无参考）。
        persona_refs: list = []
        if PERSONA_RAG_ENABLED:
            try:
                persona_refs = (
                    await asyncio.to_thread(
                        persona_rag_retrieve, POKE_QUERY_TEXT, relationship, history
                    )
                    or []
                )
            except Exception as exc:
                logger.error(
                    "[POKE] Persona RAG 失败（降级为无参考）group_id={} user_id={}: {}: {}",
                    group_id,
                    user_id,
                    type(exc).__name__,
                    redact_secrets(str(exc)),
                )
                persona_refs = []

        # 9. 程序决定是否戳回（独立限频，与 LLM 无关）。v0.6.1：这里只做
        #    资格判定，不记录戳回时间戳——时间戳只在「实际戳回成功」后记录，
        #     避免 PacketBackend 失败白白消耗 60 秒社交限频窗口。
        poke_back = _poke_back_allowed(group_id, user_id, now)

        # 10. conversation_mode=poke：唯一 Persona Core，无工具 / 无视觉 / 无长期记忆。
        messages = build_messages(
            current_user=CurrentUser(user_id=user_id, display_name=display_name),
            relationship=relationship,
            memories=[],
            history=history,
            question="",
            relationship_context=relationship_context or None,
            persona_refs=persona_refs,
            conversation_mode="poke",
            poke_back=poke_back,
        )

        # 11. LLM：主备 fallback（tools=None）。失败 → 没有文字，但戳回照常执行
        #     （fail-safe：戳回是程序动作，不依赖模型；宁可不回文字，
        #     也绝不执行模型要求的任意 action）。
        text = ""
        provider = "-"
        try:
            raw, provider = await ask_with_fallback(messages, tools=None)
            text = clean_poke_reply(raw)
        except Exception as exc:
            logger.error(
                "[POKE] LLM 调用异常（按无文字处理）group_id={} user_id={}: {}: {}",
                group_id,
                user_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )

        # 12. 发送：最多 1 条短文本 + 最多 1 次戳回；只把“实际发生”的动作写 Context。
        sent_text = False
        if text:
            sent_text = await send_group_message(bot, group_id, text)
            if sent_text:
                try:
                    await save_assistant_message(
                        group_id, getattr(bot, "self_id", None), text
                    )
                except Exception as exc:
                    logger.error(
                        "[POKE] 文字写入 Context 失败（不影响回应）group_id={}: {}: {}",
                        group_id,
                        type(exc).__name__,
                        redact_secrets(str(exc)),
                    )

        sent_poke_back = False
        if poke_back:
            # v0.6.1：send_group_poke 返回结构化结果（能力层自带 PacketBackend
            # 熔断）；只有 result.ok 才写 Context 动作占位与社交限频时间戳——
            # 失败 / 熔断跳过绝不写「机器人戳回了该用户」，避免 Context 撒谎。
            poke_result = await send_group_poke(bot, group_id, user_id)
            sent_poke_back = bool(poke_result.ok)
            if sent_poke_back:
                _note_poke_back(group_id, user_id, now)
                try:
                    await save_assistant_message(
                        group_id, getattr(bot, "self_id", None), POKE_BACK_CONTEXT_PLACEHOLDER
                    )
                except Exception as exc:
                    logger.error(
                        "[POKE] 戳回占位写入 Context 失败（不影响回应）group_id={}: {}: {}",
                        group_id,
                        type(exc).__name__,
                        redact_secrets(str(exc)),
                    )

        if sent_text and sent_poke_back:
            action = "text_and_poke"
        elif sent_text:
            action = "text"
        elif sent_poke_back:
            action = "poke_back"
        else:
            action = "none"

        # 日志隐私：只记 group_id / user_id / action / cooldown_hit / provider /
        # reply_chars；绝不打印 raw_info / 聊天正文 / Prompt / Memory / Persona。
        logger.info(
            "[POKE] group_id={} user_id={} action={} provider={} reply_chars={} poke_back={}",
            group_id,
            user_id,
            action,
            provider,
            len(text),
            sent_poke_back,
        )
        return action
