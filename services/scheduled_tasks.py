"""Scheduled Task 基础设施 + 任务注册表（v0.4）。

“为什么这次需要说话”（Scheduler 决定）与“夜子应该说什么”（Persona 决定）
是两个独立的问题：本模块只负责前者。

架构：
    触发源 B：APScheduler cron（nonebot-plugin-apscheduler，AsyncIOScheduler）
              → SCHEDULED_TASK_SPECS 任务注册表（TaskSpec 声明 + 循环解析）
              → SCHEDULED_TASKS: {task_id: ScheduledTask}
              → setup_scheduled_tasks(driver) 循环注册 cron
                + catch-up 双时机钩子（on_startup + on_bot_connect）
              → execute_scheduled_task(group_id, task, now)（task 驱动，通用）
                  → 白名单检查 → per-group 共享锁（与 DIRECT/AMBIENT 同一把）
                  → 今日状态检查（success/skipped 终态不重试；
                    running 保守不重试；failed 可 reclaim；无记录可 claim）
                  → Bot 可用性检查（未连接：不写记录、当天名额保留）
                  → 原子 claim / reclaim → skip-if-active 判断
                  → Persona RAG 风格参考（可选）
                  → conversation_mode=scheduled 的 messages
                     （唯一 Persona Core = prompt_builder.CORE_PERSONA）
                  → 统一 LLM 调用（默认无工具）→ 主动 QQ send → 写 Context
                  → 更新执行状态

不依赖任何 GroupMessageEvent；不伪造 current_user；没有固定文案；
本地 persona.txt 由 prompt_builder 统一读取，本模块绝不二次读取。

如何新增一个定时任务（三步，不碰执行 / 幂等 / 锁任何代码）：
    1. 在 SCHEDULED_TASK_SPECS 加一行 TaskSpec（prefix / task_id / event_type /
       default_time / allow_tools）；
    2. 在 services/prompt_builder.py 的 SCHEDULED_EVENT_INSTRUCTIONS 加一句
       对应 event_type 的任务事实指令（只描述事实，不写性格）；
    3. （可选）在 EVENT_QUERY_TEXTS 加 Persona RAG 检索 seed。
    .env 侧约定变量名：{PREFIX}_ENABLED / {PREFIX}_TIME / {PREFIX}_GROUP_IDS /
    {PREFIX}_CATCHUP_MINUTES / {PREFIX}_SKIP_IF_ACTIVE_MINUTES。

配置（.env，非法时间 / 非法群号在启动阶段报错退出）：
    SCHEDULED_TASKS_ENABLED=true|false        总开关（默认 false）
    MORNING_GREETING_ENABLED / ...TIME=08:00 / ...GROUP_IDS / ...（默认关闭）
    NIGHT_GREETING_ENABLED  / ...TIME=21:00 / ...GROUP_IDS / ...（默认关闭）
"""

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta

from nonebot import logger

from services import redact_secrets
from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import get_recent_messages
from services.context_store import has_recent_bot_message
from services.group_access import ALLOW_ALL_GROUPS
from services.group_access import ALLOWED_GROUP_IDS
from services.group_access import is_group_allowed
from services.group_access import parse_allowed_group_ids
from services.group_conversation import get_group_conversation_state
from services.llm_client import TOOLS
from services.llm_client import ask_with_fallback
from services.persona_rag import PERSONA_RAG_ENABLED
from services.persona_rag import retrieve as persona_rag_retrieve
from services.proactive_sender import get_onebot_bot
from services.proactive_sender import save_assistant_message
from services.proactive_sender import send_group_message
from services.prompt_builder import ScheduledEvent
from services.prompt_builder import build_messages
from services.runtime_context import TIMEZONE
from services.runtime_context import get_now
from services.scheduled_task_store import TERMINAL_STATUSES
from services.scheduled_task_store import claim_scheduled_task
from services.scheduled_task_store import get_scheduled_task_status
from services.scheduled_task_store import mark_scheduled_task
from services.scheduled_task_store import reclaim_scheduled_task

# ==========================================================================
# 配置解析（进程启动时一次；非法时间 / 非法群号 → ValueError → bot.py 退出）
# ==========================================================================

_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")

CATCHUP_MINUTES_DEFAULT = 30
SKIP_IF_ACTIVE_MINUTES_DEFAULT = 10

# Persona RAG 的检索 seed（只是“检索什么风格”的 query，绝不是输出文案）
EVENT_QUERY_TEXTS = {
    "morning_greeting": "早晨主动和群里的大家打招呼",
    "night_greeting": "晚上主动和群里的大家打招呼",
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("[SCHEDULED] {}={} 不是合法布尔值，按 {} 处理", name, raw, default)
    return default


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[SCHEDULED] {}={} 不是合法整数，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[SCHEDULED] {}={} 超出范围 [{}, {}]，使用默认 {}", name, value, low, high, default)
        return default
    return value


def parse_task_time(env_name: str, raw: str | None) -> tuple[int, int]:
    """解析 {PREFIX}_TIME（HH:MM）。非法 → ValueError（启动报错退出）。

    合法：08:00 / 8:00 / 23:59；非法：24:00 / 8:5 / 08 / abc / 空。
    错误信息只包含环境变量名，不含原始配置值。
    """
    text = (raw or "").strip()
    match = _TIME_RE.fullmatch(text)
    if not match:
        raise ValueError(
            f"{env_name} 配置非法：必须是 HH:MM 格式的 24 小时制时间"
            "（如 08:00，小时 0~23，分钟 00~59），请检查 .env"
        )
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        raise ValueError(
            f"{env_name} 配置非法：必须是 HH:MM 格式的 24 小时制时间"
            "（如 08:00，小时 0~23，分钟 00~59），请检查 .env"
        )
    return hour, minute


def parse_morning_time(raw: str | None) -> tuple[int, int]:
    """向后兼容别名：解析 MORNING_GREETING_TIME。"""
    return parse_task_time("MORNING_GREETING_TIME", raw)


@dataclass(frozen=True)
class ScheduledTask:
    """一个通用定时任务（由 TaskSpec + .env 构建）。"""

    task_id: str
    event_type: str
    enabled: bool
    hour: int
    minute: int
    target_group_ids: frozenset[int]
    allow_tools: bool
    catchup_minutes: int
    skip_if_active_minutes: int


@dataclass(frozen=True)
class TaskSpec:
    """任务声明（注册表的一行）：前缀决定 .env 变量名。

    prefix      .env 变量前缀（MORNING_GREETING → MORNING_GREETING_TIME ...）
    task_id     幂等表 / job id 使用的任务名
    event_type  进入 Prompt 的定时事件类型（对应 SCHEDULED_EVENT_INSTRUCTIONS）
    default_time 未配置 {PREFIX}_TIME 时的默认触发时刻
    allow_tools 工具权限属于任务 capability（默认 False = 无工具）
    """

    prefix: str
    task_id: str
    event_type: str
    default_time: str = "08:00"
    allow_tools: bool = False


# 任务注册表：新增定时任务只需在这里加一行（外加一句指令与可选的检索 seed）。
SCHEDULED_TASK_SPECS = (
    TaskSpec(
        prefix="MORNING_GREETING",
        task_id="morning_greeting",
        event_type="morning_greeting",
        default_time="08:00",
        allow_tools=False,  # 问候不需要联网
    ),
    TaskSpec(
        prefix="NIGHT_GREETING",
        task_id="night_greeting",
        event_type="night_greeting",
        default_time="21:00",
        allow_tools=False,
    ),
)


def _resolve_targets(raw_group_ids: str | None) -> frozenset[int]:
    """解析 {PREFIX}_GROUP_IDS；* → 全部白名单群（白名单本身是 * 时无目标）。"""
    targets, allow_all = parse_allowed_group_ids(raw_group_ids)
    if allow_all:
        return ALLOWED_GROUP_IDS if not ALLOW_ALL_GROUPS else frozenset()
    return targets


def build_task(spec: TaskSpec) -> ScheduledTask:
    """按 TaskSpec + .env 构建任务对象；非法时间 / 群号抛 ValueError。"""
    hour, minute = parse_task_time(
        f"{spec.prefix}_TIME", os.getenv(f"{spec.prefix}_TIME") or spec.default_time
    )
    return ScheduledTask(
        task_id=spec.task_id,
        event_type=spec.event_type,
        enabled=_env_bool(f"{spec.prefix}_ENABLED", default=False),
        hour=hour,
        minute=minute,
        target_group_ids=_resolve_targets(os.getenv(f"{spec.prefix}_GROUP_IDS")),
        allow_tools=spec.allow_tools,
        catchup_minutes=_env_int(
            f"{spec.prefix}_CATCHUP_MINUTES", CATCHUP_MINUTES_DEFAULT, 0, 180
        ),
        skip_if_active_minutes=_env_int(
            f"{spec.prefix}_SKIP_IF_ACTIVE_MINUTES",
            SKIP_IF_ACTIVE_MINUTES_DEFAULT,
            0,
            60,
        ),
    )


SCHEDULED_TASKS_ENABLED = _env_bool("SCHEDULED_TASKS_ENABLED", default=False)

# 启动阶段构建全部任务：非法配置直接抛 ValueError，由 bot.py 捕获后报错退出。
if SCHEDULED_TASKS_ENABLED:
    SCHEDULED_TASKS: dict[str, ScheduledTask] = {
        spec.task_id: build_task(spec) for spec in SCHEDULED_TASK_SPECS
    }
else:
    SCHEDULED_TASKS: dict[str, ScheduledTask] = {}

# 向后兼容别名（morning_greeting 是最常用的任务）
MORNING_GREETING_TASK: ScheduledTask | None = SCHEDULED_TASKS.get("morning_greeting")


def within_catchup_window(now: datetime, task: ScheduledTask) -> bool:
    """纯函数：现在是否落在「今天任务时间 + catch-up 窗口」内。

    - now 早于任务时间（如 07:59）→ False：等 cron 正常触发；
    - 任务时间已过但差值 <= catchup_minutes（如 08:10 启动、窗口 30）→ True；
    - 超出窗口（如 10:30）→ False：不突然补发一条过时任务。
    """
    if task.catchup_minutes <= 0:
        return False
    scheduled_at = now.replace(hour=task.hour, minute=task.minute, second=0, microsecond=0)
    if now < scheduled_at:
        return False
    return (now - scheduled_at) <= timedelta(minutes=task.catchup_minutes)


# ==========================================================================
# 任务执行（task 驱动，任何注册表里的任务通用）
# ==========================================================================


async def _retrieve_persona_refs(task: ScheduledTask, history: list) -> list:
    """可选 Persona RAG 风格参考：失败 / 未启用 → []（绝不阻塞定时任务）。"""
    if not PERSONA_RAG_ENABLED:
        return []
    query = EVENT_QUERY_TEXTS.get(task.event_type) or task.event_type
    try:
        refs = await asyncio.to_thread(
            persona_rag_retrieve, query, "stranger", history
        )
        return refs or []
    except Exception as exc:
        logger.error(
            "[SCHEDULED] Persona RAG 检索失败（降级为无参考）task={}：{}: {}",
            task.task_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return []


async def execute_scheduled_task(
    group_id: int,
    task: ScheduledTask | None = None,
    now: datetime | None = None,
) -> str:
    """执行一次定时任务（幂等，绝不抛出）。返回最终 run 状态。

    状态：
    - disabled / unauthorized      未启用 / 未授权群（不写任何记录）
    - no_bot                       Bot 未连接：claim 之前返回，当天名额保留，
                                   等 on_bot_connect 的 catch-up 窗口内再试
    - done                         今天已确认发送成功（终态，绝不重发）
    - skipped                      今天已决策跳过（skip-if-active 等终态）
    - claimed                      记录处于 running（可能已发出 / 并发竞争失败）：
                                   保守跳过，绝不重试
    - success / failed             本次执行结果；failed 在窗口内可 reclaim 重试
    """
    task = task or MORNING_GREETING_TASK
    if task is None or not task.enabled:
        return "disabled"

    # 1. 白名单（fail-closed）：未授权群永远不发送。
    if not is_group_allowed(group_id):
        logger.info(
            "[SCHEDULED] 未授权群跳过 task={} group_id={}", task.task_id, group_id
        )
        return "unauthorized"

    # 2. 与 DIRECT / AMBIENT 共用同一把 per-group 锁。
    state = get_group_conversation_state(group_id)
    async with state.lock:
        # 3. 拿锁后重新确认任务状态。
        if not task.enabled:
            return "disabled"
        now = now or get_now()
        today = now.strftime("%Y-%m-%d")

        # 4. 今日记录检查（retry 语义）：
        #    success / skipped_* → 终态，今天不再执行；
        #    running → 可能已发出（含崩溃残留），保守不重试；
        #    failed → 尚未真正发送成功，窗口内允许 reclaim 重试；
        #    无记录 → 首次尝试，原子 claim。
        current = await get_scheduled_task_status(task.task_id, group_id, today)
        if current in TERMINAL_STATUSES:
            logger.info(
                "[SCHEDULED] task={} 今日状态={}（终态），跳过 group_id={}",
                task.task_id,
                current,
                group_id,
            )
            return "done" if current == "success" else "skipped"
        if current == "running":
            logger.info(
                "[SCHEDULED] task={} 今日记录处于 running（可能已发送），保守跳过 group_id={}",
                task.task_id,
                group_id,
            )
            return "claimed"

        # 5. Bot 可用性检查必须放在 claim 之前：
        #    NoneBot 先启动、NapCat 后连接的 Reverse WebSocket 场景下，
        #    未连接时绝不写任何记录——当天名额保留，等 on_bot_connect 再试。
        bot = get_onebot_bot()
        if bot is None:
            logger.warning(
                "[SCHEDULED] 无可用 OneBot 连接，task={} 暂不执行（不占用当天名额）group_id={}",
                task.task_id,
                group_id,
            )
            return "no_bot"

        # 6. 原子认领：无记录 → claim；failed → reclaim。
        if current is None:
            if not await claim_scheduled_task(task.task_id, group_id, today):
                logger.info(
                    "[SCHEDULED] task={} claim 失败（并发竞争），跳过 group_id={}",
                    task.task_id,
                    group_id,
                )
                return "claimed"
        else:
            if not await reclaim_scheduled_task(task.task_id, group_id, today):
                logger.info(
                    "[SCHEDULED] task={} reclaim 失败（并发竞争或状态变化），跳过 group_id={}",
                    task.task_id,
                    group_id,
                )
                return "claimed"

        status = "failed"
        try:
            # 7. skip-if-active：最近 X 分钟内说过话（主动/被动）→ 本次跳过（终态）。
            if await has_recent_bot_message(group_id, task.skip_if_active_minutes):
                logger.info(
                    "[SCHEDULED] 最近 {} 分钟内已发言，task={} 跳过 group_id={}",
                    task.skip_if_active_minutes,
                    task.task_id,
                    group_id,
                )
                status = "skipped_active"
                return status

            # 8. 可选最近群聊上下文（DB 不可用 → []，允许“没有上下文”）。
            history = await get_recent_messages(group_id, CONTEXT_MESSAGE_LIMIT)
            persona_refs = await _retrieve_persona_refs(task, history)

            # 9. conversation_mode=scheduled：没有 current_user / current_question。
            event = ScheduledEvent(
                event_type=task.event_type,
                local_datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
                scheduled_time=f"{task.hour:02d}:{task.minute:02d}",
            )
            messages = build_messages(
                None,
                "stranger",
                [],
                history,
                "",
                persona_refs=persona_refs,
                conversation_mode="scheduled",
                scheduled_event=event,
            )

            # 10. 统一 LLM 调用：工具权限属于 task capability——
            #     allow_tools=False（默认）不提供任何工具，Prompt 的 capability
            #     也只会说 web_search=false（prompt_builder 按本次真实能力生成）。
            tools = TOOLS if task.allow_tools else None
            answer, provider = await ask_with_fallback(messages, tools=tools)
            if not answer:
                logger.error(
                    "[SCHEDULED] AI 调用失败，task={} 未发送 group_id={}（failed，窗口内可重试）",
                    task.task_id,
                    group_id,
                )
                return status

            # 11. 主动发送 + 成功后写 Context（role=assistant，与 DIRECT 同一张表，
            #     只写一次；NapCat 的 self-message 回报由 context_recorder 跳过）。
            if not await send_group_message(bot, group_id, answer):
                return status
            await save_assistant_message(group_id, getattr(bot, "self_id", None), answer)

            status = "success"
            logger.info(
                "[SCHEDULED] task={} 已发送 group_id={} provider={} chars={}",
                task.task_id,
                group_id,
                provider,
                len(answer),
            )
        except Exception as exc:
            # 任何异常都不能让 Scheduler 崩溃；状态由 finally 落 failed。
            logger.exception(
                "[SCHEDULED] task={} 执行异常 group_id={}：{}: {}",
                task.task_id,
                group_id,
                type(exc).__name__,
                redact_secrets(str(exc)),
            )
        finally:
            # 保守策略：即使发送成功但状态更新失败，记录仍停在 'running'，
            # 今天的任何重试 / 多实例 / catch-up 都会跳过（宁少一次，不重复发）。
            await mark_scheduled_task(task.task_id, group_id, today, status)
        return status


# 向后兼容别名（旧测试 / 脚本可能直接引用）
execute_morning_greeting = execute_scheduled_task


def _run_task_job(task: ScheduledTask) -> None:
    """APScheduler cron 入口：对每个目标群执行（群之间状态相互独立）。"""
    if not task.enabled:
        return
    now = get_now()
    logger.info(
        "[SCHEDULED] task={} cron 触发（本地时间 {}，目标群数 {}）",
        task.task_id,
        now.strftime("%H:%M:%S"),
        len(task.target_group_ids),
    )
    for group_id in sorted(task.target_group_ids):
        asyncio.create_task(execute_scheduled_task(group_id, task, now))


def _make_task_job(task: ScheduledTask):
    """为注册表里的一个任务生成可交给 scheduler 的 job 函数（async）。

    群之间并行触发（各群内部仍由共享锁串行），单个群异常不影响其它群。
    """

    async def job() -> None:
        _run_task_job(task)

    return job


# ==========================================================================
# catch-up（v0.4.x：启动 + Bot 连接两个时机）
#
# Reverse WebSocket：NoneBot 先启动、NapCat 后连接。on_startup 时通常还没有
# Bot，因此 catch-up 挂在两个钩子上：
#   - on_startup：Bot 已连接（或随后立刻连接）时立即补；
#   - on_bot_connect：NapCat 晚几秒 / 晚几分钟连接时再补（真正的兜底）。
# 两者都只做“窗口内且今天没有终态记录”的检查，真正执行交给
# execute_scheduled_task 的原子 claim / reclaim，绝不重复发送。
# ==========================================================================


async def _task_catchup(task: ScheduledTask) -> None:
    """单个任务的 catch-up 检查：窗口内 + 今天可重试 → 逐个目标群尝试执行。"""
    if not task.enabled:
        return
    now = get_now()
    if not within_catchup_window(now, task):
        return
    today = now.strftime("%Y-%m-%d")
    for group_id in sorted(task.target_group_ids):
        status = await get_scheduled_task_status(task.task_id, group_id, today)
        if status in TERMINAL_STATUSES or status == "running":
            # success / skipped_* → 终态；running → 可能已发出。都不重试。
            continue
        logger.info(
            "[SCHEDULED] catch-up：尝试补执行 task={} group_id={}（今日状态={}）",
            task.task_id,
            group_id,
            status or "none",
        )
        # 不阻塞启动/连接：后台任务执行；execute 内部自带 Bot 检查 / claim / 锁 / 全兜底。
        asyncio.create_task(execute_scheduled_task(group_id, task, now))


async def _run_all_catchups() -> None:
    """遍历注册表里所有启用任务执行 catch-up 检查（startup / bot_connect 共用）。"""
    for task in SCHEDULED_TASKS.values():
        if task.enabled:
            await _task_catchup(task)


async def _on_bot_connect(bot) -> None:
    """NapCat 建立 OneBot 连接后：窗口内的未完成任务获得补执行机会。"""
    logger.info("[SCHEDULED] OneBot 连接建立，检查定时任务 catch-up 窗口")
    await _run_all_catchups()


# ==========================================================================
# 注册（bot.py 调用；nonebot-plugin-apscheduler 已加载）
# ==========================================================================


def setup_scheduled_tasks(driver) -> None:
    """遍历任务注册表：注册 cron + 启动 catch-up 钩子。幂等（同 id 替换）。"""
    if not SCHEDULED_TASKS_ENABLED:
        logger.info("[SCHEDULED] SCHEDULED_TASKS_ENABLED=false，定时任务全部禁用")
        return

    from apscheduler.triggers.cron import CronTrigger
    from nonebot_plugin_apscheduler import scheduler

    registered = 0
    for task in SCHEDULED_TASKS.values():
        if not task.enabled:
            logger.info("[SCHEDULED] task={} 未启用（{}_ENABLED=false），跳过注册", task.task_id, task.task_id.upper())
            continue
        if not task.target_group_ids:
            logger.warning(
                "[SCHEDULED] task={} 没有可用目标群，不注册 cron", task.task_id
            )
            continue

        trigger = CronTrigger(hour=task.hour, minute=task.minute, timezone=TIMEZONE)
        scheduler.add_job(
            _make_task_job(task),
            trigger=trigger,
            id=f"task:{task.task_id}",
            name=f"scheduled:{task.task_id}",
            coalesce=True,       # 事件循环短暂阻塞造成的多次 misfire 合并为一次
            max_instances=1,     # 同一任务绝不同时跑两份
            misfire_grace_time=max(60, task.catchup_minutes * 60),
            replace_existing=True,
        )
        registered += 1
        logger.info(
            "[SCHEDULED] 已注册定时任务 {}：cron={:02d}:{:02d} timezone={} 目标群数={} "
            "catchup={}min skip_if_active={}min",
            task.task_id,
            task.hour,
            task.minute,
            TIMEZONE,
            len(task.target_group_ids),
            task.catchup_minutes,
            task.skip_if_active_minutes,
        )
    if registered == 0:
        logger.warning("[SCHEDULED] 没有任何启用的定时任务")
        return
    # catch-up 挂两个时机：启动（Bot 可能已连上）与 Bot 连接（NapCat 后连上的兜底）。
    driver.on_startup(_run_all_catchups)
    driver.on_bot_connect(_on_bot_connect)
