r"""Debug 管理插件（v0.2.5）：以反斜杠 \debug 开头的管理员调试命令。

安全边界：
- 只处理群消息（GroupMessageEvent），暂不处理私聊；
- 群访问白名单（fail-closed）：未授权群的 \debug 消息同样直接丢弃，
  不回复、不读写任何数据库（services/group_access.py 统一判断）；
- 管理员白名单来自 .env 的 DEBUG_ADMIN_QQ（逗号分隔的 QQ 号）；
  DEBUG_ADMIN_QQ 为空时全部 debug 命令禁用；
- 不在白名单的 QQ 回复「无权限使用调试命令。」；
- 绝不输出 API Key / Access Token / 完整环境变量；
- 不提供 shell / eval / 任意 SQL 执行，不是远程命令后门。

匹配器：
- rule=startswith("\debug")：纯文本以 \debug 开头即触发，不需要 @机器人；
- priority=1（高于 ai_chat 的 10）、block=True：\debug 消息不会再被当成
  普通 AI 问题，也不会进入 context_recorder 的群聊记录（避免把管理员设置的
  个人资料泄漏进群聊上下文）。
"""

import os

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.rule import Rule

from services.affection_store import LEVEL_LABELS
from services.affection_store import RELATIONSHIP_LABELS
from services.affection_store import affection_level
from services.affection_store import collect_participant_ids
from services.affection_store import get_affection
from services.affection_store import get_relationship_context
from services.affection_store import list_group_affections
from services.affection_store import set_affection
from services.context_store import CONTEXT_MESSAGE_LIMIT
from services.context_store import get_recent_messages
from services.group_access import is_group_allowed
from services.memory_retriever import MEMORY_TOP_K
from services.relationship_service import get_effective_relationship
from services.memory_retriever import retrieve_memories
from services.personal_memory_store import clear_user_memories
from services.personal_memory_store import count_memories
from services.personal_memory_store import delete_memory
from services.personal_memory_store import get_user_memories
from services.personal_memory_store import ping_memory_db
from services.personal_memory_store import set_memory
from services.prompt_builder import sender_display_name
from services.user_store import get_user


def _parse_admin_qq() -> set[int]:
    """解析 .env 的 DEBUG_ADMIN_QQ（逗号分隔）；空 = 全部禁用。

    非法项只记 WARNING 并跳过，不让单个配置错误拖垮启动。
    """
    raw = (os.getenv("DEBUG_ADMIN_QQ") or "").strip()
    if not raw:
        return set()
    admins: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            admins.add(int(part))
        except ValueError:
            logger.warning("[DEBUG] DEBUG_ADMIN_QQ 中存在非法项（已忽略）")
    return admins


# 进程启动时解析一次（改 .env 需重启生效）
DEBUG_ADMINS: set[int] = _parse_admin_qq()

if DEBUG_ADMINS:
    logger.info("[DEBUG] 已配置 {} 个调试管理员", len(DEBUG_ADMINS))
else:
    logger.info("[DEBUG] DEBUG_ADMIN_QQ 为空，调试命令全部禁用")

# 不带 \\debug 前缀、但 @机器人 直接跟 debug 子命令时也识别为调试命令
# （仅管理员生效；非管理员 @ 这些词仍是普通 AI 聊天，不会误报权限）。
BARE_DEBUG_KEYWORDS = ("affection", "memory", "rag", "relation", "whoami", "status", "help")


async def _at_bare_debug_rule(event: GroupMessageEvent) -> bool:
    """@机器人 + debug 子命令（无需 \\debug 前缀）→ 仅管理员识别为调试命令。"""
    text = event.get_plaintext().strip()
    if not text or not event.to_me:
        return False
    first_word = text.split(maxsplit=1)[0]
    if first_word not in BARE_DEBUG_KEYWORDS:
        return False
    return event.user_id in DEBUG_ADMINS


async def _debug_rule(event: GroupMessageEvent) -> bool:
    """调试命令匹配规则（两种形式）：

    - 纯文本以 \\debug 开头（有无 @机器人 均可，get_plaintext 会自动去掉 @）；
    - 或：@机器人 且第一个词是 debug 子命令（管理员专用便捷形式）。

    群访问白名单检查放在读取正文之前：未授权群的命令文本根本不会被读取，
    规则直接返回 False（处理器不运行，自然也不会回复 / 读写数据库）。
    """
    if not is_group_allowed(event.group_id):
        return False
    if event.get_plaintext().strip().startswith("\\debug"):
        return True
    return await _at_bare_debug_rule(event)


# 调试命令匹配器：priority=1 高于 ai_chat(10)，block=True 阻断后续处理器
# （不会进 AI 或群聊记录）。
debug = on_message(rule=Rule(_debug_rule), priority=1, block=True)


@debug.handle()
async def handle(event: GroupMessageEvent):
    # 0. 群访问白名单（fail-closed）：未授权群直接丢弃，
    #    不读取命令内容、不回复、不读写任何数据库。
    #    匹配规则层已先做了一次检查（未授权群连命令正文都不会被读取），
    #    这里保留防御性二次检查。
    if not is_group_allowed(event.group_id):
        return

    text = event.get_plaintext().strip()
    nickname = sender_display_name(event)
    reply = await _execute_debug_command(text, event.user_id, event.group_id, nickname)
    # 隐私日志：只记录解析后的命令结构（cmd / subcmd / target_user_id / key），
    # 绝不记录完整命令文本（\\debug memory set 的 value 可能含私人资料）。
    logger.info(
        "[DEBUG] user_id={} group_id={} {}",
        event.user_id,
        event.group_id,
        _debug_log_fields(text),
    )
    await debug.finish(reply)


# 日志中允许记录的 debug 子命令（这些子命令没有隐私敏感的 value）
_DEBUG_LOG_SUBCOMMANDS = ("set", "list", "del", "clear", "get")


def _debug_log_fields(text: str) -> str:
    """把一条 debug 命令解析成可安全写日志的字段串（绝不包含 value）。

    \\debug memory set <qq> <key> <value...> 只输出 target_user_id 与 key；
    value、rag query、未知命令的参数一律不进入日志。字段值先做空白折叠与截断，
    防止 token 本身携带超长内容。
    """
    parts = (text or "").split()
    if parts and parts[0] == "\\debug":
        parts = parts[1:]
    if not parts:
        return "cmd="

    fields: list[str] = [f"cmd={_log_token(parts[0])}"]
    cmd = parts[0]
    if len(parts) >= 2 and cmd in ("memory", "affection") and parts[1] in _DEBUG_LOG_SUBCOMMANDS:
        fields.append(f"subcmd={_log_token(parts[1])}")
        # memory set/list/del/clear 与 affection set/get 的第二参数都是目标 QQ 号
        if len(parts) >= 3:
            target = _parse_qq(parts[2])
            if target is not None:
                fields.append(f"target_user_id={target}")
        # memory set <qq> <key> <value...>：key 可记（键名），value 绝不记录
        if cmd == "memory" and parts[1] == "set" and len(parts) >= 4:
            fields.append(f"key={_log_token(parts[3])}")
    return " ".join(fields)


def _log_token(token: str, max_chars: int = 32) -> str:
    """日志用 token：折叠空白并截断，避免异常超长参数进入日志。"""
    cleaned = " ".join((token or "").split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"
    return cleaned


def _help_text() -> str:
    return "\n".join(
        [
            "Debug 命令列表：",
            "\\debug help                      查看本帮助",
            "\\debug whoami                    显示你的 user_id / group_id / nickname",
            "\\debug status                    显示 Bot 运行状态（不含密钥）",
            "\\debug memory set <qq> <key> <value...>  设置某人资料（value 可含空格，重复设置覆盖）",
            "\\debug memory list <qq>          查看某人全部资料",
            "\\debug memory del <qq> <key>     删除某人的一条资料",
            "\\debug memory clear <qq>         清空某人全部资料",
            "\\debug rag <query...>            只运行检索器，观察检索结果（不调用 LLM）",
            "\\debug affection set <qq> <0-100>  设置某人的亲近倾向（好感度）",
            "\\debug affection get <qq>        查看某人的好感度与等级",
            "\\debug affection list            列出本群全部已设置的好感度",
            "\\debug relation                  列出本群全部 relationship（好感度）",
            "\\debug relation context          显示将交给 LLM 的 Relationship Context（不调用 LLM）",
        ]
    )


async def _execute_debug_command(
    text: str,
    user_id: int,
    group_id: int,
    nickname: str,
) -> str:
    """执行一条 debug 命令并返回回复文本（权限校验在这里统一处理）。"""
    if not DEBUG_ADMINS:
        return "调试命令未启用。"
    if user_id not in DEBUG_ADMINS:
        return "无权限使用调试命令。"

    # 兼容两种形式："\debug affection get ..." 与 "@机器人 affection get ..."
    parts = text.split()
    if parts and parts[0] == "\\debug":
        parts = parts[1:]
    cmd = parts[0] if parts else ""

    if cmd == "help" or cmd == "":
        return _help_text()
    if cmd == "whoami":
        return f"user_id={user_id}\ngroup_id={group_id}\nnickname={nickname}"
    if cmd == "status":
        return await _status()
    if cmd == "memory":
        return await _memory_command(parts[1:], group_id)
    if cmd == "rag":
        return await _rag_command(" ".join(parts[1:]), user_id, group_id)
    if cmd == "affection":
        return await _affection_command(parts[1:], group_id)
    if cmd == "relation":
        return await _relation_command(parts[1:], user_id, group_id)
    return f"未知的调试命令：{cmd}\n输入 \\debug help 查看帮助。"


async def _status() -> str:
    """Bot 状态（绝不输出 API Key / Token / 环境变量）。"""
    provider = os.getenv("AI_PROVIDER", "deepseek").strip().lower() or "deepseek"
    fallback = (os.getenv("AI_FALLBACK") or "").strip().lower()
    memory_ok = await ping_memory_db()
    memory_count = await count_memories()
    return "\n".join(
        [
            "Bot Debug Status",
            f"provider={provider}",
            f"fallback={fallback or '-'}",
            f"memory_db={'OK' if memory_ok else 'FAIL'}",
            f"memory_count={memory_count}",
            f"memory_top_k={MEMORY_TOP_K}",
        ]
    )


def _parse_qq(raw: str) -> int | None:
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


async def _memory_command(args: list[str], group_id: int) -> str:
    """\\debug memory 子命令：set / list / del / clear。"""
    if not args:
        return "用法：\\debug memory <set|list|del|clear> ...\n输入 \\debug help 查看帮助。"
    sub = args[0]

    if sub == "set":
        # \debug memory set <qq> <key> <value...>
        if len(args) < 4:
            return "用法：\\debug memory set <qq> <key> <value...>"
        target_id = _parse_qq(args[1])
        if target_id is None:
            return "user_id 必须是正整数 QQ 号。"
        key = args[2].strip()
        value = " ".join(args[3:]).strip()
        if not key or not value:
            return "key 与 value 不能为空。"
        # 尽量带上目标用户最近用过的显示名（查不到也不影响设置）
        nickname = None
        user = await get_user(target_id)
        if user is not None and user.latest_nickname:
            nickname = user.latest_nickname
        ok = await set_memory(group_id, target_id, key, value, nickname=nickname)
        if not ok:
            return "设置失败（记忆库不可用或参数非法）。"
        return f"已设置 {target_id} 的 {key} = {value}"

    if sub == "list":
        if len(args) < 2:
            return "用法：\\debug memory list <qq>"
        target_id = _parse_qq(args[1])
        if target_id is None:
            return "user_id 必须是正整数 QQ 号。"
        rows = await get_user_memories(group_id, target_id)
        if not rows:
            return f"{target_id}：（无记录）"
        lines = [f"{target_id}:"]
        for row in rows:
            lines.append(f"{row.memory_key} = {row.memory_value}")
        return "\n".join(lines)

    if sub == "del":
        if len(args) < 3:
            return "用法：\\debug memory del <qq> <key>"
        target_id = _parse_qq(args[1])
        if target_id is None:
            return "user_id 必须是正整数 QQ 号。"
        key = args[2].strip()
        deleted = await delete_memory(group_id, target_id, key)
        return "已删除。" if deleted else "未找到该记录（或删除失败）。"

    if sub == "clear":
        if len(args) < 2:
            return "用法：\\debug memory clear <qq>"
        target_id = _parse_qq(args[1])
        if target_id is None:
            return "user_id 必须是正整数 QQ 号。"
        deleted = await clear_user_memories(group_id, target_id)
        return f"已清空 {deleted} 条记录。"

    return f"未知的 memory 子命令：{sub}\n输入 \\debug help 查看帮助。"


async def _rag_command(query: str, user_id: int, group_id: int) -> str:
    """\\debug rag：只运行检索器并展示结果，不调用 LLM。"""
    if not query:
        return "用法：\\debug rag <query...>"
    items = await retrieve_memories(group_id, user_id, query, MEMORY_TOP_K)
    lines = ["RAG Debug", f"query={query}", "", "retrieved:"]
    if not items:
        lines.append("（无匹配）")
    for index, item in enumerate(items, 1):
        value = item.value
        if len(value) > 80:
            value = value[:80] + "…"
        lines.append(
            f"{index}. user={item.user_id} key={item.key} value={value} "
            f"score={item.score} reason={'+'.join(item.reasons)}"
        )
    return "\n".join(lines)


def _parse_affection_score(raw: str) -> int | None:
    """解析好感度数值：必须为 0~100 的整数。"""
    try:
        score = int(raw)
    except ValueError:
        return None
    if not (0 <= score <= 100):
        return None
    return score


async def _affection_command(args: list[str], group_id: int) -> str:
    """\\debug affection 子命令：set / get / list。"""
    if not args:
        return "用法：\\debug affection <set|get|list> ...\n输入 \\debug help 查看帮助。"
    sub = args[0]

    if sub == "set":
        if len(args) < 3:
            return "用法：\\debug affection set <qq> <0-100>"
        target_id = _parse_qq(args[1])
        if target_id is None:
            return "user_id 必须是正整数 QQ 号。"
        score = _parse_affection_score(args[2])
        if score is None:
            return "score 必须是 0~100 的整数。"
        ok = await set_affection(group_id, target_id, score)
        if not ok:
            return "设置失败（数据库不可用）。"
        level = affection_level(score)
        return f"已设置 {target_id} affection={score} (level={level}, {LEVEL_LABELS[level]})"

    if sub == "get":
        if len(args) < 2:
            return "用法：\\debug affection get <qq>"
        target_id = _parse_qq(args[1])
        if target_id is None:
            return "user_id 必须是正整数 QQ 号。"
        score = await get_affection(group_id, target_id)
        level = affection_level(score)
        relationship = await get_effective_relationship(target_id)
        relationship_label = RELATIONSHIP_LABELS.get(relationship, relationship)
        return (
            f"{target_id}\n"
            f"affection={score} level={level} ({LEVEL_LABELS[level]})\n"
            f"relationship={relationship} ({relationship_label})"
        )

    if sub == "list":
        rows = await list_group_affections(group_id)
        if not rows:
            return "（无记录：本群尚未设置任何好感度）"
        lines = [f"affection list (group={group_id}):"]
        for user_id, score in rows:
            level = affection_level(score)
            lines.append(f"{user_id} = {score} ({level})")
        return "\n".join(lines)

    return f"未知的 affection 子命令：{sub}\n输入 \\debug help 查看帮助。"


async def _relation_command(args: list[str], user_id: int, group_id: int) -> str:
    """\\debug relation / \\debug relation context。

    - 不带参数：列出本群全部已设置的好感度；
    - context：展示程序准备交给 LLM 的 Relationship Context（不调用 LLM）。
    """
    if args and args[0] == "context":
        history = await get_recent_messages(group_id, CONTEXT_MESSAGE_LIMIT)
        participants = collect_participant_ids(history, user_id)
        scores = [(uid, await get_affection(group_id, uid)) for uid in participants]
        scores.sort(key=lambda item: (-item[1], item[0]))
        lines = ["Relationship Debug", "", "participants:"]
        for uid, score in scores:
            level = affection_level(score)
            relationship = await get_effective_relationship(uid)
            lines.append(f"{uid} affection={score} level={level} relationship={relationship}")
        if len(scores) >= 2:
            lines.append("")
            lines.append("priority tendency:")
            lines.append(" > ".join(str(uid) for uid, _ in scores))
        context = await get_relationship_context(group_id, participants, user_id)
        lines.append("")
        lines.append("---- 交给 LLM 的块 ----")
        lines.append(context if context else "（空：数据库不可用或没有参与者）")
        return "\n".join(lines)

    rows = await list_group_affections(group_id)
    if not rows:
        return "（无记录：本群尚未设置任何好感度）"
    lines = [f"relationship list (group={group_id}):"]
    for uid, score in rows:
        relationship = await get_effective_relationship(uid)
        lines.append(f"{uid} = {score} (relationship={relationship})")
    return "\n".join(lines)


# 供测试直接调用（避免在 matcher 上下文外触发 finish）
__all__ = [
    "debug",
    "handle",
    "DEBUG_ADMINS",
    "_execute_debug_command",
    "_debug_log_fields",
    "_memory_command",
    "_rag_command",
    "_affection_command",
    "_relation_command",
    "_status",
]
