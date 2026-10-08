r"""/paint —— 让夜子画一张图（管理员命令，v0.10）。

用法：
    /paint 一只戴着毛线帽的橘猫，坐在窗台上晒太阳

流程：群白名单 → 管理员校验 → 冷却 / 每日上限 → 提示语 → 调用生图 API →
      落盘留档（png + 同名 txt 记录谁在什么时候画的什么）→ 发图 + 夜子的一句话。

安全边界（与 \debug 一致）：
- 只处理群消息，不处理私聊；
- 群访问白名单 fail-closed：未授权群直接丢弃，不读取命令正文；
- 只有 PAINT_ADMIN_QQ（默认沿用 DEBUG_ADMIN_QQ）里的 QQ 能用；
- 冷却（默认 60 秒/人）与每日上限（默认 20 张/天，全局）双保险，避免刷爆额度；
- 日志只记「谁 / 多少字 / 结果」，prompt 正文截断后记录，且先过 redact_secrets；
- 绝不输出 API Key；任何一步失败都只回一句降级文案，绝不抛异常。

匹配方式与 \debug 相同：纯文本以 /paint 开头即可（带不带 @机器人 都行，
get_plaintext() 会自动去掉 @），priority=1 + block=True，因此不会被当成
普通聊天，也不会进入群聊上下文记录。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.adapters.onebot.v11 import Message
from nonebot.adapters.onebot.v11 import MessageSegment
from nonebot.rule import Rule

from services import redact_secrets
from services.context_store import add_message
from services.deepseek import ask_deepseek
from services.group_access import is_group_allowed
from services.image_gen import CONFIG as PAINT_CONFIG
from services.image_gen import edit_image
from services.image_gen import fetch_image_bytes
from services.image_gen import generate_image
from services.image_gen import guess_mime
from services.prompt_builder import BOT_NAME
from services.prompt_builder import STATIC_SYSTEM_PROMPT
from services.prompt_builder import build_edit_prompt
from services.prompt_builder import build_paint_edit_messages
from services.prompt_builder import build_paint_prompt_messages
from services.prompt_builder import clean_image_prompt
from services.prompt_builder import paint_context_text
from services.prompt_builder import sender_display_name

# ---------------------------------------------------------------- 配置

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 触发前缀：半角 / 和全角 ／ 都支持（中文输入法很容易打出全角）
PAINT_PREFIXES = ("/paint", "／paint")

# 描述长度上限（防止把整篇文章塞进生图接口）
MAX_PROMPT_CHARS = 500

# 每人冷却秒数 / 每天总张数上限
DEFAULT_COOLDOWN_SECONDS = 60
DEFAULT_DAILY_LIMIT = 20


def _parse_qq_set(raw: str) -> set[int]:
    """解析逗号分隔的 QQ 号；非法项只记 WARNING 并跳过（不影响启动）。"""
    admins: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            admins.add(int(part))
        except ValueError:
            logger.warning("[PAINT] 管理员白名单里有非法项（已忽略）")
    return admins


def _load_admin_qq(env: dict | None = None) -> set[int]:
    r"""PAINT_ADMIN_QQ 优先；没配就沿用 \debug 的 DEBUG_ADMIN_QQ。"""
    src = os.environ if env is None else env
    raw = (src.get("PAINT_ADMIN_QQ") or "").strip()
    if not raw:
        raw = (src.get("DEBUG_ADMIN_QQ") or "").strip()
    return _parse_qq_set(raw)


def _num(env: dict | None, key: str, fallback: int) -> int:
    src = os.environ if env is None else env
    try:
        value = int(str(src.get(key) or fallback))
    except (TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


PAINT_ADMINS: set[int] = _load_admin_qq()
PAINT_COOLDOWN_SECONDS: int = _num(None, "PAINT_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS)
PAINT_DAILY_LIMIT: int = _num(None, "PAINT_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)
PAINT_DIR: Path = Path(os.getenv("PAINT_DIR") or (PROJECT_ROOT / "data" / "paint"))
# 先把用户要求交给夜子改写成具体画面，再喂给绘图模型（默认开）
PAINT_REWRITE_PROMPT: bool = (os.getenv("PAINT_REWRITE_PROMPT") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
PAINT_REWRITE_MAX_CHARS: int = _num(None, "PAINT_REWRITE_MAX_CHARS", 120)

if PAINT_ADMINS:
    logger.info(
        "[PAINT] 已配置 {} 个可用管理员，冷却 {}s，每日上限 {} 张",
        len(PAINT_ADMINS),
        PAINT_COOLDOWN_SECONDS,
        PAINT_DAILY_LIMIT,
    )
else:
    logger.info("[PAINT] 未配置管理员白名单（PAINT_ADMIN_QQ / DEBUG_ADMIN_QQ 都为空），命令不可用")


# ---------------------------------------------------------------- 纯函数（便于测试）


def extract_prompt(text: str) -> str | None:
    """从消息文本里取出绘图描述。

    - 不是 /paint 命令 → None
    - 是命令但没写描述 → ""（调用方据此回复用法）
    - 前缀后面必须是空白或结束，避免 /painter 被误判成 /paint

    空白判断用 `str.isspace()`：很多人习惯先打 `/paint` 再按回车写描述，
    所以换行、Tab、全角空格都必须算数（只写 " " 和 "\\t" 会漏掉换行）。
    """
    stripped = (text or "").strip()
    lowered = stripped.lower()
    for prefix in PAINT_PREFIXES:
        if not lowered.startswith(prefix):
            continue
        rest = stripped[len(prefix) :]
        if rest[:1] and not rest[:1].isspace():
            continue
        return rest.strip()[:MAX_PROMPT_CHARS]
    return None


@dataclass
class QuotaState:
    """内存态用量统计（进程内；重启即清零）。"""

    day: str = ""
    count: int = 0
    last_at: dict | None = None


def today_key(now: float | None = None) -> str:
    """本地日期字符串，用作每日上限的统计键。"""
    return datetime.fromtimestamp(now if now is not None else time.time()).strftime("%Y-%m-%d")


def check_quota(
    state: QuotaState,
    user_id: int,
    now: float,
    cooldown: int = PAINT_COOLDOWN_SECONDS,
    daily_limit: int = PAINT_DAILY_LIMIT,
) -> tuple[bool, str, int]:
    """能不能画？返回 (是否允许, 原因代码, 剩余秒数/额度提示数字)。

    纯函数：只读状态、不做任何副作用，方便测试。
    原因代码：ok / cooldown / daily_limit
    """
    day = today_key(now)
    # 跨天自动重置计数（读的时候按「今天」算，不依赖定时任务）
    used_today = state.count if state.day == day else 0
    if daily_limit > 0 and used_today >= daily_limit:
        return False, "daily_limit", daily_limit

    last = (state.last_at or {}).get(user_id)
    if last is not None and cooldown > 0:
        remain = int(cooldown - (now - float(last)))
        if remain > 0:
            return False, "cooldown", remain
    return True, "ok", 0


def record_use(state: QuotaState, user_id: int, now: float) -> None:
    """记一次成功用量（跨天时先把计数清零）。"""
    day = today_key(now)
    if state.day != day:
        state.day = day
        state.count = 0
    state.count += 1
    if state.last_at is None:
        state.last_at = {}
    state.last_at[user_id] = now


def collect_image_url(message) -> str:
    """从一条消息里取出第一张 http(s) 图片的地址。

    只认 `url` 字段（NapCat 给的 CDN 直链）—— `file` 在部分实现里只是本机
    文件名或 QQ file id，拿它当 URL 会得到一个访问不了的地址
    （这一点是 v0.7 感知层踩过坑之后定下的规则）。
    """
    for segment in message or []:
        if getattr(segment, "type", "") != "image":
            continue
        url = str((getattr(segment, "data", None) or {}).get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            return url
    return ""


def has_image_segment(message) -> bool:
    """这条消息里有没有图片段（哪怕拿不到地址）。"""
    return any(getattr(segment, "type", "") == "image" for segment in (message or []))


def reply_message(event):
    """被引用消息的段落（适配器已经取好了，不会额外发请求）。"""
    reply = getattr(event, "reply", None)
    return getattr(reply, "message", None) if reply is not None else None


def choose_reference(event) -> tuple[str, bool]:
    """决定用哪张图做参考。

    返回 (图片地址, 是否"本来该有图但拿不到")：
    - 本条消息里的图优先，其次是被引用消息里的图；
    - 有图片段但取不到 http 地址时，第二个返回 True —— 调用方据此给出
      明确提示，而不是默默退化成文生图（那样画出来的人跟用户指的完全无关）。
    """
    current = getattr(event, "message", None)
    url = collect_image_url(current)
    if url:
        return url, False

    replied = reply_message(event)
    url = collect_image_url(replied)
    if url:
        return url, False

    missing = has_image_segment(current) or has_image_segment(replied)
    return "", missing


def build_image_filename(user_id: int, now: float, mime: str) -> str:
    """存档文件名：时间 + QQ，便于回看是谁什么时候画的。"""
    suffix = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".png")
    return f"{datetime.fromtimestamp(now).strftime('%Y%m%d-%H%M%S')}-{user_id}{suffix}"


def save_archive(image, meta: str, directory: Path | None = None, now: float | None = None,
                 user_id: int = 0) -> Path | None:
    """把图片和一行说明写到 data/paint/。失败只记日志、返回 None（不影响发图）。"""
    target_dir = directory or PAINT_DIR
    stamp = now if now is not None else time.time()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / build_image_filename(user_id, stamp, image.mime)
        path.write_bytes(image.data)
        # 同名 txt 记录元信息，方便事后回看（不含 API Key、不含 base64）
        path.with_suffix(".txt").write_text(meta, encoding="utf-8")
        return path
    except Exception as exc:
        logger.error("[PAINT] 存档失败：{}: {}", type(exc).__name__, redact_secrets(str(exc)))
        return None


async def rewrite_prompt(user_prompt: str, editing: bool = False) -> str:
    """让模型把要求整理成绘图模型能懂的话；失败就退回用户原话。

    两种模式用的是**两套完全不同的规则**：

    - `editing=False`（文生图）：夜子按自己的审美写一整段画面描述。
      这一步是 v0.10.1 加的 —— 直接把"画一个你最喜欢的东西"交给绘图模型，
      它不知道"你"是谁，只会画一张没个性的通用图。
    - `editing=True`（改图）：只整理"要改什么"，**禁止描述原图**，而且不带人格。
      v0.10.2 踩的坑：用户引用自己的照片说"画一张上面这个人抱着手机的图片"，
      走的是文生图那套规则，夜子看不见图、又带着自己的人格，于是写成
      "一个黑发少女坐在窗边读书" —— 把照片里的人整个换成了她自己。
    """
    if not PAINT_REWRITE_PROMPT:
        return user_prompt
    messages = (
        build_paint_edit_messages(user_prompt, PAINT_REWRITE_MAX_CHARS)
        if editing
        else build_paint_prompt_messages(user_prompt, PAINT_REWRITE_MAX_CHARS)
    )
    try:
        reply = await ask_deepseek(messages)
    except Exception as exc:  # ask_deepseek 内部已兜底，这里再保一层
        logger.warning("[PAINT] 提示词改写失败：{}", type(exc).__name__)
        return user_prompt

    cleaned = clean_image_prompt(reply or "", PAINT_REWRITE_MAX_CHARS)
    if not cleaned:
        logger.warning("[PAINT] 改写结果为空，退回用户原话")
        return user_prompt
    logger.info(
        "[PAINT] 提示词改写（{}）：{} 字 -> {} 字",
        "改图" if editing else "文生图",
        len(user_prompt),
        len(cleaned),
    )
    return cleaned


async def persona_line(prompt: str) -> str:
    """让夜子说一句「画好了」。失败就用兜底文案（绝不让这句话拖垮整条命令）。"""
    instruction = (
        "（系统提示：你刚刚画好了一张图，画面内容是「{}」。"
        "请用一句话告诉对方画好了，20 字以内，不要加引号、不要用 emoji、"
        "不要描述作画过程。）"
    ).format(prompt[:80])
    try:
        reply = await ask_deepseek(
            [
                {"role": "system", "content": STATIC_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
        )
    except Exception as exc:  # ask_deepseek 内部已兜底，这里再保一层
        logger.warning("[PAINT] 生成配文失败：{}", type(exc).__name__)
        return FALLBACK_LINE
    line = (reply or "").strip().replace("\n", " ")[:60]
    return line or FALLBACK_LINE


FALLBACK_LINE = "画好啦，看看喜不喜欢～"
USAGE_TEXT = "用法：/paint 描述，比如 /paint 一只戴着毛线帽的橘猫"


# ---------------------------------------------------------------- 匹配器

_state = QuotaState(last_at={})


async def _paint_rule(event: GroupMessageEvent) -> bool:
    """命中 /paint 命令。群白名单在读取正文之前检查（未授权群连正文都不读）。"""
    if not is_group_allowed(event.group_id):
        return False
    return extract_prompt(event.get_plaintext()) is not None


paint = on_message(rule=Rule(_paint_rule), priority=1, block=True)


@paint.handle()
async def handle_paint(event: GroupMessageEvent):
    # 防御性二次检查（规则层已经查过一次）
    if not is_group_allowed(event.group_id):
        return

    prompt = extract_prompt(event.get_plaintext()) or ""
    nickname = sender_display_name(event)

    # 1. 管理员校验（非管理员给一句明确回复，然后结束）
    if event.user_id not in PAINT_ADMINS:
        await paint.finish("这个命令只有管理员能用哦。")
        return

    # 2. 参数校验
    if not prompt:
        await paint.finish(USAGE_TEXT)
        return

    # 3. 功能是否可用
    if not PAINT_CONFIG.configured:
        await paint.finish("生图功能还没配置好（缺少 PAINT_API_KEY）。")
        return

    # 4. 冷却 / 每日上限（成功后才计数，失败不占用额度）
    now = time.time()
    allowed, reason, extra = check_quota(_state, event.user_id, now)
    if not allowed:
        if reason == "cooldown":
            await paint.finish(f"刚画过啦，{extra} 秒后再来～")
        else:
            await paint.finish(f"今天已经画满 {extra} 张了，明天再来吧。")
        return

    # 5. 先给个回执，避免生图期间群里没反应
    await paint.send("夜子正在画…")

    # 6. 先判定有没有参考图（只读消息段，不发网络请求）。
    #    必须在改写之前判定：有图 = 改图模式，改写规则完全不同（见 rewrite_prompt）。
    ref_url, image_missing = choose_reference(event)
    editing = bool(ref_url)
    if not editing and image_missing:
        await paint.finish("看到了图片，但取不到可用的地址，没法拿它当参考。")
        return

    # 7. 关键一步：让夜子把要求落成绘图模型能懂的话。
    #    - 文生图：「画一个你最喜欢的东西」→ 她按自己的审美写一段具体画面；
    #    - 改图：只整理"要改什么"，并且**不许描述原图**（描述会盖掉原图主体）。
    scene = await rewrite_prompt(prompt, editing=editing)
    # 改图时再套一层锚点：这段话由程序拼，模型改不掉。
    # 即使把改写关掉也照样加 —— 它是防止"主体被换掉"的最后一道保险。
    image_prompt = build_edit_prompt(scene, prompt) if editing else scene

    # 8. 取参考图字节（这一步才发网络请求）
    reference: bytes | None = None
    if ref_url:
        reference = await fetch_image_bytes(ref_url)
        if reference is None:
            await paint.finish("那张参考图我读不出来，换一张再试试？")
            return

    mode = "img2img" if reference is not None else "text2img"
    logger.info(
        "[PAINT] 收到请求：group={} user={} mode={} prompt_chars={}",
        event.group_id,
        event.user_id,
        mode,
        len(prompt),
    )

    if reference is not None:
        image = await edit_image(image_prompt, reference, guess_mime(reference))
    else:
        image = await generate_image(image_prompt)
    if image is None:
        await paint.finish("画失败了，等下再试试吧。")
        return

    # 8. 落盘留档（best-effort：存档失败也照常发图）
    #    同时记下用户原话与改写后的画面，方便复盘"她到底画了什么、为什么这么画"
    meta = (
        f"time: {datetime.fromtimestamp(now):%Y-%m-%d %H:%M:%S}\n"
        f"group: {event.group_id}\n"
        f"user: {event.user_id} ({nickname})\n"
        f"mode: {mode}\n"
        f"model: {PAINT_CONFIG.model}\n"
        f"size: {PAINT_CONFIG.size}\n"
        f"asked: {prompt}\n"
        f"image_prompt: {image_prompt}\n"
    )
    saved = save_archive(image, meta, now=now, user_id=event.user_id)
    # 参考图也留一份（图生图出问题时，能回看当时用的是哪张）
    if reference is not None and saved is not None:
        try:
            suffix = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(
                guess_mime(reference), ".png"
            )
            saved.with_name(saved.stem + "-ref" + suffix).write_bytes(reference)
        except Exception as exc:
            logger.warning("[PAINT] 参考图存档失败：{}", type(exc).__name__)

    # 9. 夜子配一句话（用 scene 而不是 image_prompt：后者带着程序拼的锚点，
    #    不是"画了什么"而是"被要求了什么"，念出来很怪）
    line = await persona_line(scene)
    record_use(_state, event.user_id, now)

    # 10. 把「我画了这张图」写进群聊上下文。
    #     不写的话，聊天模型（另一家模型、也不知道绘图能力）在群友追问
    #     "你这画的是书吗" 时会否认这张图是自己画的 —— 实测踩过。
    await add_message(
        group_id=event.group_id,
        user_id=event.self_id,
        nickname=BOT_NAME,
        role="assistant",
        content=paint_context_text(scene, edited=reference is not None),
    )

    logger.info(
        "[PAINT] 完成：user={} mode={} bytes={} saved={} line_chars={}",
        event.user_id,
        mode,
        len(image.data),
        bool(saved),
        len(line),
    )

    segments = [MessageSegment.image(image.data), MessageSegment.text("\n" + line)]
    await paint.finish(Message(segments))
