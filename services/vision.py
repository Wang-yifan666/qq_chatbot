"""QQ 图片 → 模型 image input（v0.5 DIRECT Vision MVP）。

职责边界（保持 v0.4 架构分工）：
- ai_chat 只负责编排，图片的提取 / 校验 / 限制 / 消息构造全部集中在本模块；
- 从 OneBot GroupMessageEvent 的 message 里读取真正的 image MessageSegment
  （segment.type == "image"，data["url"]），绝不从 get_plaintext() 或 CQ Code
  字符串里正则解析图片；
- 应用层限制：数量上限（VISION_MAX_IMAGES）、单图大小保护（NapCat 提供
  file_size 时生效）、detail 参数（VISION_DETAIL）；
- 纯函数 attach_images_to_last_user_message()：把图片以 OpenAI multimodal
  image_url block 附加到最后一个 role=user 消息（图片只允许出现在 user）；
- build_context_text()：写入 SQLite Context 的文字占位符（绝不写 URL / Base64）。

隐私红线：
- 日志绝不输出图片 URL / Base64 / 二进制 / CDN token，只记数量统计；
- 白名单检查在 ai_chat 最前面：未授权群根本不会调用本模块；
- 图片 URL 绝不落入 Context / 记忆 / 任何数据库。

配置（.env，均有默认值与范围校验，非法值安全回落默认）：
    VISION_ENABLED=true           DIRECT 视觉开关
    VISION_MAX_IMAGES=4           单次请求最多交给模型的图片数（1~10）
    VISION_DETAIL=auto            OpenAI detail 参数（auto | low | high）
    VISION_MAX_IMAGE_BYTES=10485760  单图大小上限（字节；NapCat 提供 file_size 时校验）
"""

import base64
import io
import os
from dataclasses import dataclass

import httpx
from nonebot import logger
from nonebot.adapters.onebot.v11 import GroupMessageEvent

# ===== 配置 =====

VISION_ENABLED_DEFAULT = True
VISION_MAX_IMAGES_DEFAULT = 4
VISION_MAX_IMAGES_MIN = 1
VISION_MAX_IMAGES_MAX = 10

VISION_DETAIL_DEFAULT = "auto"
VISION_DETAIL_VALUES = ("auto", "low", "high")

VISION_MAX_IMAGE_BYTES_DEFAULT = 10 * 1024 * 1024  # 10MB（DeepSeek 单图官方上限附近）
VISION_MAX_IMAGE_BYTES_MIN = 1024 * 1024          # 1MB
VISION_MAX_IMAGE_BYTES_MAX = 50 * 1024 * 1024     # 50MB

# 稳定的降级文案（ai_chat 直接使用；与模型/服务商无关）
VISION_DISABLED_REPLY = "我现在看不到图片。"
VISION_READ_FAILED_REPLY = "这张图片我暂时读不到。"
VISION_ALL_FAILED_REPLY = "我这会儿暂时看不了图片，稍后再试试。"

# DeepSeek 视觉 API 只接受这些格式（实测 BMP 会被 400 拒绝）。
# 其它 Pillow 能读的格式（bmp/tiff 等）会走 convert_unsupported_images 转成 JPEG。
SUPPORTED_IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp", "gif"})
# 下载图片用于格式转换的超时（秒）
_CONVERT_DOWNLOAD_TIMEOUT = 15.0


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("[VISION] {}={} 不是合法布尔值，使用默认 {}", name, raw, default)
    return default


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[VISION] {}={} 不是合法整数，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[VISION] {}={} 超出范围 [{}, {}]，使用默认 {}", name, value, low, high, default)
        return default
    return value


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in choices:
        return raw
    logger.warning("[VISION] {}={} 不是合法取值（{}），使用默认 {}", name, raw, "/".join(choices), default)
    return default


# 进程启动时解析一次（改 .env 需重启生效）
VISION_ENABLED = _env_bool("VISION_ENABLED", default=VISION_ENABLED_DEFAULT)
VISION_MAX_IMAGES = _env_int(
    "VISION_MAX_IMAGES",
    VISION_MAX_IMAGES_DEFAULT,
    VISION_MAX_IMAGES_MIN,
    VISION_MAX_IMAGES_MAX,
)
VISION_DETAIL = _env_choice("VISION_DETAIL", VISION_DETAIL_DEFAULT, VISION_DETAIL_VALUES)
VISION_MAX_IMAGE_BYTES = _env_int(
    "VISION_MAX_IMAGE_BYTES",
    VISION_MAX_IMAGE_BYTES_DEFAULT,
    VISION_MAX_IMAGE_BYTES_MIN,
    VISION_MAX_IMAGE_BYTES_MAX,
)


# ===== 数据模型 =====


@dataclass(frozen=True)
class VisionImage:
    """一张已通过校验、可以交给模型的图片。

    常规：只携带 external URL（绝不落盘）；
    转换：url 为空、data_url 为 JPEG base64 data URL（内存中转，同样绝不落盘）。
    """

    url: str
    detail: str
    file_name: str = ""
    data_url: str | None = None


@dataclass(frozen=True)
class ImageExtraction:
    """一次 @ 消息的图片提取结果（数量统计用于日志，绝不含 URL）。"""

    images: list[VisionImage]  # 被接受、将交给模型的图片（保持消息内顺序）
    total: int                 # 消息里出现的 image segment 总数
    accepted: int
    rejected: int


def _parse_file_size(raw) -> int | None:
    """解析 NapCat image segment 的 file_size（可能是 str 或 int）；不可解析 → None。"""
    if raw is None:
        return None
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def extract_images(event: GroupMessageEvent) -> ImageExtraction:
    """从 OneBot 事件中提取 image MessageSegment（白名单检查必须在此之前完成）。

    规则：
    - 只认 segment.type == "image"；优先 data["url"] 作为模型 external image URL；
    - 没有有效 URL → 拒绝（不计入模型输入）；
    - NapCat 提供 file_size 且超过 VISION_MAX_IMAGE_BYTES → 拒绝；
    - 超过 VISION_MAX_IMAGES 的多余图片按消息内顺序截断，只拒绝多余部分。
    """
    images: list[VisionImage] = []
    total = 0
    rejected = 0
    for segment in event.get_message():
        if getattr(segment, "type", None) != "image":
            continue
        total += 1
        data = getattr(segment, "data", None) or {}
        url = str(data.get("url") or "").strip()
        if not url:
            rejected += 1
            continue
        size = _parse_file_size(data.get("file_size"))
        if size is not None and size > VISION_MAX_IMAGE_BYTES:
            rejected += 1
            continue
        if len(images) >= VISION_MAX_IMAGES:
            rejected += 1
            continue
        images.append(
            VisionImage(
                url=url,
                detail=VISION_DETAIL,
                file_name=str(data.get("file") or ""),
            )
        )
    return ImageExtraction(
        images=images,
        total=total,
        accepted=len(images),
        rejected=rejected,
    )


# ===== 消息构造（纯函数） =====


def build_image_blocks(images: list[VisionImage]) -> list[dict]:
    """把 VisionImage 列表转成 OpenAI multimodal image_url block（只进 user content）。

    转换过的图片用 data_url（base64 JPEG），否则用 external URL。
    """
    blocks = []
    for image in images:
        payload = image.data_url if image.data_url else image.url
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": payload, "detail": image.detail},
            }
        )
    return blocks


# ===== 格式兼容（BMP 等 DeepSeek 不收的格式 → 内存中转成 JPEG） =====


def _image_ext(image: VisionImage) -> str:
    """从 file 名 / URL 路径取小写扩展名；取不到返回空串。"""
    for name in (image.file_name, image.url.split("?", 1)[0]):
        if "." in name:
            return name.rsplit(".", 1)[-1].lower()
    return ""


async def convert_unsupported_images(
    images: list[VisionImage],
) -> tuple[list[VisionImage], int]:
    """把 DeepSeek 不支持的图片格式（如 BMP）转成 JPEG data URL。

    - 只转换扩展名不在 SUPPORTED_IMAGE_EXTENSIONS 里的图片；
    - 下载 → Pillow 内存解码 → JPEG 压缩 → base64 data URL，全程不落盘；
    - 单张失败：保留原 URL 原样返回（模型侧仍可能接受），绝不因转换失败丢图；
    - 日志只记数量，绝不输出 URL / Base64 / 二进制。
    """
    converted = 0
    failed = 0
    output: list[VisionImage] = []
    for image in images:
        ext = _image_ext(image)
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            output.append(image)
            continue
        try:
            async with httpx.AsyncClient(timeout=_CONVERT_DOWNLOAD_TIMEOUT) as client:
                response = await client.get(image.url)
                response.raise_for_status()
                raw = response.content
            from PIL import Image as PILImage

            with PILImage.open(io.BytesIO(raw)) as img:
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                jpeg = buf.getvalue()
            data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
            output.append(
                VisionImage(
                    url=image.url,
                    detail=image.detail,
                    file_name=image.file_name,
                    data_url=data_url,
                )
            )
            converted += 1
        except Exception:
            # 转换失败：保留原图原样（外链直传仍可能成功），只计数。
            failed += 1
            output.append(image)
    if converted or failed:
        logger.info(
            "[VISION] format conversion done total={} converted={} kept_original={}",
            len(images),
            converted,
            failed,
        )
    return output, failed


def attach_images_to_last_user_message(
    messages: list[dict],
    images: list[VisionImage],
) -> list[dict]:
    """纯函数：拷贝 messages，把图片 block 附加到最后一个 role=user 消息上。

    - 原 messages 不被修改（主备共用同一结构时由调用方决定何时 attach）；
    - 最后一个 user 消息的字符串 content 变成 [{"type":"text",...}, image_url...]；
    - 图片只允许出现在 role=user；system / assistant / tool 绝不带图片；
    - images 为空时原样返回。
    """
    if not images:
        return list(messages)
    output = [dict(m) for m in messages]
    for message in reversed(output):
        if message.get("role") != "user":
            continue
        text = message.get("content") if isinstance(message.get("content"), str) else ""
        content: list[dict] = [{"type": "text", "text": text}]
        content.extend(build_image_blocks(images))
        message["content"] = content
        break
    return output


def messages_have_images(messages: list[dict]) -> bool:
    """判断 messages 是否包含 multimodal image content（任何 user 消息）。"""
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            return True
    return False


# ===== Context 占位符 =====


def build_context_text(question: str, image_count: int) -> str:
    """构造写入 SQLite Context 的用户消息文本（文字化占位，绝不写 URL / Base64）。

    - 无图片：原样返回 question；
    - 有文字 + 图片：question 后追加「[附带 N 张图片]」；
    - 纯图片：只有「[发送了 N 张图片]」。
    """
    if image_count <= 0:
        return question
    note = f"[附带 {image_count} 张图片]" if question else f"[发送了 {image_count} 张图片]"
    return f"{question}\n{note}" if question else note
