"""生图 API 客户端封装（纯 Transport 层，v0.10）。

调用方式参考换API 的《GPT-Image-2 绘图使用说明》：
https://www.huanapi.com/articles/9

要点：
- 走 **OpenAI Images API**（`POST /v1/images/generations`），
  官方 gpt-image-2 与任何 OpenAI 兼容网关用同一套代码，只改 base_url / model；
- 鉴权头是标准的 `Authorization: Bearer <令牌>`；
  换API 要求令牌分组为 **GTP**，否则会 401/403；
- 返回两种形态都兼容：`url`（文档说普通使用推荐）与 `b64_json`（文档说
  「想让程序自行保存图片」用这个）。本 Bot **一定会把图存到本地**，
  所以默认用 `b64_json`，少一次下载、也不受临时链接过期影响；
- 文档明确「不支持」的参数（stream / partial_images）与「不建议」的参数
  （style）一律不传；vendor 私有参数统一走 extra_body，避免被 SDK 的
  类型定义改写或丢掉；
- 任何失败都返回 None 并记日志，绝不抛异常 —— 一个画图功能不该把 Bot 弄崩；
- 日志一律先过 redact_secrets，绝不输出 API Key / base64 / 带签名的 URL。
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass

import httpx
from nonebot import logger
from openai import AsyncOpenAI

from services import redact_secrets

DEFAULT_BASE_URL = "https://www.huanapi.com/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "1024x1024"

# 文档给出的取值范围
# - quality: low / medium / high / auto
# - output_format: 推荐 png 或 jpeg（不建议 webp）
# - response_format: url / b64_json
DEFAULT_QUALITY = "auto"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_RESPONSE_FORMAT = "b64_json"
# 图生图时尽量保留原图主体和细节（文档说图片编辑可传 high）
DEFAULT_INPUT_FIDELITY = "high"
ALLOWED_QUALITIES = ("low", "medium", "high", "auto")
ALLOWED_OUTPUT_FORMATS = ("png", "jpeg")
ALLOWED_RESPONSE_FORMATS = ("url", "b64_json")

# 尺寸硬性规则（来自文档）
SIZE_MAX_SIDE = 3840
SIZE_MIN_PIXELS = 655_360
SIZE_MAX_PIXELS = 8_294_400
SIZE_ASPECT_LIMIT = 3.0
SIZE_MULTIPLE = 16

# 生图比聊天慢得多，默认给 180 秒
DEFAULT_TIMEOUT = 180.0
# 图片体积上限（4K PNG 可能到十几 MB，所以给宽一点）
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
# 下载 url 型返回时的超时
DOWNLOAD_TIMEOUT = 90.0

DEFAULTS = {
    "PAINT_ENABLED": "true",
    "PAINT_BASE_URL": DEFAULT_BASE_URL,
    "PAINT_MODEL": DEFAULT_MODEL,
    "PAINT_SIZE": DEFAULT_SIZE,
    "PAINT_QUALITY": DEFAULT_QUALITY,
    "PAINT_OUTPUT_FORMAT": DEFAULT_OUTPUT_FORMAT,
    "PAINT_RESPONSE_FORMAT": DEFAULT_RESPONSE_FORMAT,
    "PAINT_TIMEOUT_SECONDS": str(int(DEFAULT_TIMEOUT)),
    "PAINT_MAX_BYTES": str(DEFAULT_MAX_BYTES),
}


@dataclass
class PaintConfig:
    """生图配置（全部来自环境变量，非法值安全回落）。"""

    enabled: bool
    api_key: str
    base_url: str
    model: str
    size: str
    quality: str
    output_format: str
    response_format: str
    moderation: str
    input_fidelity: str
    timeout: float
    max_bytes: int

    @property
    def configured(self) -> bool:
        """真正可用：开关打开 + 配了 key。"""
        return bool(self.enabled and self.api_key)


@dataclass
class GeneratedImage:
    """一次生图的结果。"""

    data: bytes
    mime: str
    source: str  # "b64_json" 或 "url"，仅用于日志
    revised_prompt: str | None = None


# ==========================================================================
# 纯函数（便于测试）
# ==========================================================================


def validate_size(size: str) -> tuple[str, str | None]:
    """按文档规则校验尺寸。

    返回 (可用尺寸, 拒绝原因)。`auto` 永远合法。
    规则：
    - 形如 `<宽>x<高>`，宽高都必须是 16 的倍数；
    - 最大边长 ≤ 3840；
    - 长边 / 短边 ≤ 3；
    - 总像素数在 655,360 ~ 8,294,400 之间。
    """
    text = (size or "").strip().lower()
    if not text:
        return DEFAULT_SIZE, "空的尺寸"
    if text == "auto":
        return "auto", None

    match = re.fullmatch(r"(\d{1,5})\s*[x×*]\s*(\d{1,5})", text)
    if not match:
        return DEFAULT_SIZE, f"无法解析的尺寸 {size!r}"

    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return DEFAULT_SIZE, "宽高必须为正"
    if max(width, height) > SIZE_MAX_SIDE:
        return DEFAULT_SIZE, f"最大边长不能超过 {SIZE_MAX_SIDE}"
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE:
        return DEFAULT_SIZE, f"宽高都必须是 {SIZE_MULTIPLE} 的倍数"
    ratio = max(width, height) / min(width, height)
    if ratio > SIZE_ASPECT_LIMIT:
        return DEFAULT_SIZE, f"长边/短边不能超过 {SIZE_ASPECT_LIMIT:g}:1"
    pixels = width * height
    if pixels < SIZE_MIN_PIXELS or pixels > SIZE_MAX_PIXELS:
        return DEFAULT_SIZE, (
            f"总像素数需在 {SIZE_MIN_PIXELS}~{SIZE_MAX_PIXELS} 之间（当前 {pixels}）"
        )
    return f"{width}x{height}", None


def guess_mime(data: bytes) -> str:
    """按文件头判断图片类型；判断不出来就按 png 处理。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


# 兼容旧名字（测试与外部调用都用这个）
_guess_mime = guess_mime


def extract_image_bytes(item, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[bytes, str] | None:
    """从一个 image 结果对象里取出图片字节（纯函数，便于测试）。

    - item.b64_json 有值 → base64 解码（本 Bot 默认走这条）；
    - 否则 item.url 有值 → 返回空字节 + "url" 标记，交给上层去下载；
    - 取不到 / 解码失败 / 超过体积上限 → None。
    """
    raw_b64 = (getattr(item, "b64_json", None) or "").strip()
    if raw_b64:
        try:
            data = base64.b64decode(raw_b64, validate=False)
        except Exception:
            return None
        if not data or len(data) > max_bytes:
            return None
        return data, guess_mime(data)

    url = (getattr(item, "url", None) or "").strip()
    if url:
        return b"", "url"
    return None


# ==========================================================================
# 配置
# ==========================================================================


def _as_float(raw: str, fallback: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _as_int(raw: str, fallback: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _as_bool(raw: str, fallback: bool = True) -> bool:
    text = (raw or "").strip().lower()
    if not text:
        return fallback
    return text not in ("0", "false", "no", "off")


def _one_of(value: str, allowed: tuple[str, ...], fallback: str, label: str) -> str:
    text = (value or "").strip().lower()
    if text in allowed:
        return text
    if text:
        logger.warning("[PAINT] {}={} 不在允许列表 {} 内，回落到 {}", label, text, allowed, fallback)
    return fallback


def get_config(env: dict | None = None) -> PaintConfig:
    """从环境变量解析配置。非法值一律回落，绝不因为一个错别字让 Bot 起不来。"""
    src = os.environ if env is None else env

    def val(key: str) -> str:
        return str(src.get(key) or DEFAULTS.get(key) or "").strip()

    size, reason = validate_size(val("PAINT_SIZE"))
    if reason:
        logger.warning("[PAINT] PAINT_SIZE 不合法（{}），回落到 {}", reason, size)

    return PaintConfig(
        enabled=_as_bool(val("PAINT_ENABLED"), True),
        api_key=str(src.get("PAINT_API_KEY") or "").strip(),
        base_url=val("PAINT_BASE_URL") or DEFAULT_BASE_URL,
        model=val("PAINT_MODEL") or DEFAULT_MODEL,
        size=size,
        quality=_one_of(val("PAINT_QUALITY"), ALLOWED_QUALITIES, DEFAULT_QUALITY, "PAINT_QUALITY"),
        output_format=_one_of(
            val("PAINT_OUTPUT_FORMAT"), ALLOWED_OUTPUT_FORMATS, DEFAULT_OUTPUT_FORMAT,
            "PAINT_OUTPUT_FORMAT",
        ),
        response_format=_one_of(
            val("PAINT_RESPONSE_FORMAT"), ALLOWED_RESPONSE_FORMATS, DEFAULT_RESPONSE_FORMAT,
            "PAINT_RESPONSE_FORMAT",
        ),
        # moderation 留空表示跟随服务端默认值；填了才发过去
        moderation=(str(src.get("PAINT_MODERATION") or "").strip().lower()),
        # 图生图时保留原图主体的程度（文档：可传 high）。
        # 显式写空字符串 = 关闭该字段；键不存在 = 用默认值。
        input_fidelity=(
            str(src["PAINT_INPUT_FIDELITY"]).strip().lower()
            if "PAINT_INPUT_FIDELITY" in src
            else DEFAULT_INPUT_FIDELITY
        ),
        timeout=_as_float(val("PAINT_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT),
        max_bytes=_as_int(val("PAINT_MAX_BYTES"), DEFAULT_MAX_BYTES),
    )


CONFIG = get_config()

if CONFIG.configured:
    # 只记「配了什么」，绝不打印 key 本身
    logger.info(
        "[PAINT] 生图已启用：model={} size={} quality={} format={}/{} base_url={}",
        CONFIG.model,
        CONFIG.size,
        CONFIG.quality,
        CONFIG.output_format,
        CONFIG.response_format,
        CONFIG.base_url,
    )
elif CONFIG.enabled:
    logger.info("[PAINT] 未配置 PAINT_API_KEY，生图命令不可用")
else:
    logger.info("[PAINT] PAINT_ENABLED=false，生图命令已关闭")


_client: AsyncOpenAI | None = None


def _get_client(cfg: PaintConfig) -> AsyncOpenAI:
    """懒加载全局唯一的异步客户端（首次调用时才创建，复用底层连接池）。"""
    global _client
    if _client is None:
        if not cfg.api_key:
            raise RuntimeError("缺少 PAINT_API_KEY")
        _client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=1,  # 生图较慢，重试次数刻意保守（避免重复计费）
        )
    return _client


def build_request_body(prompt: str, cfg: PaintConfig) -> dict:
    """构造请求体（纯函数，便于测试）。

    只描述「要发什么 JSON」，不关心怎么发 —— 发送时除 model / prompt / n 之外的
    字段统一通过 extra_body 传给 SDK（见 generate_image）。
    """
    body: dict = {
        "model": cfg.model,
        "prompt": prompt,
        "size": cfg.size,
        "quality": cfg.quality,
        "n": 1,  # 文档：仅支持 1，多图请循环请求
        "output_format": cfg.output_format,
        "response_format": cfg.response_format,
    }
    if cfg.moderation:
        body["moderation"] = cfg.moderation
    # 刻意不传：stream / partial_images（文档说不支持）、style（文档说不建议）
    return body


# SDK 只认识它自己声明过的参数；size / quality / output_format / response_format /
# moderation 这类字段要么 SDK 没有、要么它的枚举定义是给旧模型的，所以统一走 extra_body，
# 让它们原样进入 JSON body（否则会直接 TypeError，或者被 SDK 的类型改写）。
_STANDARD_KEYS = ("model", "prompt", "n")


def build_edit_body(prompt: str, cfg: PaintConfig) -> dict:
    """构造图生图请求体（纯函数）。

    与文生图同一套参数，另外加上 input_fidelity —— 文档说图片编辑时传 high
    可以尽量保留原图主体和细节。
    """
    body = build_request_body(prompt, cfg)
    if cfg.input_fidelity:
        body["input_fidelity"] = cfg.input_fidelity
    return body


async def _download(url: str, max_bytes: int) -> bytes | None:
    """下载 url 型返回（response_format=url 时）。"""
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
        if not data or len(data) > max_bytes:
            logger.warning("[PAINT] 下载到的图片为空或超过上限（{} 字节）", len(data) if data else 0)
            return None
        return data
    except Exception as exc:
        logger.error("[PAINT] 下载图片失败：{}: {}", type(exc).__name__, redact_secrets(str(exc)))
        return None


async def edit_image(
    prompt: str,
    image: bytes,
    mime: str = "image/png",
    cfg: PaintConfig | None = None,
) -> GeneratedImage | None:
    """在给定图片的基础上改图（图生图）。失败返回 None，不抛异常。"""
    conf = cfg or CONFIG
    if not conf.configured:
        return None
    if not image:
        return None

    body = build_edit_body(prompt, conf)
    standard = {k: v for k, v in body.items() if k in _STANDARD_KEYS}
    extra = {k: v for k, v in body.items() if k not in _STANDARD_KEYS}
    # SDK 的 image 参数接受 (文件名, 内容, content-type) 三元组
    suffix = {"image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(mime, "png")
    upload = (f"reference.{suffix}", image, mime)

    try:
        response = await _get_client(conf).images.edit(image=upload, **standard, extra_body=extra)
    except Exception as exc:
        logger.error(
            "[PAINT] 图生图请求失败：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc))[:300],
        )
        return None

    return await _response_to_image(response, conf, mode="edit")


async def fetch_image_bytes(url: str, max_bytes: int | None = None) -> bytes | None:
    """按 URL 取回图片字节（图生图要用原图上传，所以必须落到本地）。"""
    limit = max_bytes or CONFIG.max_bytes
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:
        logger.error("[PAINT] 取参考图失败：{}: {}", type(exc).__name__, redact_secrets(str(exc))[:200])
        return None
    if not data or len(data) > limit:
        logger.warning("[PAINT] 参考图为空或超过上限（{} 字节）", len(data) if data else 0)
        return None
    return data


async def _response_to_image(response, conf: PaintConfig, mode: str) -> GeneratedImage | None:
    """把 API 返回统一转成 GeneratedImage（b64 解码 / url 下载两种形态）。"""
    items = list(getattr(response, "data", None) or [])
    if not items:
        logger.warning("[PAINT] 生图返回为空（没有 data）")
        return None

    item = items[0]
    extracted = extract_image_bytes(item, conf.max_bytes)
    if extracted is None:
        logger.warning("[PAINT] 生图返回里既没有 b64_json 也没有 url")
        return None

    data, mime = extracted
    source = "b64_json"
    if not data:  # url 形态
        source = "url"
        downloaded = await _download((getattr(item, "url", "") or "").strip(), conf.max_bytes)
        if downloaded is None:
            return None
        data, mime = downloaded, guess_mime(downloaded)

    revised = (getattr(item, "revised_prompt", None) or "").strip() or None
    # 只记尺寸与来源，绝不记 base64 / URL / prompt 全文
    logger.info(
        "[PAINT] 生图成功（{}）：{} 字节 mime={} source={}", mode, len(data), mime, source
    )
    return GeneratedImage(data=data, mime=mime, source=source, revised_prompt=revised)


async def generate_image(prompt: str, cfg: PaintConfig | None = None) -> GeneratedImage | None:
    """根据 prompt 生成一张图片（文生图）。失败返回 None，调用方不用 try。"""
    conf = cfg or CONFIG
    if not conf.configured:
        return None

    body = build_request_body(prompt, conf)
    standard = {k: v for k, v in body.items() if k in _STANDARD_KEYS}
    extra = {k: v for k, v in body.items() if k not in _STANDARD_KEYS}

    try:
        response = await _get_client(conf).images.generate(**standard, extra_body=extra)
    except Exception as exc:
        # 只记异常类型和摘要；先过 redact_secrets，绝不把 key 带进日志
        logger.error(
            "[PAINT] 生图请求失败：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc))[:300],
        )
        return None

    return await _response_to_image(response, conf, mode="generate")
