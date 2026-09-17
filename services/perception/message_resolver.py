"""Message Resolver（v0.7）：整个项目唯一的 QQ 消息解释层。

插件层不再针对 image / reply / forward / file 写 if/else 特殊分支；
真正的消息解释全部集中在这里：

    QQ Event
      ↓  Authorization / Mode Detection（仍在插件层，最先执行）
      ↓
    Message Resolver（本模块：递归解析 reply / forward / file）
      ↓
    NormalizedMessage（content.py，保持原始顺序）
      ↓
    Multimodal Content Builder（multimodal_builder.py）
      ↓
    Conversation Context（Persona / Memory / Relationship 仍由原 pipeline 负责）

本模块只回答“用户发了什么”，**绝不**决定“夜子会怎么回应”：
不写人格文案、不做模板化回复、不 import prompt_builder / persona_rag / memory。

支持范围（全部走同一条递归路径，不为任何类型写第二套逻辑）：

| segment | 处理 |
| --- | --- |
| text / at | 纯文本（at 本体不进入正文，与 get_plaintext 语义一致） |
| image | 复用 services/vision.py 的 VisionImage + 格式转换 |
| reply | OneBot get_msg → **再次进入本 Resolver** → NormalizedMessage.reply |
| forward | segment 自带 content 优先，否则 get_forward_msg → 递归解析节点 |
| file | services/file_reader.py（扩展名 + magic bytes 双重校验 → 只读解析） |
| 其它 | 结构化 SystemNotice（视频 / 语音 / 未知类型，绝不假装理解） |

硬限制与降级（全部在 services/perception/limits.py 统一定义）：
- REPLY_MAX_DEPTH：被回复消息链最大展开层数，超出补“[被回复的消息内容过深，未继续展开]”；
- visited_message_ids：A 回复 B、B 回复 A 的循环直接停止（不再展开）；
- FORWARD_MAX_DEPTH / FORWARD_MAX_NODES / FORWARD_MAX_IMAGES / FORWARD_MAX_FILES /
  FORWARD_MAX_TEXT_CHARS：超出只加结构化说明，**绝不报错**；
- visited_forward_ids：A → B → A 的嵌套转发循环直接停止；
- get_msg / get_forward_msg 的 timeout / API error / 消息已删除 / 无权限
  全部 graceful degradation，聊天流程继续。

隐私（需求 26 / 27）：
- 本模块**不写任何数据库**：落库占位符由 build_context_text() 生成；
- 日志只记 group_id / user_id / message_id / 数量 / 类别 / 耗时，
  绝不记图片 URL、CDN token、Base64、文件正文、转发正文。
"""

import asyncio
import time
from dataclasses import dataclass
from dataclasses import field

from nonebot import logger

from services.perception.content import CONTENT_ITEM_TYPES
from services.perception.content import ContentItem
from services.perception.content import FileContent
from services.perception.content import FileRef
from services.perception.content import ForwardContent
from services.perception.content import ForwardNode
from services.perception.content import ImageContent
from services.perception.content import MessageMetadata
from services.perception.content import NormalizedMessage
from services.perception.content import SystemNotice
from services.perception.content import TextContent
from services.perception.file_reader import cleanup_result
from services.perception.file_reader import read_file
from services.perception.file_reader import sanitize_file_name
from services.perception.limits import BUDGET_SOURCE_CURRENT
from services.perception.limits import BUDGET_SOURCE_FORWARD
from services.perception.limits import BUDGET_SOURCE_REPLY
from services.perception.limits import FORWARD_MAX_DEPTH
from services.perception.limits import FORWARD_MAX_FILES
from services.perception.limits import FORWARD_MAX_IMAGES
from services.perception.limits import FORWARD_MAX_NODES
from services.perception.limits import FORWARD_MAX_TEXT_CHARS
from services.perception.limits import FORWARD_UNAVAILABLE_TEXT
from services.perception.limits import REPLY_MAX_DEPTH
from services.perception.limits import REPLY_UNAVAILABLE_TEXT
from services.perception.limits import ContentBudget
from services.perception.limits import image_limit_note
from services.perception.limits import truncation_note
from services.perception.multimodal_builder import ConversationContent
from services.perception.multimodal_builder import build_conversation_content
from services.perception.net import cleanup_download
from services import vision as vision_module
from services.vision import SUPPORTED_IMAGE_EXTENSIONS
from services.vision import VISION_DETAIL
from services.vision import VISION_MAX_IMAGE_BYTES
from services.vision import VISION_MAX_IMAGES
from services.vision import VisionImage

__all__ = [
    "MessageResolver",
    "ResolvedConversation",
    "ConversationContent",
    "build_data_url_from_file",
    "segment_data",
]

# ===== 稳定的降级文案（结构化程序事实，不是人格回复）=====

REPLY_DEPTH_LIMIT_TEXT = "[被回复的消息内容过深，未继续展开]"
REPLY_CYCLE_TEXT = "[引用链出现循环，已停止展开]"
FORWARD_DEPTH_LIMIT_TEXT = "[该合并转发嵌套过深，未继续展开]"
FORWARD_CYCLE_TEXT = "[该合并转发出现循环引用，已停止展开]"
FORWARD_EMPTY_TEXT = "[该合并转发没有可读取的内容]"
REPLY_EMPTY_TEXT = "[被回复的消息没有可读取的内容]"
VIDEO_NOTICE_TEXT = "[用户发送了一段视频，当前版本不支持视频理解]"
RECORD_NOTICE_TEXT = "[用户发送了一段语音，当前版本不支持语音理解]"
UNKNOWN_SEGMENT_TEXT = "[用户发送了一条暂不支持解析的消息类型：{segment_type}]"

IMAGE_READ_FAILED_TEXT = "[用户发送了一张图片，但图片读取失败]"
IMAGE_VISION_DISABLED_TEXT = "[用户发送了一张图片，但视觉能力当前已关闭]"
IMAGE_NO_URL_TEXT = "[用户发送了一张图片，但找不到可读取的地址]"
IMAGE_TOO_LARGE_TEXT = "[用户发送了一张图片，但图片体积超过限制，未读取]"

# OneBot 调用超时（秒）：竞态于 NapCat 侧，失败只降级不报错。
API_TIMEOUT_SECONDS = 15.0

# 图片扩展名 → data URL MIME type
_IMAGE_MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}

# 一次请求内最多展示的图片数（可被 vision_max_images / 请求预算进一步收紧）。
MAX_IMAGES_HARD_LIMIT = 10

# 群文件默认 busid（OneBot get_group_file_url 的可选参数，部分实现必填）。
FILE_BUSID_DEFAULT = 102


def segment_data(segment) -> dict:
    """安全读取 OneBot MessageSegment 的 data（可能是 dict / 其它）。"""
    data = getattr(segment, "data", None)
    return data if isinstance(data, dict) else {}


def segment_type(segment) -> str:
    """安全读取 MessageSegment 的类型字符串。"""
    return str(getattr(segment, "type", "") or "")


def _as_str(value) -> str:
    return "" if value is None else str(value)


class _DictSegment:
    """把 API 返回的 segment dict 包装成与 MessageSegment 相同的访问形状。

    NapCat / OneBot 的 get_msg / get_forward_msg 返回的是
    `{"type": "text", "data": {...}}` 这样的**普通 dict**，而事件里的
    MessageSegment 是带 `.type` / `.data` 属性的对象。统一包装之后，
    同一套解析逻辑（segment_type / segment_data）对两种来源都成立，
    不会出现“API 路径解析不出任何内容”的隐蔽 bug。
    """

    __slots__ = ("type", "data")

    def __init__(self, raw: dict) -> None:
        self.type = str(raw.get("type") or "")
        data = raw.get("data")
        self.data = data if isinstance(data, dict) else {}

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"_DictSegment(type={self.type!r})"


def as_segments(value) -> list:
    """把事件 / API 返回的消息内容统一成 segment 列表。

    - list / tuple：逐项处理；
    - 其中 `{"type": ..., "data": {...}}` 形态的 dict 包装成 _DictSegment；
    - 其它 dict（例如合并转发节点 `{"user_id": ..., "message": [...]}`）
      **原样保留**，由 _convert_forward_node 自己按节点结构解析；
    - 字符串（少数实现返回 CQ Code）：不自己正则解析，返回空列表。
    """
    if value is None or isinstance(value, str):
        return []
    if not isinstance(value, (list, tuple)):
        return []
    result: list = []
    for raw in value:
        if isinstance(raw, dict) and "type" in raw:
            result.append(_DictSegment(raw))
        else:
            result.append(raw)
    return result


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_mapping(value) -> dict:
    """把 dict / _DictSegment 统一看成 dict（合并转发节点两种形态都要支持）。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, _DictSegment):
        return value.data
    return {}


def _get(value, key: str):
    """从 dict / pydantic 模型 / 任意对象里安全取字段（None 安全）。

    适配器给的 `event.reply` 是 pydantic 模型（属性访问），OneBot API 返回的
    是普通 dict（键访问）。同一套逻辑必须同时支持两种形态，否则就会出现
    “API 路径读得到、适配器路径读不到”的隐蔽漏读。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _first_str(source: dict, *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_int(source: dict, *keys: str) -> int:
    for key in keys:
        value = _as_int(source.get(key))
        if value:
            return value
    return 0


def _is_http_url(value: str) -> bool:
    """是否是可直接交给模型的 http(s) 外链（拒绝本机路径 / file id / data URL）。"""
    text = (value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def _first_url(source: dict, *keys: str) -> str:
    """取第一个真正的 http(s) 外链；没有则返回空字符串（绝不返回本机路径）。"""
    for key in keys:
        value = _first_str(source, key)
        if value and _is_http_url(value):
            return value
    return ""


async def build_data_url_from_file(path: str, file_name: str = "") -> VisionImage | None:
    """把本机图片文件读成 data URL（内存中转，绝不落库）。

    只用于「文件就是图片」的场景（QQ 文件段里发 .png/.jpg）：
    复用现有 Vision Pipeline 的 data_url 通道，不新增第二套视觉逻辑。
    """
    import base64
    import io
    import os

    suffix = ""
    base = os.path.basename(path)
    if "." in base:
        suffix = base.rsplit(".", 1)[-1].lower()
    if not suffix and file_name and "." in file_name:
        suffix = file_name.rsplit(".", 1)[-1].lower()
    mime = _IMAGE_MIME_TYPES.get(suffix, "image/png")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    if not raw:
        return None
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        # 复用 vision.py 已有的格式兼容逻辑（Pillow 内存转 JPEG data URL）。
        try:
            from PIL import Image as PILImage

            with PILImage.open(io.BytesIO(raw)) as img:
                converted = img.convert("RGB")
                buffer = io.BytesIO()
                converted.save(buffer, format="JPEG", quality=85)
            return VisionImage(
                url="",
                detail=VISION_DETAIL,
                file_name=file_name or base,
                data_url="data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii"),
            )
        except Exception:
            return None
    return VisionImage(
        url="",
        detail=VISION_DETAIL,
        file_name=file_name or base,
        data_url=f"data:{mime};base64," + base64.b64encode(raw).decode("ascii"),
    )


@dataclass
class ResolvedConversation:
    """一次 DIRECT 请求的完整感知结果（当前消息 + 被回复消息链）。

    notice 是**程序生成的资源提示**（图片/文件/转发超限说明），
    属于可信程序事实，最终由 multimodal_builder 放进可信状态块，
    绝不平铺进用户文本。
    """

    message: NormalizedMessage
    reply: NormalizedMessage | None = None
    notice: str = ""
    image_total: int = 0
    image_accepted: int = 0
    file_total: int = 0
    forward_node_total: int = 0
    budget: ContentBudget | None = None
    latency_ms: int = 0


class _BudgetState:
    """一次解析过程中的资源计数（节点 / 图片 / 文件 / 转发节点序号）。"""

    def __init__(
        self,
        max_images: int,
        max_files: int,
        max_nodes: int,
        max_text_chars: int,
    ) -> None:
        self.max_images = max(0, max_images)
        self.max_files = max(0, max_files)
        self.max_nodes = max(0, max_nodes)
        self.max_text_chars = max(0, max_text_chars)
        self.images = 0
        self.files = 0
        self.nodes = 0
        self.text_chars = 0
        self.notices: list[str] = []

    def allow_image(self) -> bool:
        if self.images >= self.max_images:
            return False
        self.images += 1
        return True

    def allow_file(self) -> bool:
        if self.files >= self.max_files:
            return False
        self.files += 1
        return True

    def allow_node(self) -> bool:
        if self.nodes >= self.max_nodes:
            return False
        self.nodes += 1
        return True

    def clip_text(self, text: str) -> str:
        remaining = self.max_text_chars - self.text_chars
        if remaining <= 0:
            self.notices.append(
                truncation_note(1, "条转发文本").replace("1 条转发文本", "转发文本")
            )
            return ""
        if len(text) <= remaining:
            self.text_chars += len(text)
            return text
        clipped = text[:remaining] + "…（已截断）"
        self.text_chars += len(clipped)
        self.notices.append("[合并转发文本超过长度限制，后续文本未展开]")
        return clipped

    def note(self, text: str) -> None:
        if text:
            self.notices.append(text)


class MessageResolver:
    """统一 QQ Message Resolver（一次 DIRECT 请求创建一个实例）。

    一个实例贯穿“当前消息 → 被回复消息 → 合并转发节点 → 嵌套转发”的全部
    递归解析，因此 visited_message_ids / visited_forward_ids / 预算计数在整个
    请求内共享——循环保护与硬限制天然覆盖所有层级。
    """

    def __init__(
        self,
        bot=None,
        *,
        budget: ContentBudget | None = None,
        reply_max_depth: int | None = None,
        forward_max_depth: int | None = None,
        forward_max_nodes: int | None = None,
        forward_max_images: int | None = None,
        forward_max_files: int | None = None,
        forward_max_text_chars: int | None = None,
    ) -> None:
        self.bot = bot
        self.budget = budget if budget is not None else ContentBudget()
        # 视觉能力在实例化时读取一次（与插件层同一份配置）：
        # 一次解析内部保持一致，不会出现“同一条消息里有的图片进模型、有的被丢弃”。
        self._vision_enabled = bool(getattr(vision_module, "VISION_ENABLED", False))

        # 各类硬限制：显式参数优先（测试与调用方可覆盖），否则用进程启动时解析的配置。
        self.reply_max_depth = REPLY_MAX_DEPTH if reply_max_depth is None else max(0, reply_max_depth)
        self.forward_max_depth = (
            FORWARD_MAX_DEPTH if forward_max_depth is None else max(0, forward_max_depth)
        )
        self.forward_max_nodes = (
            FORWARD_MAX_NODES if forward_max_nodes is None else max(0, forward_max_nodes)
        )
        self.forward_max_images = (
            FORWARD_MAX_IMAGES if forward_max_images is None else max(0, forward_max_images)
        )
        self.forward_max_files = (
            FORWARD_MAX_FILES if forward_max_files is None else max(0, forward_max_files)
        )
        self.forward_max_text_chars = (
            FORWARD_MAX_TEXT_CHARS
            if forward_max_text_chars is None
            else max(0, forward_max_text_chars)
        )

        # 循环保护（跨整个请求共享）
        self.visited_message_ids: set[str] = set()
        self.visited_forward_ids: set[str] = set()

        # 本次请求的「当前群」：群文件换下载直链时优先用它（机器人只对自己
        # 所在的群有取链接权限；合并转发里的文件属于原群，用原群会 1200）。
        self._request_group_id: int = 0

        # 待清理的图片临时目录（请求结束统一删除）
        self._temp_dirs: list[str] = []

    # ===== 视觉能力（每次实例化时读取当前配置） =====

    @property
    def vision_enabled(self) -> bool:
        """DIRECT 视觉是否开启（实例化时已由 vision 模块读取，运行期可被测试 / 开关覆盖）。"""
        return self._vision_enabled

    @property
    def vision_max_images(self) -> int:
        """本次请求实际允许交给模型的图片数（配置 / 硬上限 / 请求预算三者取最小）。"""
        configured = int(getattr(vision_module, "VISION_MAX_IMAGES", 0) or 0)
        return max(0, min(configured, MAX_IMAGES_HARD_LIMIT, self.budget.max_total_images))

    # ==================================================================
    # 对外 API
    # ==================================================================

    async def resolve_event(self, event) -> ResolvedConversation:
        """解析一条 GroupMessageEvent（含 reply / forward / file，递归）。"""
        started = time.monotonic()
        group_id = _as_int(getattr(event, "group_id", 0))
        user_id = _as_int(getattr(event, "user_id", 0))
        self_id = _as_int(getattr(event, "self_id", 0))
        message_id = _as_int(getattr(event, "message_id", 0))
        self._request_group_id = group_id

        segments = as_segments(getattr(event, "message", None))
        if not segments:
            getter = getattr(event, "get_message", None)
            if callable(getter):
                try:
                    segments = as_segments(getter())
                except Exception:
                    segments = []

        vision_enabled = self.vision_enabled
        # 一次请求内的图片总预算是「视觉上限」与「合并转发图片上限」的较小值：
        # 无论图片出现在当前消息、被回复消息还是转发节点里，都共用同一个计数器，
        # 因此 FORWARD_MAX_IMAGES 之类的硬限制无法被嵌套结构绕过。
        state = _BudgetState(
            max_images=(
                min(self.vision_max_images, self.forward_max_images)
                if vision_enabled
                else 0
            ),
            max_files=self.forward_max_files,
            max_nodes=self.forward_max_nodes,
            max_text_chars=self.forward_max_text_chars,
        )
        items, reply = await self._convert_segments(
            segments,
            source=BUDGET_SOURCE_CURRENT,
            state=state,
            depth=0,
            vision_enabled=vision_enabled,
            file_group_id=group_id,
        )

        # v0.7.1 修复（真实环境漏读引用/转发的根因）：
        # NoneBot 的 OneBot V11 适配器在事件进入任何 matcher **之前** 就会执行
        # `_check_reply()`——它先用 `bot.get_msg()` 把被回复消息取回来写进
        # `event.reply`，然后把 reply segment 从 `event.message` 里 **删除**。
        # 因此运行期事件里通常根本没有 reply segment，只扫 segment 必然得到
        # reply=0 / forward_nodes=0（引用里的合并转发永远读不到）。
        # 这里优先复用适配器已经取回的内容（不再重复调用一次 get_msg）；
        # segment 路径保留，用于适配器调用失败或非适配器调用方（单测直接构造事件）。
        if reply is None:
            preloaded_reply = getattr(event, "reply", None)
            if preloaded_reply is not None:
                reply = await self._resolve_preloaded_reply(
                    preloaded_reply,
                    group_id=group_id,
                    state=state,
                    depth=0,
                    vision_enabled=vision_enabled,
                )

        if not segments:
            # 事件没有可枚举的 segment（部分适配器 / 精简事件对象只提供纯文本，
            # 例如 get_message() 为空但 get_plaintext() 有内容）。
            # 这里用纯文本兜底，保证“有文字就一定进 AI pipeline”，
            # 不会因为拿不到 segment 而误判成“只 @ 了机器人”。
            plaintext = _event_plaintext(event)
            if plaintext:
                items.append(TextContent(text=plaintext))
        metadata = MessageMetadata(
            group_id=group_id,
            user_id=user_id,
            self_id=self_id,
            message_id=message_id,
            sender_name=_sender_name(event),
            message_type="group",
        )
        message = NormalizedMessage(items=tuple(items), metadata=metadata, reply=reply)

        raw_image_total = sum(1 for segment in segments if segment_type(segment) == "image")
        notice = self._consume_budget_notices(state)
        # file_total 必须**递归统计整棵内容树**（含引用与合并转发节点里的文件）。
        # 只数顶层会让「转发里的 PDF」在日志里显示成 file_total=0 —— 2026-09-16
        # 排查文件读取问题时，正是这个数字把方向带偏过一次。
        # 注意：image_total 保持「当前消息里的图片数」语义（ai_chat 用它判断
        # “只 @ 了机器人又没带图”），不要一起改成递归。
        all_items = list(_walk_items(items))
        if reply is not None:
            all_items.extend(_walk_items(reply.items))
        resolved = ResolvedConversation(
            message=message,
            reply=reply,
            notice=notice,
            image_total=raw_image_total,
            image_accepted=sum(
                1
                for item in items
                if isinstance(item, ImageContent) and item.image is not None
            ),
            file_total=sum(1 for item in all_items if isinstance(item, FileContent)),
            forward_node_total=state.nodes,
            budget=self.budget,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        logger.info(
            "[RESOLVER] group_id={} user_id={} message_id={} items={} images_total={} "
            "images_accepted={} reply={} file_total={} forward_nodes={} latency_ms={}",
            group_id,
            user_id,
            message_id,
            len(items),
            resolved.image_total,
            resolved.image_accepted,
            1 if reply is not None else 0,
            resolved.file_total,
            resolved.forward_node_total,
            resolved.latency_ms,
        )
        return resolved

    async def build_conversation(self, conversation: ResolvedConversation) -> ConversationContent:
        """把解析结果转成有序 multimodal blocks（不涉及 Persona / Memory）。"""
        content = await build_conversation_content(conversation)
        if not self.vision_enabled:
            # VISION_ENABLED=false：结果里保留 TextContent / 占位说明，
            # 但绝不把图片转换成 image block（含转发 / 文件里的图片）。
            content.items = [
                block for block in content.items if block.get("type") != "image_url"
            ]
            content.reply_images = []
            content.has_any_image = False
        return content

    def cleanup(self) -> None:
        """删除本次解析产生的全部图片临时文件（请求结束必须调用）。"""
        while self._temp_dirs:
            cleanup_download(self._temp_dirs.pop())

    # ==================================================================
    # 内部：segment → ContentItem（递归核心）
    # ==================================================================

    async def _convert_segments(
        self,
        segments: list,
        *,
        source: str,
        state: _BudgetState,
        depth: int,
        vision_enabled: bool,
        file_group_id: int = 0,
    ) -> tuple[list[ContentItem], NormalizedMessage | None]:
        """把一个 segment 序列按**原始顺序**转成 ContentItem 列表。

        vision_enabled 由最外层读取一次后沿递归传递：保证同一次解析里
        “视觉开关”不会中途变化（否则同一条消息里会出现有的图片进模型、
        有的图片被丢弃的不一致状态）。

        file_group_id 同样沿递归传递：群文件段只有 file_id、没有 url，
        换下载直链时必须带**文件所属的那个群号**（合并转发里的文件属于
        原群，不是当前群），否则会拿到 404 / 无权限的链接。
        """
        items: list[ContentItem] = []
        reply: NormalizedMessage | None = None
        for segment in segments:
            seg_type = segment_type(segment)
            data = segment_data(segment)
            if seg_type == "text":
                text = _as_str(data.get("text"))
                if text:
                    items.append(self._text_item(text, source=source, state=state))
            elif seg_type == "at":
                # @ 本体不进入正文（与 event.get_plaintext() 语义一致）
                continue
            elif seg_type == "image":
                items.append(self._image_item(data, state=state, vision_enabled=vision_enabled))
            elif seg_type == "reply":
                reply = await self._resolve_reply(
                    data,
                    state=state,
                    depth=depth,
                    vision_enabled=vision_enabled,
                    file_group_id=file_group_id,
                )
            elif seg_type == "forward":
                items.append(
                    await self._resolve_forward(
                        data,
                        state=state,
                        depth=depth,
                        vision_enabled=vision_enabled,
                        file_group_id=file_group_id,
                    )
                )
            elif seg_type == "file":
                items.append(
                    await self._resolve_file(
                        data,
                        state=state,
                        vision_enabled=vision_enabled,
                        file_group_id=file_group_id,
                    )
                )
            elif seg_type == "video":
                items.append(SystemNotice(text=VIDEO_NOTICE_TEXT))
            elif seg_type == "record":
                items.append(SystemNotice(text=RECORD_NOTICE_TEXT))
            elif seg_type in ("face", "mface", "bface", "sface", "rps", "dice", "shake"):
                # QQ 表情 / 骰子一类：没有可读文本，也不需要模型理解；
                # 保留一个简短事实，避免“消息里凭空少了一段”的困惑。
                items.append(SystemNotice(text="[用户发送了一个表情]"))
            elif seg_type in ("json", "xml"):
                items.append(SystemNotice(text="[用户发送了一张卡片消息，当前版本不解析卡片内容]"))
            else:
                items.append(SystemNotice(text=UNKNOWN_SEGMENT_TEXT.format(segment_type=seg_type or "unknown")))
        return items, reply

    # ===== 图片 =====

    def _text_item(self, text: str, *, source: str, state: _BudgetState) -> ContentItem:
        """文本项：**当前用户消息**永不裁剪，外部内容（转发 / 引用）按总预算截断。

        优先级（需求 31）：当前用户输入 > Reply > Forward > File > History。
        因此绝不能为了塞进一段长转发而裁掉用户刚说的话。
        """
        if source == BUDGET_SOURCE_CURRENT:
            return TextContent(text=text)
        clipped = state.clip_text(text)
        if not clipped:
            return SystemNotice(text="[该段文本超过长度限制，未展开]")
        return TextContent(text=clipped)

    def _image_item(self, data: dict, *, state: _BudgetState, vision_enabled: bool) -> ImageContent:
        """按现有 Vision 规则校验图片（白名单已在插件层完成）。

        只接受真正的 http(s) 外链：
        - `data["url"]` 是 NapCat 给出的 CDN 地址（唯一被信任的来源）；
        - `data["file"]` 在部分实现里是本机文件名 / QQ file id，**绝不能**当作
          URL 交给模型（构造不出可访问地址时按“读取不到图片”降级）；
        - `data:image/...;base64,...` 这类内联数据也不在这里展开（体量不可控，
          统一交给文件读取路径处理）。
        """
        if not vision_enabled:
            return ImageContent(image=None, placeholder=IMAGE_VISION_DISABLED_TEXT)
        url = _first_str(data, "url")
        if not _is_http_url(url):
            return ImageContent(image=None, placeholder=IMAGE_NO_URL_TEXT)
        size = _as_int(data.get("file_size"))
        if size and size > VISION_MAX_IMAGE_BYTES:
            return ImageContent(image=None, placeholder=IMAGE_TOO_LARGE_TEXT)
        if not state.allow_image():
            state.note(image_limit_note(1))
            return ImageContent(
                image=None,
                placeholder="[用户发送的图片数量超过限制，该图片未加载]",
            )
        image = VisionImage(
            url=url,
            detail=VISION_DETAIL,
            file_name=_first_str(data, "file", "name"),
        )
        return ImageContent(image=image)

    # ===== 文件 =====

    async def _resolve_file(
        self,
        data: dict,
        *,
        state: _BudgetState,
        vision_enabled: bool,
        file_group_id: int = 0,
    ) -> FileContent:
        file_name = _first_str(data, "name", "file", "file_name", "filename")
        file_id = _first_str(data, "file_id", "id", "file")
        url = _first_url(data, "url", "file_url", "download_url")
        if not url and file_id:
            # v0.7.2：NapCat 的群文件段只有 file/file_id/file_size，**根本没有 url**
            # （实测：`enableLocalFile2Url=false` 时如此，合并转发节点里也一样）。
            # 必须用 get_group_file_url 拿 file_id 换一个带时效的下载直链，
            # 否则文件永远停在“无下载地址”，转发里发 PDF 就永远读不到。
            url = await self._fetch_group_file_url(file_group_id, file_id)
        file_ref = FileRef(
            file_name=file_name,
            url=url,
            file_id=file_id,
            file_size=_as_int(data.get("file_size")),
            source="message",
        )
        return await self._read_file_item(
            file_ref, state=state, vision_enabled=vision_enabled
        )

    async def _fetch_group_file_url(self, file_group_id: int, file_id: str) -> str:
        """用 `get_group_file_url` 把群文件 file_id 换成下载直链。

        候选群号顺序经过真机实测确定（2026-09-16，NapCat v4.18）：
        - **当前群**（机器人实际所在的群）→ `retcode=0`，能拿到直链；
        - **文件所属原群**（合并转发里的文件来自别的群）→ `retcode=1200`，
          因为机器人不在那个群里，没有取文件链接的权限。

        所以合并转发里的文件也必须优先用「收到消息的当前群」去换，
        原群只作为兜底（机器人在该群时可用）。拿不到一律返回空字符串，
        由 File Reader 按“无下载地址”优雅降级，绝不中断整条消息的解析。
        """
        if not file_id:
            return ""
        candidates: list[int] = []
        for gid in (self._request_group_id, file_group_id):
            if gid and gid not in candidates:
                candidates.append(gid)
        # 同一群内先按标准参数调用；部分实现要求显式 busid，再补一次。
        for gid in candidates:
            for params in (
                {"group_id": gid, "file_id": file_id},
                {"group_id": gid, "file_id": file_id, "busid": FILE_BUSID_DEFAULT},
            ):
                payload = await self._call_api("get_group_file_url", **params)
                url = _first_url(
                    _as_mapping(payload), "url", "file_url", "download_url"
                )
                if url:
                    return url
        logger.info("[FILE] get_group_file_url 未返回可用地址，按无下载地址降级")
        return ""

    async def _read_file_item(
        self, file_ref: FileRef, *, state: _BudgetState, vision_enabled: bool
    ) -> FileContent:
        """读取一个文件并转成 FileContent（绝不抛异常、绝不落库）。"""
        display_name = sanitize_file_name(file_ref.file_name)
        if not state.allow_file():
            return FileContent(
                file_ref=file_ref,
                file_name=display_name,
                ok=False,
                note="[该消息中的文件数量超过限制，该文件未读取]",
            )
        result = await read_file(file_ref, budget=self.budget)
        if result.parser_type == "image" and result.local_path:
            if result.local_dir:
                self._temp_dirs.append(result.local_dir)
            vision = (
                await build_data_url_from_file(result.local_path, result.file_name)
                if vision_enabled
                else None
            )
            if vision is not None and state.allow_image():
                return FileContent(
                    file_ref=file_ref,
                    file_name=result.file_name or display_name,
                    parser_type="image",
                    image=vision,
                    ok=True,
                )
            cleanup_result(result)
            return FileContent(
                file_ref=file_ref,
                file_name=result.file_name or display_name,
                parser_type="image",
                ok=False,
                note=IMAGE_READ_FAILED_TEXT,
            )
        if not result.ok:
            return FileContent(
                file_ref=file_ref,
                file_name=result.file_name or display_name,
                parser_type=result.parser_type,
                ok=False,
                note=result.note or f"[用户发送了文件 {display_name}，但读取失败]",
            )
        if not result.text:
            return FileContent(
                file_ref=file_ref,
                file_name=result.file_name or display_name,
                parser_type=result.parser_type,
                ok=True,
                note=result.note,
            )
        clipped = state.clip_text(result.text)
        if not clipped:
            return FileContent(
                file_ref=file_ref,
                file_name=result.file_name or display_name,
                parser_type=result.parser_type,
                ok=False,
                note="[合并转发文本超过长度限制，该文件正文未展开]",
            )
        return FileContent(
            file_ref=file_ref,
            file_name=result.file_name or display_name,
            parser_type=result.parser_type,
            text=clipped,
            ok=True,
        )

    # ===== Reply =====

    async def _resolve_reply(
        self,
        data: dict,
        *,
        state: _BudgetState,
        depth: int,
        vision_enabled: bool,
        file_group_id: int = 0,
    ) -> NormalizedMessage:
        """解析被回复消息：get_msg → 再次进入同一个 Resolver（递归）。"""
        message_id = _first_str(data, "id", "message_id")
        if not message_id:
            return NormalizedMessage(items=(SystemNotice(text=REPLY_UNAVAILABLE_TEXT),))
        if message_id in self.visited_message_ids:
            logger.info("[RESOLVER] reply 循环引用，停止展开 message_id={}", message_id)
            return NormalizedMessage(items=(SystemNotice(text=REPLY_CYCLE_TEXT),))
        if depth + 1 > self.reply_max_depth:
            logger.info(
                "[RESOLVER] reply 达到最大深度 depth={} max={}",
                depth + 1,
                self.reply_max_depth,
            )
            return NormalizedMessage(
                items=(SystemNotice(text=REPLY_DEPTH_LIMIT_TEXT),)
            )
        self.visited_message_ids.add(message_id)

        payload = await self._call_api("get_msg", message_id=message_id)
        if payload is None:
            return NormalizedMessage(items=(SystemNotice(text=REPLY_UNAVAILABLE_TEXT),))

        message_field = payload.get("message")
        segments = as_segments(message_field)
        if not segments and isinstance(message_field, str):
            # 少数实现返回 CQ Code 字符串；不自己正则解析（避免第二套解析逻辑），
            # 直接按“无法读取”降级，绝不猜测内容。
            logger.info("[RESOLVER] get_msg 返回字符串 message，按无法解析处理")
            return NormalizedMessage(items=(SystemNotice(text=REPLY_UNAVAILABLE_TEXT),))

        sender = _as_mapping(payload.get("sender"))
        sender_name = _first_str(sender, "card", "nickname") or _first_str(
            payload, "sender_name"
        )
        metadata = MessageMetadata(
            group_id=_as_int(payload.get("group_id")),
            user_id=_first_int(payload, "user_id") or _first_int(sender, "user_id"),
            self_id=0,
            message_id=_as_int(payload.get("message_id")) or _as_int(message_id),
            sender_name=sender_name,
            message_type=_first_str(payload, "message_type") or "group",
        )
        items, nested_reply = await self._convert_segments(
            segments,
            source=BUDGET_SOURCE_REPLY,
            state=state,
            depth=depth + 1,
            vision_enabled=vision_enabled,
            file_group_id=_as_int(payload.get("group_id")) or file_group_id,
        )
        if not items and nested_reply is None:
            items = [SystemNotice(text=REPLY_EMPTY_TEXT)]
        return NormalizedMessage(
            items=tuple(items), metadata=metadata, reply=nested_reply
        )

    async def _resolve_preloaded_reply(
        self,
        preloaded,
        *,
        group_id: int,
        state: _BudgetState,
        depth: int,
        vision_enabled: bool,
    ) -> NormalizedMessage:
        """解析适配器已预取的 `event.reply`（NoneBot `_check_reply` 的产物）。

        与 `_resolve_reply` 的差别：**不调用 get_msg**——被回复消息的内容已经
        在事件对象里，再请求一次纯属浪费一次 API 往返（而且真实环境里 segment
        已被适配器删除，根本无从触发）。深度限制、循环保护、预算与降级语义
        与 `_resolve_reply` 完全保持一致。
        """
        message_id = _as_str(_get(preloaded, "message_id")).strip()
        if message_id and message_id in self.visited_message_ids:
            logger.info("[RESOLVER] reply 循环引用，停止展开 message_id={}", message_id)
            return NormalizedMessage(items=(SystemNotice(text=REPLY_CYCLE_TEXT),))
        if depth + 1 > self.reply_max_depth:
            logger.info(
                "[RESOLVER] reply 达到最大深度 depth={} max={}",
                depth + 1,
                self.reply_max_depth,
            )
            return NormalizedMessage(items=(SystemNotice(text=REPLY_DEPTH_LIMIT_TEXT),))
        if message_id:
            self.visited_message_ids.add(message_id)

        segments = as_segments(_get(preloaded, "message"))
        sender = _get(preloaded, "sender")
        sender_name = _as_str(_get(sender, "card")).strip() or _as_str(
            _get(sender, "nickname")
        ).strip()
        metadata = MessageMetadata(
            group_id=group_id,
            user_id=_as_int(_get(sender, "user_id")),
            self_id=0,
            message_id=_as_int(_get(preloaded, "message_id")),
            sender_name=sender_name,
            message_type=_as_str(_get(preloaded, "message_type")) or "group",
        )
        items, nested_reply = await self._convert_segments(
            segments,
            source=BUDGET_SOURCE_REPLY,
            state=state,
            depth=depth + 1,
            vision_enabled=vision_enabled,
            file_group_id=_as_int(_get(preloaded, "group_id")) or group_id,
        )
        if not items and nested_reply is None:
            items = [SystemNotice(text=REPLY_EMPTY_TEXT)]
        return NormalizedMessage(
            items=tuple(items), metadata=metadata, reply=nested_reply
        )

    # ===== Forward（合并转发）=====

    async def _resolve_forward(
        self,
        data: dict,
        *,
        state: _BudgetState,
        depth: int,
        vision_enabled: bool,
        file_group_id: int = 0,
    ) -> ForwardContent:
        """解析合并转发：自带 content 优先，否则 get_forward_msg，然后递归节点。"""
        forward_id = _first_str(data, "id", "forward_id", "res_id", "file")
        inline = as_segments(data.get("content")) or as_segments(data.get("messages"))

        if forward_id and forward_id in self.visited_forward_ids:
            logger.info("[RESOLVER] forward 循环引用，停止展开 forward_id={}", forward_id)
            return ForwardContent(
                forward_id=forward_id,
                nodes=(),
                ok=False,
                note=FORWARD_CYCLE_TEXT,
            )
        if forward_id:
            self.visited_forward_ids.add(forward_id)

        if depth + 1 > self.forward_max_depth:
            logger.info(
                "[RESOLVER] forward 达到最大深度 depth={} max={}",
                depth + 1,
                self.forward_max_depth,
            )
            return ForwardContent(
                forward_id=forward_id,
                nodes=(),
                ok=False,
                note=FORWARD_DEPTH_LIMIT_TEXT,
            )

        raw_nodes = inline
        if not raw_nodes:
            payload = await self._call_api("get_forward_msg", id=forward_id) if forward_id else None
            if payload is None:
                return ForwardContent(
                    forward_id=forward_id,
                    nodes=(),
                    ok=False,
                    note=FORWARD_UNAVAILABLE_TEXT,
                )
            raw_nodes = (
                as_segments(payload.get("messages"))
                or as_segments(payload.get("message"))
                or as_segments(payload.get("data"))
            )

        if not raw_nodes:
            return ForwardContent(
                forward_id=forward_id, nodes=(), ok=False, note=FORWARD_EMPTY_TEXT
            )

        nodes: list[ForwardNode] = []
        skipped_images = 0
        for raw in raw_nodes:
            if not state.allow_node():
                remaining = len(raw_nodes) - len(nodes)
                state.note(truncation_note(remaining, "条转发消息"))
                break
            node, skipped = await self._convert_forward_node(
                raw,
                state=state,
                depth=depth + 1,
                vision_enabled=vision_enabled,
                file_group_id=file_group_id,
            )
            nodes.append(node)
            skipped_images += skipped

        if skipped_images:
            state.note(image_limit_note(skipped_images))

        return ForwardContent(
            forward_id=forward_id,
            nodes=tuple(nodes),
            truncated_note="",
            ok=bool(nodes),
            note="" if nodes else FORWARD_EMPTY_TEXT,
        )

    async def _convert_forward_node(
        self,
        raw,
        *,
        state: _BudgetState,
        depth: int,
        vision_enabled: bool,
        file_group_id: int = 0,
    ) -> tuple[ForwardNode, int]:
        """把一个转发节点转成 ForwardNode（保留发送者身份与顺序）。

        兼容两种真实结构：
        - NapCat / OneBot v11 标准节点：{"type": "node", "data": {...}}
        - 部分实现直接给出 node 内容：{"user_id": ..., "message": [...], ...}
        """
        node = _as_mapping(raw)
        node_data = _as_mapping(node.get("data"))
        merged = {**node_data, **{k: v for k, v in node.items() if k != "data"}}

        sender = _as_mapping(merged.get("sender"))
        sender_id = _first_str(merged, "user_id", "uin", "sender_id")
        if not sender_id:
            sender_id = _first_str(sender, "user_id", "uin")
        sender_name = _first_str(merged, "nickname", "name", "sender_name")
        if not sender_name:
            sender_name = _first_str(sender, "card", "nickname")

        timestamp = _first_int(merged, "time", "timestamp")
        message_field = merged.get("message", merged.get("content"))
        segments = as_segments(message_field)

        # 转发节点自带所属群号：节点里的文件属于**原群**，换下载直链必须用这个
        # 群号（实测 NapCat 会给出 group_id；缺失时退回外层群号）。
        node_group_id = _first_int(merged, "group_id") or file_group_id

        items, _nested_reply = await self._convert_segments(
            segments,
            source=BUDGET_SOURCE_FORWARD,
            state=state,
            depth=depth,
            vision_enabled=vision_enabled,
            file_group_id=node_group_id,
        )
        # 节点内出现的图片数量（用于统计“因数量限制未加载”）
        skipped = sum(
            1
            for item in items
            if isinstance(item, ImageContent) and item.image is None
        )
        return (
            ForwardNode(
                index=state.nodes,
                sender_id=sender_id,
                sender_name=sender_name,
                timestamp=timestamp,
                items=tuple(items),
            ),
            skipped,
        )

    # ===== OneBot API =====

    async def _call_api(self, action: str, **params) -> dict | None:
        """调用 OneBot API（get_msg / get_forward_msg），失败一律降级返回 None。

        失败场景（全部 graceful degradation）：
        timeout / 网络断开 / NapCat 返回 retcode!=0 / 消息已删除 / 无权限 / 非 dict 响应。
        日志只记 action 与结构化类别，绝不记消息正文。
        """
        if self.bot is None:
            logger.info("[RESOLVER] {} 调用失败（category=no_bot）", action)
            return None
        call_api = getattr(self.bot, "call_api", None)
        if not callable(call_api):
            logger.info("[RESOLVER] {} 调用失败（category=no_call_api）", action)
            return None
        try:
            response = await asyncio.wait_for(
                call_api(action, **params), timeout=API_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.info("[RESOLVER] {} 超时（category=timeout）", action)
            return None
        except Exception as exc:
            retcode = getattr(exc, "retcode", None) or getattr(exc, "code", None)
            logger.info(
                "[RESOLVER] {} 调用失败（category=api_error retcode={}）",
                action,
                retcode if retcode is not None else "none",
            )
            return None
        payload = _as_mapping(response)
        if not payload:
            logger.info("[RESOLVER] {} 返回非预期结构（category=bad_response）", action)
            return None
        return payload

    # ===== 内部工具 =====

    def _consume_budget_notices(self, state: _BudgetState) -> str:
        """把预算提示去重后合并成一段可信程序说明（绝不含正文）。"""
        seen: list[str] = []
        for note in state.notices:
            if note and note not in seen:
                seen.append(note)
        return "\n".join(seen)


def _walk_items(items):
    """深度遍历 ContentItem 树（含合并转发节点），逐个产出 item。

    引用/转发里的文件、图片都是嵌套在 ForwardContent.nodes[].items 里的，
    只看顶层会漏掉它们。
    """
    for item in items:
        yield item
        if isinstance(item, ForwardContent):
            for node in item.nodes:
                yield from _walk_items(node.items)


def _event_plaintext(event) -> str:
    """安全读取事件的纯文本（get_plaintext 可能不存在或抛异常）。"""
    getter = getattr(event, "get_plaintext", None)
    if not callable(getter):
        return ""
    try:
        return str(getter() or "").strip()
    except Exception:
        return ""


def _sender_name(event) -> str:
    """显示名（群名片 → 昵称 → QQ 号），与 prompt_builder 的语义保持一致。"""
    sender = getattr(event, "sender", None)
    if sender is not None:
        card = _as_str(getattr(sender, "card", ""))
        if card.strip():
            return card.strip()
        nickname = _as_str(getattr(sender, "nickname", ""))
        if nickname.strip():
            return nickname.strip()
    return str(_as_int(getattr(event, "user_id", 0)))


def is_known_item_type(item: ContentItem) -> bool:
    """ContentItem 类型是否在白名单内（日志 / 测试用）。"""
    return getattr(item, "type", "") in CONTENT_ITEM_TYPES
