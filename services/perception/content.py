"""NormalizedMessage 数据模型（v0.7）：QQ 输入的中间表示（IR）。

核心不变量（必须遵守，否则 Forward / Reply 语义会丢）：

1. **原始顺序必须保留**。QQ 消息「文字A 图片1 文字B 图片2」解析后仍然是
   TextContent(A) / ImageContent(1) / TextContent(B) / ImageContent(2)；
   绝不合并成「文字A+文字B」再附「图片1+图片2」。
   multimodal_builder 依赖这个顺序生成真正有序的 LLM content blocks。
2. **Reply 与当前消息分离**。被回复的历史消息放在 `NormalizedMessage.reply`，
   不平铺进 `items`；模型必须能区分“当前用户说的话”和“用户引用的历史消息”。
3. **Forward 保留节点身份**。每个转发节点的 sender_id / sender_name / timestamp /
   节点序号都保留，绝不只是把正文拼成一段字符串。
4. **感知层不产生自然语言回复**。这里只有事实描述（“用户发送了图片”一类占位），
   绝不出现“让我看看这个 PDF~”这种人格化文案。

数据模型不持有 QQ CDN token / Base64 之外的任何持久化职责：
这些字段只存在于内存，绝不写入 SQLite（落库走 build_context_text 的占位符）。
"""

from dataclasses import dataclass
from dataclasses import field

from services.vision import VisionImage

# ContentItem.type 的合法取值（同时供测试与日志做白名单校验）。
CONTENT_ITEM_TYPES = (
    "text",
    "image",
    "file",
    "forward",
    "system_notice",
)

# 转发节点序号从 1 开始（Prompt 里显示“节点 1”）。
FORWARD_NODE_START_INDEX = 1


@dataclass(frozen=True)
class ContentItem:
    """归一化内容项基类（保持消息内原始顺序）。

    type 是类型标签；子类各自携带自己的字段。用 isinstance 分发而不是
    单个大 dataclass 带一堆 Optional 字段，保证 builder / 测试都明确知道
    每种类型真实携带了什么。
    """

    type: str = ""


# ===== 文本 =====


@dataclass(frozen=True)
class TextContent(ContentItem):
    """一段用户文本（已去掉 @ 本体 / CQ Code）。"""

    text: str = ""
    type: str = "text"


# ===== 图片 =====


@dataclass(frozen=True)
class ImageContent(ContentItem):
    """一张图片。

    image 为空（None）时表示“程序知道这里有一张图，但没有可用内容”
    （Vision 关闭 / 无 URL / 超限被拒）——builder 会用 placeholder 说明，
    模型不会误以为图片已被看见。
    """

    image: VisionImage | None = None
    placeholder: str = ""
    type: str = "image"


# ===== 文件 =====


@dataclass(frozen=True)
class FileRef:
    """QQ 文件段的原始引用（不可信数据，绝不落库）。"""

    file_name: str = ""
    url: str = ""
    file_id: str = ""
    file_size: int = 0
    source: str = "message"  # message | forward


@dataclass(frozen=True)
class FileContent(ContentItem):
    """一个文件的感知结果（正文 / 图片 / 失败说明）。"""

    file_ref: FileRef = field(default_factory=FileRef)
    file_name: str = ""
    parser_type: str = ""       # text | pdf | docx | xlsx | pptx | image | unsupported
    text: str = ""              # 提取到的正文（已按预算截断）
    image: VisionImage | None = None  # 图片类文件复用 Vision Pipeline
    ok: bool = False
    note: str = ""              # 失败原因 / 截断说明（只含结构化信息，不含正文）
    type: str = "file"


# ===== 合并转发 =====


@dataclass(frozen=True)
class ForwardNode:
    """合并转发中的一个节点（保留发送者身份与顺序）。"""

    index: int
    sender_id: str = ""
    sender_name: str = ""
    timestamp: int = 0
    items: tuple[ContentItem, ...] = ()


@dataclass(frozen=True)
class ForwardContent(ContentItem):
    """一次合并转发的完整解析结果（含嵌套）。"""

    forward_id: str = ""
    nodes: tuple[ForwardNode, ...] = ()
    truncated_note: str = ""
    ok: bool = True
    note: str = ""
    type: str = "forward"


# ===== 系统说明（占位 / 降级） =====


@dataclass(frozen=True)
class SystemNotice(ContentItem):
    """程序生成的事实说明（不是人格回复）。

    用途：能力关闭 / 读取失败 / 上下文截断等需要告知模型的程序事实，
    例如「[该合并转发后续还有 63 张图片，因图片数量限制未加载]」。
    """

    text: str = ""
    type: str = "system_notice"


# ===== 顶层结构 =====


@dataclass(frozen=True)
class MessageMetadata:
    """程序生成的可信 metadata（不是用户文本）。

    group_id / user_id / message_id / self_id 由事件（程序）提供，
    聊天文本无法修改；sender_name 是用户可改的显示名（不可信显示文本，
    只作数据，不能作为身份判定依据）。
    """

    group_id: int = 0
    user_id: int = 0
    self_id: int = 0
    message_id: int = 0
    sender_name: str = ""
    message_type: str = "group"


@dataclass(frozen=True)
class NormalizedMessage:
    """一条 QQ 消息的完整归一化结果。

    items 只包含“当前这条消息”的内容（保持顺序）；
    被回复的历史消息单独放在 reply（也是一个 NormalizedMessage）。
    """

    items: tuple[ContentItem, ...] = ()
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    reply: "NormalizedMessage | None" = None

    # ===== 便捷统计（builder / 落库占位符 / 日志都用这些，避免各处重复遍历） =====

    def iter_items(self, item_type: str | None = None):
        """按顺序遍历 items；item_type 非空时只产出该类型。"""
        for item in self.items:
            if item_type is None or item.type == item_type:
                yield item

    def text(self, sep: str = "\n") -> str:
        """当前消息里的纯文本（按原始顺序拼接），不含 reply。"""
        return sep.join(
            item.text for item in self.items if isinstance(item, TextContent) and item.text
        ).strip()

    def image_count(self) -> int:
        """当前消息里的图片总数（含被拒绝 / 占位的图片）。"""
        return sum(1 for _ in self.iter_items("image"))

    def usable_image_count(self) -> int:
        """当前消息里真正可以交给模型的图片数。"""
        return sum(
            1
            for item in self.iter_items("image")
            if isinstance(item, ImageContent) and item.image is not None
        )

    def file_count(self) -> int:
        return sum(1 for _ in self.iter_items("file"))

    def forward_count(self) -> int:
        return sum(1 for _ in self.iter_items("forward"))

    def is_empty(self) -> bool:
        """既没有文本、也没有任何多媒体内容（只有 @ 本体）。"""
        return not self.items

    def has_any_content(self) -> bool:
        """有没有任何“用户实际发了东西”的迹象（文本 / 图片 / 文件 / 转发）。"""
        return bool(self.items)
