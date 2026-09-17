"""统一内容预算与硬限制（v0.7）。

两件事，放在同一个文件里是因为它们必须一起被审视：

1. **硬限制配置**（REPLY_* / FORWARD_* / FILE_* / *_MAX_*）：进程启动时解析一次，
   非法值安全回落默认值。任何感知层模块都不允许自己再读一份环境变量。
2. **ContentBudget**：一次请求内的统一“外部内容预算”。File / Forward 极易
   把 Context 撑爆，因此**必须**在感知阶段就按优先级削减，而不是等到 Prompt。

优先级（高 → 低，与需求 31 一致）：
    1. 当前用户输入（永不裁剪）
    2. Reply 被回复内容
    3. Forward 内容
    4. File 正文
    5. Conversation History（由 context_serializer 的既有 CONTEXT_MAX_CHARS 负责）

设计原则：
- 预算只做“削减 + 记录被削减了多少”，绝不抛异常、绝不中断聊天流程；
- 每个维度的超限都产生一句**结构化的降级说明**（SystemNotice），
  让模型知道“内容存在但没展开”，而不是以为自己看全了；
- 所有截断都发生在这里，各 parser 不再各写一套 truncate。
"""

from dataclasses import dataclass
from dataclasses import field

from services.perception.env_config import env_int
# ===== Reply（被回复消息）=====
REPLY_MAX_DEPTH_DEFAULT = 2

# ===== Forward（合并转发）=====
FORWARD_MAX_DEPTH_DEFAULT = 2
FORWARD_MAX_NODES_DEFAULT = 50
FORWARD_MAX_IMAGES_DEFAULT = 8
FORWARD_MAX_FILES_DEFAULT = 5
FORWARD_MAX_TEXT_CHARS_DEFAULT = 30000

# ===== File（文件读取）=====
FILE_MAX_BYTES_DEFAULT = 20 * 1024 * 1024  # 20MB
FILE_MAX_TEXT_CHARS_DEFAULT = 30000
PDF_MAX_PAGES_DEFAULT = 30
DOCX_MAX_PARAGRAPHS_DEFAULT = 2000
XLSX_MAX_SHEETS_DEFAULT = 5
XLSX_MAX_ROWS_PER_SHEET_DEFAULT = 500
XLSX_MAX_COLS_DEFAULT = 50
PPTX_MAX_SLIDES_DEFAULT = 50

# ===== 外部内容总量（一次请求）=====
MAX_EXTERNAL_TEXT_CHARS_DEFAULT = 60000
MAX_TOTAL_IMAGES_DEFAULT = 8
MAX_TOTAL_FILES_DEFAULT = 5

# ===== 网络（下载统一超时）=====
DOWNLOAD_CONNECT_TIMEOUT_DEFAULT = 10
DOWNLOAD_READ_TIMEOUT_DEFAULT = 20


def _int(name: str, default: int, low: int, high: int) -> int:
    return env_int(name, default, low, high, "PERCEPTION")


# 进程启动时解析一次（改 .env 需重启生效）
REPLY_MAX_DEPTH = _int("REPLY_MAX_DEPTH", REPLY_MAX_DEPTH_DEFAULT, 0, 5)

FORWARD_MAX_DEPTH = _int("FORWARD_MAX_DEPTH", FORWARD_MAX_DEPTH_DEFAULT, 0, 5)
FORWARD_MAX_NODES = _int("FORWARD_MAX_NODES", FORWARD_MAX_NODES_DEFAULT, 1, 500)
FORWARD_MAX_IMAGES = _int("FORWARD_MAX_IMAGES", FORWARD_MAX_IMAGES_DEFAULT, 0, 50)
FORWARD_MAX_FILES = _int("FORWARD_MAX_FILES", FORWARD_MAX_FILES_DEFAULT, 0, 50)
FORWARD_MAX_TEXT_CHARS = _int(
    "FORWARD_MAX_TEXT_CHARS", FORWARD_MAX_TEXT_CHARS_DEFAULT, 500, 200000
)

FILE_MAX_BYTES = _int("FILE_MAX_BYTES", FILE_MAX_BYTES_DEFAULT, 1024, 200 * 1024 * 1024)
FILE_MAX_TEXT_CHARS = _int(
    "FILE_MAX_TEXT_CHARS", FILE_MAX_TEXT_CHARS_DEFAULT, 500, 200000
)
PDF_MAX_PAGES = _int("PDF_MAX_PAGES", PDF_MAX_PAGES_DEFAULT, 1, 500)
DOCX_MAX_PARAGRAPHS = _int("DOCX_MAX_PARAGRAPHS", DOCX_MAX_PARAGRAPHS_DEFAULT, 1, 20000)
XLSX_MAX_SHEETS = _int("XLSX_MAX_SHEETS", XLSX_MAX_SHEETS_DEFAULT, 1, 50)
XLSX_MAX_ROWS_PER_SHEET = _int(
    "XLSX_MAX_ROWS_PER_SHEET", XLSX_MAX_ROWS_PER_SHEET_DEFAULT, 1, 10000
)
XLSX_MAX_COLS = _int("XLSX_MAX_COLS", XLSX_MAX_COLS_DEFAULT, 1, 500)
PPTX_MAX_SLIDES = _int("PPTX_MAX_SLIDES", PPTX_MAX_SLIDES_DEFAULT, 1, 500)

MAX_EXTERNAL_TEXT_CHARS = _int(
    "MAX_EXTERNAL_TEXT_CHARS", MAX_EXTERNAL_TEXT_CHARS_DEFAULT, 1000, 400000
)
MAX_TOTAL_IMAGES = _int("MAX_TOTAL_IMAGES", MAX_TOTAL_IMAGES_DEFAULT, 1, 50)
MAX_TOTAL_FILES = _int("MAX_TOTAL_FILES", MAX_TOTAL_FILES_DEFAULT, 0, 50)

DOWNLOAD_CONNECT_TIMEOUT = _int(
    "DOWNLOAD_CONNECT_TIMEOUT", DOWNLOAD_CONNECT_TIMEOUT_DEFAULT, 1, 120
)
DOWNLOAD_READ_TIMEOUT = _int(
    "DOWNLOAD_READ_TIMEOUT", DOWNLOAD_READ_TIMEOUT_DEFAULT, 1, 300
)

# ===== 稳定的降级占位文案（程序事实，不是人格回复）=====
REPLY_UNAVAILABLE_TEXT = "[引用消息无法读取]"
FORWARD_UNAVAILABLE_TEXT = "[用户发送了一条合并转发，但内容获取失败]"
FILE_UNAVAILABLE_TEXT = "[用户发送了文件 {file_name}，但读取失败]"
IMAGE_UNAVAILABLE_TEXT = "[用户发送了一张图片，但图片读取失败]"
UNSUPPORTED_FILE_TEXT = "暂不支持读取该文件类型。"

# 预算来源标签（从高到低），日志只记标签与数量，绝不记正文。
BUDGET_SOURCE_CURRENT = "current"
BUDGET_SOURCE_REPLY = "reply"
BUDGET_SOURCE_FORWARD = "forward"
BUDGET_SOURCE_FILE = "file"


@dataclass
class BudgetUsage:
    """一次请求的预算消耗统计（只用于日志与测试，不含任何正文）。"""

    text_chars: dict[str, int] = field(default_factory=dict)
    images: dict[str, int] = field(default_factory=dict)
    files: dict[str, int] = field(default_factory=dict)
    truncated: list[str] = field(default_factory=list)

    def add_text(self, source: str, chars: int) -> None:
        self.text_chars[source] = self.text_chars.get(source, 0) + max(0, chars)

    def add_image(self, source: str, count: int = 1) -> None:
        self.images[source] = self.images.get(source, 0) + max(0, count)

    def add_file(self, source: str, count: int = 1) -> None:
        self.files[source] = self.files.get(source, 0) + max(0, count)

    def note_truncated(self, note: str) -> None:
        if note:
            self.truncated.append(note)

    def total_text_chars(self) -> int:
        return sum(self.text_chars.values())

    def total_images(self) -> int:
        return sum(self.images.values())

    def total_files(self) -> int:
        return sum(self.files.values())


class ContentBudget:
    """一次请求的统一内容预算（感知层所有模块共用同一个实例）。

    用法：
        budget = ContentBudget()
        budget.allow_image(BUDGET_SOURCE_FORWARD)   # False = 该图不加载
        text = budget.clip_text(text, BUDGET_SOURCE_FILE)

    语义：
    - `allow_image` / `allow_file` 先看“全局上限”再看“来源上限”，
      任一超限都返回 False，并由调用方补一句降级说明；
    - `clip_text` 返回按剩余额度截断后的文本（截断会追加“…（已截断）”），
      完全没额度时返回空字符串；
    - 预算绝不抛异常：任何异常语义词（越界、负数）都按 0 处理。
    """

    def __init__(
        self,
        max_external_text_chars: int | None = None,
        max_total_images: int | None = None,
        max_total_files: int | None = None,
    ) -> None:
        self.max_external_text_chars = (
            MAX_EXTERNAL_TEXT_CHARS
            if max_external_text_chars is None
            else max(0, max_external_text_chars)
        )
        self.max_total_images = (
            MAX_TOTAL_IMAGES if max_total_images is None else max(0, max_total_images)
        )
        self.max_total_files = (
            MAX_TOTAL_FILES if max_total_files is None else max(0, max_total_files)
        )
        self.usage = BudgetUsage()

    # ===== 文本 =====

    def remaining_text_chars(self) -> int:
        return max(0, self.max_external_text_chars - self.usage.total_text_chars())

    def clip_text(self, text: str, source: str = BUDGET_SOURCE_FILE) -> str:
        """按剩余总预算截断文本（超限补“…（已截断）”），并记账。"""
        text = text or ""
        if not text:
            return ""
        remaining = self.remaining_text_chars()
        if remaining <= 0:
            self.usage.note_truncated(f"{source}:no_text_budget")
            return ""
        if len(text) <= remaining:
            self.usage.add_text(source, len(text))
            return text
        clipped = text[:remaining] + "…（已截断）"
        self.usage.add_text(source, len(clipped))
        self.usage.note_truncated(f"{source}:text_clipped")
        return clipped

    # ===== 图片 =====

    def allow_image(self, source: str = BUDGET_SOURCE_FORWARD) -> bool:
        if self.usage.total_images() >= self.max_total_images:
            return False
        self.usage.add_image(source)
        return True

    def remaining_images(self) -> int:
        return max(0, self.max_total_images - self.usage.total_images())

    # ===== 文件 =====

    def allow_file(self, source: str = BUDGET_SOURCE_FORWARD) -> bool:
        if self.usage.total_files() >= self.max_total_files:
            return False
        self.usage.add_file(source)
        return True

    def remaining_files(self) -> int:
        return max(0, self.max_total_files - self.usage.total_files())


def truncation_note(remaining: int, unit: str) -> str:
    """生成统一的“还有 N 项因限制未展开”说明（number 只含数量，不含内容）。"""
    if remaining <= 0:
        return ""
    return f"[后续 {remaining} {unit}因上下文限制未展开]"


def image_limit_note(remaining: int) -> str:
    """合并转发里图片数量超限时的说明（与需求 32 的措辞一致）。"""
    if remaining <= 0:
        return ""
    return f"[该合并转发后续还有 {remaining} 张图片，因图片数量限制未加载]"
