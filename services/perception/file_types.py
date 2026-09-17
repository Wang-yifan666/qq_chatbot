"""文件类型识别（v0.7）：扩展名 + magic bytes / OOXML 内部结构 双重判断。

安全动机（需求 18）：**不能只相信文件名**。
`evil.exe` 改名成 `homework.pdf` 时不得交给 PDF parser；`a.zip` 改名成
`report.docx` 也不得启动任何解压逻辑。

Windows 部署友好：**不依赖 python-magic / libmagic**（在 Windows 上需要额外
native DLL，部署成本高）。这里用「已知魔数表 + OOXML(zip) 内部结构探测」实现
等价的保守判断：
- 魔数明确冲突 → `mismatch`（拒绝解析）；
- 魔数无法判定（纯文本代码 / yaml 等没有魔数）→ `ok`，交给只读文本 parser；
- zip 容器必须同时匹配 `[Content_Types].xml` 与对应目录才能认定为 docx/xlsx/pptx。
"""

import os
import zipfile
from dataclasses import dataclass

# ===== 分类：允许 / 拒绝 =====

TEXT_EXTENSIONS = frozenset({"txt", "md", "json", "yaml", "yml"})
CODE_EXTENSIONS = frozenset(
    {
        "py", "c", "cpp", "cc", "h", "hpp", "java", "js", "ts",
        "html", "css", "xml", "sql", "sh",
    }
)
DOCUMENT_EXTENSIONS = frozenset({"pdf", "docx", "xlsx", "pptx"})
IMAGE_FILE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})

ALLOWED_EXTENSIONS = (
    TEXT_EXTENSIONS | CODE_EXTENSIONS | DOCUMENT_EXTENSIONS | IMAGE_FILE_EXTENSIONS
)

# 明确禁止（可执行 / 脚本 / 压缩包）：不解析、不解压、不执行
FORBIDDEN_EXTENSIONS = frozenset(
    {
        "exe", "dll", "apk", "so", "bat", "cmd", "ps1", "scr", "com", "msi",
        "jar", "vbs", "jsx", "bin", "dylib",
        "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "tgz", "iso", "cab",
    }
)

# 解析器类型（与 FileContent.parser_type 对应）
PARSER_TEXT = "text"
PARSER_PDF = "pdf"
PARSER_DOCX = "docx"
PARSER_XLSX = "xlsx"
PARSER_PPTX = "pptx"
PARSER_IMAGE = "image"
PARSER_UNSUPPORTED = "unsupported"

# 类型校验结论
VERDICT_OK = "ok"                  # 扩展名与内容一致（或无魔数可判的直接文本）
VERDICT_MISMATCH = "mismatch"      # 扩展名与魔数明确冲突 → 拒绝解析
VERDICT_FORBIDDEN = "forbidden"    # 明确禁止的类型（可执行 / 压缩包）
VERDICT_UNKNOWN = "unknown"        # 不在白名单内的未知类型

MISMATCH_NOTE = "文件扩展名与实际内容类型不一致，已拒绝解析。"
FORBIDDEN_NOTE = "暂不支持读取该文件类型。"
UNKNOWN_NOTE = "暂不支持读取该文件类型。"

# 可执行文件魔数（MZ = PE/EXE/DLL；ELF；Java class；shell 脚本 shebang）
_EXECUTABLE_MAGIC_PREFIXES = (
    b"MZ",
    b"\x7fELF",
    b"\xca\xfe\xba\xbe",
    b"#!",
)

# 最小可解析字节数：OOXML 至少需要一个 EOCD（22 字节）以上
MIN_BYTES_FOR_ARCHIVE = 22


@dataclass(frozen=True)
class FileTypeInfo:
    """一次文件类型判断的结果（只含结构化信息，绝不含文件正文）。"""

    extension: str = ""
    kind: str = ""              # text | code | document | image | forbidden | unknown
    parser: str = PARSER_UNSUPPORTED
    verdict: str = VERDICT_UNKNOWN
    magic: str = ""             # 仅用于日志的类别名（pdf / zip / elf / ...）
    note: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == VERDICT_OK and self.parser != PARSER_UNSUPPORTED


def extension_of(file_name: str) -> str:
    """取小写扩展名（不含点）；无法取得时返回空字符串。"""
    if not file_name:
        return ""
    base = os.path.basename(str(file_name).replace("\\", "/"))
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[-1].strip().lower()


def _kind_of(extension: str) -> str:
    if extension in TEXT_EXTENSIONS:
        return "text"
    if extension in CODE_EXTENSIONS:
        return "code"
    if extension in DOCUMENT_EXTENSIONS:
        return "document"
    if extension in IMAGE_FILE_EXTENSIONS:
        return "image"
    if extension in FORBIDDEN_EXTENSIONS:
        return "forbidden"
    return "unknown"


def _parser_of(extension: str) -> str:
    kind = _kind_of(extension)
    if kind in ("text", "code"):
        return PARSER_TEXT
    if kind == "image":
        return PARSER_IMAGE
    if kind == "document":
        return extension  # pdf / docx / xlsx / pptx
    return PARSER_UNSUPPORTED


def _zip_internals(path: str) -> set[str]:
    """只读 zip 中央目录（绝不解压），返回内部条目名集合；失败返回空集合。"""
    try:
        with zipfile.ZipFile(path) as archive:
            return {name.replace("\\", "/") for name in archive.namelist()}
    except Exception:
        return set()


def _looks_like_ooxml(internals: set[str], expected_dir: str) -> bool:
    return "[Content_Types].xml" in internals and any(
        name.startswith(expected_dir) for name in internals
    )


def sniff_content_kind(path: str) -> str:
    """按文件内容（魔数 / OOXML 结构）判断真实类别。

    返回：pdf | docx | xlsx | pptx | zip | executable | gzip | unknown...
    绝不读取整个文件，只读前 16 字节 + zip 中央目录。
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        return "unreadable"

    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(_EXECUTABLE_MAGIC_PREFIXES):
        return "executable"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        if os.path.getsize(path) < MIN_BYTES_FOR_ARCHIVE:
            return "zip"
        internals = _zip_internals(path)
        if _looks_like_ooxml(internals, "word/"):
            return "docx"
        if _looks_like_ooxml(internals, "xl/"):
            return "xlsx"
        if _looks_like_ooxml(internals, "ppt/"):
            return "pptx"
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "gif"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    return "unknown"


# 扩展名 → 期望的内容类别（多重集合）
_EXPECTED_CONTENT: dict[str, frozenset[str]] = {
    "pdf": frozenset({"pdf"}),
    "docx": frozenset({"docx"}),
    "xlsx": frozenset({"xlsx"}),
    "pptx": frozenset({"pptx"}),
    "jpg": frozenset({"jpeg"}),
    "jpeg": frozenset({"jpeg"}),
    "png": frozenset({"png"}),
    "webp": frozenset({"webp"}),
}

# 这些内容的类别名可以接受任意文本扩展名（纯文本没有魔数，内容判定为 unknown）
_TEXTUAL_CONTENT = frozenset({"unknown"})


def classify(file_name: str, path: str) -> FileTypeInfo:
    """扩展名 + 内容双重判断，返回 FileTypeInfo。

    判定顺序：
    1. 扩展名在禁止清单 → forbidden（不看内容，也不解压）；
    2. 扩展名不在白名单 → unknown；
    3. 读内容类别：
       - 文档 / 图片类扩展名：内容类别必须匹配，否则 mismatch；
       - 纯文本 / 代码类扩展名：内容类别必须是“无魔数文本”（unknown），
         若其实是可执行文件 / 压缩包 / PDF → mismatch（这就是 exe 改名 txt 的场景）；
    4. 文本类文件额外做一次“是否可信为文本”的粗检（含 NUL 字节 → mismatch）。
    """
    extension = extension_of(file_name)
    kind = _kind_of(extension)
    parser = _parser_of(extension)

    if kind == "forbidden":
        return FileTypeInfo(
            extension=extension,
            kind=kind,
            parser=PARSER_UNSUPPORTED,
            verdict=VERDICT_FORBIDDEN,
            magic="forbidden",
            note=FORBIDDEN_NOTE,
        )
    if kind == "unknown":
        return FileTypeInfo(
            extension=extension,
            kind=kind,
            parser=PARSER_UNSUPPORTED,
            verdict=VERDICT_UNKNOWN,
            magic="unknown",
            note=UNKNOWN_NOTE,
        )
    if parser == PARSER_UNSUPPORTED:
        return FileTypeInfo(
            extension=extension,
            kind=kind,
            parser=PARSER_UNSUPPORTED,
            verdict=VERDICT_UNKNOWN,
            magic="unknown",
            note=UNKNOWN_NOTE,
        )

    content_kind = sniff_content_kind(path)

    if kind in ("text", "code"):
        if content_kind in _TEXTUAL_CONTENT:
            if _has_nul_byte(path):
                return FileTypeInfo(
                    extension=extension,
                    kind=kind,
                    parser=parser,
                    verdict=VERDICT_MISMATCH,
                    magic="binary",
                    note=MISMATCH_NOTE,
                )
            return FileTypeInfo(
                extension=extension,
                kind=kind,
                parser=parser,
                verdict=VERDICT_OK,
                magic="text",
            )
        return FileTypeInfo(
            extension=extension,
            kind=kind,
            parser=parser,
            verdict=VERDICT_MISMATCH,
            magic=content_kind,
            note=MISMATCH_NOTE,
        )

    expected = _EXPECTED_CONTENT.get(extension, frozenset())
    if content_kind in expected:
        return FileTypeInfo(
            extension=extension,
            kind=kind,
            parser=parser,
            verdict=VERDICT_OK,
            magic=content_kind,
        )
    return FileTypeInfo(
        extension=extension,
        kind=kind,
        parser=parser,
        verdict=VERDICT_MISMATCH,
        magic=content_kind,
        note=MISMATCH_NOTE,
    )


def _has_nul_byte(path: str, probe_bytes: int = 8192) -> bool:
    """粗检“看起来是不是二进制”（前 8KB 出现 NUL 字节即认为不是文本）。

    这是保守判断：宁可拒绝一个奇怪文件，也不把二进制当文本塞进 Prompt。
    """
    try:
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(probe_bytes)
    except OSError:
        return True
