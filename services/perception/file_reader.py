"""File Reader（v0.7）：QQ file → 安全获取 → 类型判断 → 内容提取 → 结构化结果。

职责边界（**Perception ≠ Persona**）：
- 本模块只回答“这个文件里写了什么”；
- 不回答用户、不做人格化文案、不写模板回复（禁止 `if is_pdf: return "让我看看~"`）；
- 不知道 Persona / Memory / Relationship / Prompt 的存在（不 import 它们）。

流程：

    FileRef（OneBot file segment，来自消息或合并转发节点）
      ↓ 下载：流式 + connect/read timeout + Content-Length + 累计字节双重上限
      ↓ 类型：扩展名 + magic bytes / OOXML 内部结构 双重判断（不一致直接拒绝）
      ↓ 解析：asyncio.to_thread(只读 parser)，PDF 只取文字层（不做 OCR）
      ↓ 预算：文本按 ContentBudget 统一截断
    FileReadResult（结构化，绝不落库、绝不落盘残留）

安全红线（需求 16~20）：
- 绝不 eval / exec / os.system / shell=True / subprocess 执行用户文件；
- 代码文件（.py/.cpp/.sh/.js）只被当作**字符串读取**，永不运行；
- **绝不自动解压**任何压缩包（zip/rar/7z/tar/gz）；
- 禁止类型：exe / dll / apk / so / bat / cmd / ps1 / scr / com；
- 临时文件写在 tempfile 安全目录，用固定的 `payload.<ext>` 名字，
  **绝不使用 QQ 原始 filename 作为本地路径**；处理完立即删除；
- 日志只记 sanitized 文件名 / 字节数 / parser / 成功失败 / 耗时，
  绝不记 URL、CDN token、Base64、文件正文。
"""

import asyncio
import time
from dataclasses import dataclass

from nonebot import logger

from services.perception import file_types
from services.perception import parsers
from services.perception.content import FileRef
from services.perception.limits import FILE_MAX_BYTES
from services.perception.limits import FILE_MAX_TEXT_CHARS
from services.perception.limits import ContentBudget
from services.perception.limits import FILE_UNAVAILABLE_TEXT
from services.perception.net import cleanup_download
from services.perception.net import download_to_temp_file

# 只用于日志的 parser 白名单校验（结构化日志，绝不写正文）
_LOG_SAFE_NAME_MAX_CHARS = 64


@dataclass(frozen=True)
class FileReadResult:
    """一次文件读取的结构化结果（感知事实，不是回答）。"""

    ok: bool
    file_name: str = ""          # sanitized 显示名（可能被截断长度，不含控制字符）
    parser_type: str = file_types.PARSER_UNSUPPORTED
    text: str = ""               # 提取到的正文（已按预算截断）
    local_path: str = ""         # 图片类文件：临时文件路径（调用方负责清理）
    local_dir: str = ""          # 临时目录（调用方负责清理）
    size: int = 0
    note: str = ""               # 失败 / 截断说明（结构化，不含正文）
    latency_ms: int = 0
    keep_temp: bool = False      # True = 临时文件仍被结果引用，调用方负责 cleanup_result()


def sanitize_file_name(file_name: str, max_chars: int = _LOG_SAFE_NAME_MAX_CHARS) -> str:
    """把不可信文件名清洗成可安全显示 / 可安全写日志的形式。

    - 只取 basename（去掉任何目录成分，杜绝路径穿越 / 绝对路径）；
    - 去掉控制字符与路径分隔符；
    - 超长截断（含省略号）。
    """
    raw = str(file_name or "")
    # 统一分隔符后只取最后一段：`../../etc/passwd` → `passwd`
    base = raw.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    cleaned = cleaned.strip().strip(".")
    if not cleaned:
        return "未命名文件"
    if len(cleaned) > max_chars:
        return cleaned[: max(0, max_chars - 1)] + "…"
    return cleaned


def _fallback_text(file_name: str) -> str:
    """读取失败时给模型的**结构化事实占位**（不是人格回复）。"""
    return FILE_UNAVAILABLE_TEXT.format(file_name=sanitize_file_name(file_name))


async def read_file(
    file_ref: FileRef,
    budget: ContentBudget | None = None,
    max_bytes: int | None = None,
    max_text_chars: int | None = None,
) -> FileReadResult:
    """完整读取一个 QQ 文件（下载 → 类型校验 → 解析 → 预算截断）。

    任何失败都返回 ok=False + note，**绝不抛异常**：调用方据此生成
    “[用户发送了文件 xxx，但读取失败]” 之类的结构化占位，聊天流程继续。
    """
    started = time.monotonic()
    display_name = sanitize_file_name(file_ref.file_name)
    limit_bytes = FILE_MAX_BYTES if max_bytes is None else max(1024, int(max_bytes))
    limit_chars = FILE_MAX_TEXT_CHARS if max_text_chars is None else max(1, int(max_text_chars))

    if not file_ref.url:
        # OneBot / NapCat 没给下载地址（例如转发节点里的 file 段）：
        # 诚实降级，绝不伪造成“已读取”。
        logger.info(
            "[FILE] 无下载地址（source={} parser=none）name={}",
            file_ref.source,
            display_name,
        )
        return FileReadResult(
            ok=False,
            file_name=display_name,
            note=_fallback_text(display_name),
            latency_ms=_elapsed_ms(started),
        )

    download = await download_to_temp_file(file_ref.url, limit_bytes, file_ref.file_name)
    if not download.ok:
        logger.info(
            "[FILE] 下载失败（category={}）name={} source={}",
            download.note,
            display_name,
            file_ref.source,
        )
        return FileReadResult(
            ok=False,
            file_name=display_name,
            note=_fallback_text(display_name),
            latency_ms=_elapsed_ms(started),
        )

    try:
        result = await _read_local_file(
            download.path,
            download.dir_path,
            file_ref,
            display_name,
            download.size,
            limit_chars,
            budget,
            started,
        )
    except Exception as exc:  # 兜底：任何解析异常都不能让聊天失败
        logger.error("[FILE] 读取异常（category=unexpected）：{}", type(exc).__name__)
        cleanup_download(download.dir_path)
        return FileReadResult(
            ok=False,
            file_name=display_name,
            size=download.size,
            note=_fallback_text(display_name),
            latency_ms=_elapsed_ms(started),
        )
    if not result.keep_temp:
        # 非图片路径：正文已经在内存里，立刻删除临时文件（幂等）。
        cleanup_download(download.dir_path)
    return result


async def _read_local_file(
    path: str,
    dir_path: str,
    file_ref: FileRef,
    display_name: str,
    size: int,
    limit_chars: int,
    budget: ContentBudget | None,
    started: float,
) -> FileReadResult:
    """对已下载到临时目录的文件做类型校验 + 解析。"""
    info = file_types.classify(file_ref.file_name, path)

    if info.verdict == file_types.VERDICT_MISMATCH:
        # 需求 18：类型不一致必须拒绝解析，只记结构化类别，绝不记内容。
        logger.warning(
            "[FILE] file_type_mismatch name={} ext={} magic={}",
            display_name,
            info.extension or "none",
            info.magic or "unknown",
        )
        cleanup_download(dir_path)
        return FileReadResult(
            ok=False,
            file_name=display_name,
            parser_type=file_types.PARSER_UNSUPPORTED,
            size=size,
            note=file_types.MISMATCH_NOTE,
            latency_ms=_elapsed_ms(started),
        )

    if not info.allowed:
        cleanup_download(dir_path)
        return FileReadResult(
            ok=False,
            file_name=display_name,
            parser_type=file_types.PARSER_UNSUPPORTED,
            size=size,
            note=info.note or file_types.UNKNOWN_NOTE,
            latency_ms=_elapsed_ms(started),
        )

    # 图片类文件：不解析，交给上层复用 Vision Pipeline（保留临时文件，
    # 由 resolver 转成 data URL 后统一清理）。
    if info.parser == file_types.PARSER_IMAGE:
        return FileReadResult(
            ok=True,
            file_name=display_name,
            parser_type=file_types.PARSER_IMAGE,
            local_path=path,
            local_dir=dir_path,
            size=size,
            latency_ms=_elapsed_ms(started),
            keep_temp=True,
        )

    # 同步 CPU / IO 解析放到线程池，绝不阻塞 NoneBot 事件循环。
    result = await asyncio.to_thread(parsers.parse_by_type, info.parser, path, limit_chars)
    cleanup_download(dir_path)

    if not result.ok:
        logger.info(
            "[FILE] 解析失败（parser={}）name={} size={}",
            info.parser,
            display_name,
            size,
        )
        return FileReadResult(
            ok=False,
            file_name=display_name,
            parser_type=info.parser,
            size=size,
            note=result.note or parsers.PARSE_FAILED_NOTE,
            latency_ms=_elapsed_ms(started),
        )

    text = result.text
    if text and budget is not None:
        text = budget.clip_text(text, "file")

    logger.info(
        "[FILE] 解析成功 parser={} name={} size={} text_chars={} truncated={} latency_ms={}",
        info.parser,
        display_name,
        size,
        len(text),
        result.truncated,
        _elapsed_ms(started),
    )
    return FileReadResult(
        ok=True,
        file_name=display_name,
        parser_type=info.parser,
        text=text,
        size=size,
        # 解析成功但正文为空时（例如扫描版 PDF / 空表格）保留解析器的结构化说明，
        # 让模型知道“文件读到了，但没有可提取文本”，而不是以为程序出错。
        note=result.note if not text else "",
        latency_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def cleanup_result(result: FileReadResult) -> None:
    """清理 FileReadResult 持有的临时目录（图片类文件用）。"""
    if result.local_dir:
        cleanup_download(result.local_dir)


def is_supported_extension(file_name: str) -> bool:
    """扩展名是否在允许解析的白名单内（不看内容，供上层快速预判）。"""
    return file_types.extension_of(file_name) in file_types.ALLOWED_EXTENSIONS
