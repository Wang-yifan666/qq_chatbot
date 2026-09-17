"""受限网络下载（v0.7）：感知层唯一的 HTTP 下载入口。

为什么单独一个模块：
- 文件下载与图片格式转换都需要“流式下载 + 大小上限 + 超时”，
  复制两份 HTTP 客户端代码是明确的技术债；
- 安全要求集中在三处：**绝不无上限加载**（不用 response.content）、
  **绝不使用 QQ 原始文件名作为本地路径**、**处理完必须删除临时文件**。

安全要点：
- connect / read 超时分开配置（`DOWNLOAD_CONNECT_TIMEOUT` /
  `DOWNLOAD_READ_TIMEOUT`，见 services/perception/limits.py）；
- 先看 `Content-Length`（声明超限直接拒绝），再逐块累计真实字节数，
  超过上限立即中止并删除临时文件；
- 临时目录用 `tempfile.mkdtemp()`（安全随机目录），临时文件名是
  `payload.<sanitized ext>`，与用户提供的文件名无关；
- 日志绝不输出 URL / CDN token / 文件正文，只记字节数与失败类别。
"""

import os
import shutil
import tempfile
from dataclasses import dataclass

import httpx
from nonebot import logger

from services.perception.env_config import env_int
from services.perception.limits import DOWNLOAD_CONNECT_TIMEOUT
from services.perception.limits import DOWNLOAD_READ_TIMEOUT

# 单次下载最多允许的临时文件字节数（防止配置被改成 0 或负数时无上限）
_MIN_MAX_BYTES = 1024

# 流式下载的块大小（64KB）
CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True)
class DownloadResult:
    """写到临时文件的一次下载结果。

    ok=True 时 `path` 一定存在，调用方负责调用 cleanup_download()（或让
    TempFileDownload 上下文管理器回收）。
    """

    ok: bool
    path: str = ""
    dir_path: str = ""
    size: int = 0
    note: str = ""  # 结构化原因：declared_too_large | exceeded_limit | timeout | http_error | network_error


def sanitize_suffix(file_name: str, default: str = "") -> str:
    """从不可信文件名里只取一个安全的扩展名后缀（供临时文件使用）。

    - 只保留 `[A-Za-z0-9]`，最多 10 个字符；
    - 绝不返回路径分隔符、绝不返回用户提供的完整名字；
    - 取不到时返回 default（可为空字符串）。
    """
    extension = ""
    if file_name:
        base = os.path.basename(str(file_name).replace("\\", "/"))
        if "." in base:
            extension = base.rsplit(".", 1)[-1]
    cleaned = "".join(ch for ch in extension if ch.isalnum())[:10].lower()
    if not cleaned:
        return default
    return "." + cleaned


def make_temp_dir() -> str:
    """创建一次性安全临时目录（调用方负责删除）。"""
    return tempfile.mkdtemp(prefix="qqbot_perception_")


def cleanup_download(dir_path: str) -> None:
    """删除临时目录及其内容；失败只记 DEBUG，绝不影响聊天流程。"""
    if not dir_path:
        return
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - rmtree(ignore_errors) 已吞掉多数异常
        logger.debug("[PERCEPTION] 临时目录清理失败：{}", type(exc).__name__)


def _timeouts() -> httpx.Timeout:
    """统一的 connect / read / write / pool 超时。"""
    connect = env_int(
        "DOWNLOAD_CONNECT_TIMEOUT", DOWNLOAD_CONNECT_TIMEOUT, 1, 120, "PERCEPTION"
    )
    read = env_int("DOWNLOAD_READ_TIMEOUT", DOWNLOAD_READ_TIMEOUT, 1, 300, "PERCEPTION")
    return httpx.Timeout(connect=connect, read=read, write=connect, pool=connect)


async def download_to_temp_file(
    url: str,
    max_bytes: int,
    file_name: str = "",
) -> DownloadResult:
    """流式下载 URL 到临时文件，超过 max_bytes 立即中止。

    返回 DownloadResult.ok=False 时不会残留任何临时文件。
    """
    url = (url or "").strip()
    if not url:
        return DownloadResult(ok=False, note="empty_url")

    limit = max(_MIN_MAX_BYTES, int(max_bytes or 0))
    dir_path = make_temp_dir()
    suffix = sanitize_suffix(file_name)
    path = os.path.join(dir_path, f"payload{suffix}")
    declared = 0
    written = 0
    try:
        async with httpx.AsyncClient(timeout=_timeouts(), follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                raw_length = response.headers.get("Content-Length")
                if raw_length:
                    try:
                        declared = int(raw_length)
                    except (TypeError, ValueError):
                        declared = 0
                if declared and declared > limit:
                    cleanup_download(dir_path)
                    return DownloadResult(ok=False, note="declared_too_large")
                with open(path, "wb") as handle:
                    async for chunk in response.aiter_bytes(CHUNK_SIZE):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > limit:
                            handle.close()
                            cleanup_download(dir_path)
                            return DownloadResult(ok=False, note="exceeded_limit")
                        handle.write(chunk)
    except httpx.TimeoutException:
        cleanup_download(dir_path)
        return DownloadResult(ok=False, note="timeout")
    except httpx.HTTPStatusError:
        cleanup_download(dir_path)
        return DownloadResult(ok=False, note="http_error")
    except Exception as exc:
        cleanup_download(dir_path)
        logger.debug("[PERCEPTION] 下载失败（category=network_error）：{}", type(exc).__name__)
        return DownloadResult(ok=False, note="network_error")

    if written <= 0:
        cleanup_download(dir_path)
        return DownloadResult(ok=False, note="empty_body")
    return DownloadResult(ok=True, path=path, dir_path=dir_path, size=written)


async def download_to_memory(url: str, max_bytes: int) -> tuple[bytes | None, str]:
    """流式下载到内存（用于图片格式转换这类必须拿到 bytes 的场景）。

    返回 (bytes | None, note)；note 为结构化原因，绝不是异常原文。
    """
    url = (url or "").strip()
    if not url:
        return None, "empty_url"
    limit = max(_MIN_MAX_BYTES, int(max_bytes or 0))
    buffer = bytearray()
    try:
        async with httpx.AsyncClient(timeout=_timeouts(), follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                raw_length = response.headers.get("Content-Length")
                if raw_length:
                    try:
                        if int(raw_length) > limit:
                            return None, "declared_too_large"
                    except (TypeError, ValueError):
                        pass
                async for chunk in response.aiter_bytes(CHUNK_SIZE):
                    if not chunk:
                        continue
                    if len(buffer) + len(chunk) > limit:
                        return None, "exceeded_limit"
                    buffer.extend(chunk)
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPStatusError:
        return None, "http_error"
    except Exception as exc:
        logger.debug("[PERCEPTION] 下载失败（category=network_error）：{}", type(exc).__name__)
        return None, "network_error"
    if not buffer:
        return None, "empty_body"
    return bytes(buffer), ""
