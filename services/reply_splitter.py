"""回复拆分（v0.2.3）：把长回答按自然段拆成多条 QQ 消息，防刷屏。

规则：
1. 按“空行分隔的自然段”拆分；连续空行视为一个分隔；
2. Markdown fenced code block ```...``` 整体作为一个单元，绝不从代码块中间拆；
3. 单段超过 MAX_CHARS 时按句末标点（。！？!?；;）安全切分，实在不行再硬切；
4. 超过 MAX_PARTS 后，剩余内容合并进最后一条（禁止无限刷屏）；
5. 空段忽略；全文只有一段时仍返回一条；
6. SPLIT_REPLY_ENABLED=false 时调用方保持原行为（整条发送）。
"""

import os
import re

from nonebot import logger

SPLIT_REPLY_MAX_PARTS_DEFAULT = 6
SPLIT_REPLY_MAX_PARTS_MIN = 1
SPLIT_REPLY_MAX_PARTS_MAX = 20

SPLIT_REPLY_MAX_CHARS_DEFAULT = 1000
SPLIT_REPLY_MAX_CHARS_MIN = 100
SPLIT_REPLY_MAX_CHARS_MAX = 4000

SPLIT_REPLY_DELAY_MS_DEFAULT = 250
SPLIT_REPLY_DELAY_MS_MIN = 0
SPLIT_REPLY_DELAY_MS_MAX = 3000


def is_split_enabled() -> bool:
    raw = (os.getenv("SPLIT_REPLY_ENABLED") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _parse_int_env(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[SPLIT] {}={} 非法，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[SPLIT] {}={} 超出范围，使用默认 {}", name, value, default)
        return default
    return value


# 进程启动时解析一次（改 .env 需重启生效）
SPLIT_REPLY_ENABLED = is_split_enabled()
SPLIT_REPLY_MAX_PARTS = _parse_int_env(
    "SPLIT_REPLY_MAX_PARTS", SPLIT_REPLY_MAX_PARTS_DEFAULT, SPLIT_REPLY_MAX_PARTS_MIN, SPLIT_REPLY_MAX_PARTS_MAX
)
SPLIT_REPLY_MAX_CHARS = _parse_int_env(
    "SPLIT_REPLY_MAX_CHARS", SPLIT_REPLY_MAX_CHARS_DEFAULT, SPLIT_REPLY_MAX_CHARS_MIN, SPLIT_REPLY_MAX_CHARS_MAX
)
SPLIT_REPLY_DELAY_MS = _parse_int_env(
    "SPLIT_REPLY_DELAY_MS", SPLIT_REPLY_DELAY_MS_DEFAULT, SPLIT_REPLY_DELAY_MS_MIN, SPLIT_REPLY_DELAY_MS_MAX
)

# 段落切分点（句末标点后）
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


def _split_paragraphs(text: str) -> list[str]:
    """按空行拆自然段；fenced code block 整体保留为一个单元。"""
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False

    for line in text.splitlines():
        stripped = line.strip()
        # 进入/退出围栏
        if stripped.startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if not stripped and not in_fence:
            # 代码块外的空行 = 分段
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append("\n".join(current).strip())

    return [p for p in paragraphs if p]


def _chunk_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """超长段落按句末标点安全切分；单句仍超长时硬切。"""
    if len(paragraph) <= max_chars:
        return [paragraph]

    chunks: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        candidate = sentence if not buffer else buffer + sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer.strip())
            buffer = ""
        if len(sentence) <= max_chars:
            buffer = sentence
        else:
            # 单句超长：硬切
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars].strip())
                sentence = sentence[max_chars:]
            buffer = sentence
    if buffer.strip():
        chunks.append(buffer.strip())
    return [c for c in chunks if c]


def split_reply(
    answer: str,
    max_parts: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """把回答拆成若干条 QQ 消息；返回至少一条。"""
    if max_parts is None:
        max_parts = SPLIT_REPLY_MAX_PARTS
    if max_chars is None:
        max_chars = SPLIT_REPLY_MAX_CHARS

    parts: list[str] = []
    for paragraph in _split_paragraphs(answer):
        parts.extend(_chunk_long_paragraph(paragraph, max_chars))
    if not parts:
        parts = [answer.strip() or ""]

    if len(parts) > max_parts:
        # 防刷屏：前 max_parts-1 条 + 剩余全部合并进最后一条
        parts = parts[: max_parts - 1] + ["\n\n".join(parts[max_parts - 1 :])]

    return [p for p in parts if p] or [""]
