"""联网搜索服务（v0.2.3）。

统一 web_search 接口：query -> [{"title", "url", "snippet"}]，避免整个项目
绑定单一服务商。后端可插拔（WEB_SEARCH_BACKEND）。

当前内置后端：duckduckgo（DuckDuckGo Instant Answer API，无需 Key，
异步 httpx 调用，绝不使用同步 requests 阻塞事件循环）。

搜索结果属于 UNTRUSTED EXTERNAL DATA：只作事实参考，由 Prompt 层
（TRUST_MODEL）与 Tool Orchestrator（白名单 + Schema 校验）双重防护。

配置（.env）：
- WEB_SEARCH_ENABLED=true/false（只有程序决定 capability，聊天无法修改）
- WEB_SEARCH_BACKEND=duckduckgo
- WEB_SEARCH_TIMEOUT=15（秒）
- WEB_SEARCH_MAX_RESULTS=5
"""

import html as _html
import os
import re

import httpx

from nonebot import logger

from services import log_message_content_enabled
from services import redact_secrets
from services import safe_log_text

WEB_SEARCH_TIMEOUT_DEFAULT = 15.0
WEB_SEARCH_TIMEOUT_MIN = 3.0
WEB_SEARCH_TIMEOUT_MAX = 60.0

WEB_SEARCH_MAX_RESULTS_DEFAULT = 5
WEB_SEARCH_MAX_RESULTS_MIN = 1
WEB_SEARCH_MAX_RESULTS_MAX = 10

SNIPPET_MAX_CHARS = 500
TITLE_MAX_CHARS = 120


def is_web_search_enabled() -> bool:
    """联网能力开关（只有程序读环境变量决定，聊天内容不能修改）。"""
    raw = (os.getenv("WEB_SEARCH_ENABLED") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _parse_float_env(name: str, default: float, low: float, high: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("[WEB SEARCH] {}={} 非法，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[WEB SEARCH] {}={} 超出范围，使用默认 {}", name, value, default)
        return default
    return value


def _parse_int_env(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[WEB SEARCH] {}={} 非法，使用默认 {}", name, raw, default)
        return default
    if not (low <= value <= high):
        logger.warning("[WEB SEARCH] {}={} 超出范围，使用默认 {}", name, value, default)
        return default
    return value


# 进程启动时解析一次（改 .env 需重启生效）
WEB_SEARCH_ENABLED = is_web_search_enabled()
WEB_SEARCH_BACKEND = (os.getenv("WEB_SEARCH_BACKEND") or "duckduckgo").strip().lower()
WEB_SEARCH_TIMEOUT = _parse_float_env(
    "WEB_SEARCH_TIMEOUT", WEB_SEARCH_TIMEOUT_DEFAULT, WEB_SEARCH_TIMEOUT_MIN, WEB_SEARCH_TIMEOUT_MAX
)
WEB_SEARCH_MAX_RESULTS = _parse_int_env(
    "WEB_SEARCH_MAX_RESULTS", WEB_SEARCH_MAX_RESULTS_DEFAULT, WEB_SEARCH_MAX_RESULTS_MIN, WEB_SEARCH_MAX_RESULTS_MAX
)

if WEB_SEARCH_ENABLED:
    logger.info("[WEB SEARCH] enabled backend={} timeout={}s max_results={}", WEB_SEARCH_BACKEND, WEB_SEARCH_TIMEOUT, WEB_SEARCH_MAX_RESULTS)
else:
    logger.info("[WEB SEARCH] disabled")


def _log_search_done(backend: str, query: str, result_count: int) -> None:
    """搜索完成日志：默认只记 query 长度（搜索 query 可能含敏感信息）；
    LOG_MESSAGE_CONTENT=true 时才记录清洗 + 截断后的 query。"""
    if log_message_content_enabled():
        logger.info(
            "[WEB SEARCH] backend={} query_chars={} query={} results={}",
            backend,
            len(query),
            safe_log_text(query, 80),
            result_count,
        )
    else:
        logger.info(
            "[WEB SEARCH] backend={} query_chars={} results={}",
            backend,
            len(query),
            result_count,
        )


async def search(query: str) -> list[dict[str, str]]:
    """执行联网搜索，返回 [{title, url, snippet}]。

    网络/解析错误向上抛出，由 Tool Orchestrator 捕获并降级为“搜索失败”；
    未知后端返回 [] 并记录 ERROR。
    """
    if WEB_SEARCH_BACKEND == "duckduckgo":
        return await _search_duckduckgo(query)
    if WEB_SEARCH_BACKEND == "bing":
        return await _search_bing(query)
    logger.error("[WEB SEARCH] 未知后端 {}，本次搜索失败", WEB_SEARCH_BACKEND)
    return []


def _strip_html(text: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


async def _search_bing(query: str) -> list[dict[str, str]]:
    """Bing 网页搜索（无 Key，HTML 解析；国内网络通常可达）。

    注：HTML 结构可能随 Bing 改版变化，解析失败时返回 []（上层会降级提示搜索失败）。
    """
    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = await client.get(
            "https://www.bing.com/search",
            params={"q": query, "mkt": "zh-CN", "count": str(WEB_SEARCH_MAX_RESULTS + 5)},
        )
        response.raise_for_status()
        page = response.text

    # 结果块：<li class="b_algo">…<h2><a href>title</a></h2>…<p>snippet</p>
    block_re = re.compile(
        r'<li class="b_algo".*?<h2[^>]*>\s*<a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'.*?(?:<p[^>]*>(?P<snippet>.*?)</p>)?',
        re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for match in block_re.finditer(page):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet") or "")
        url = _html.unescape(match.group("url"))
        if not title and not snippet:
            continue
        results.append(
            {"title": title[:TITLE_MAX_CHARS], "url": url, "snippet": snippet[:SNIPPET_MAX_CHARS]}
        )
        if len(results) >= WEB_SEARCH_MAX_RESULTS:
            break
    _log_search_done("bing", query, len(results))
    return results


async def _search_duckduckgo(query: str) -> list[dict[str, str]]:
    """DuckDuckGo Instant Answer API（无需 Key）。"""
    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        response.raise_for_status()
        data = response.json()

    raw_results: list[dict[str, str]] = []
    if data.get("AbstractText"):
        raw_results.append(
            {
                "title": (data.get("Heading") or query)[:TITLE_MAX_CHARS],
                "url": data.get("AbstractURL") or "",
                "snippet": str(data["AbstractText"])[:SNIPPET_MAX_CHARS],
            }
        )
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            icon = topic.get("Icon") or {}
            raw_results.append(
                {
                    "title": str(topic["Text"])[:TITLE_MAX_CHARS],
                    "url": icon.get("URL") or topic.get("FirstURL") or "",
                    "snippet": str(topic["Text"])[:SNIPPET_MAX_CHARS],
                }
            )

    # 去重 + 数量限制
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_results:
        key = item["snippet"][:80]
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) >= WEB_SEARCH_MAX_RESULTS:
            break
    _log_search_done("duckduckgo", query, len(results))
    return results
