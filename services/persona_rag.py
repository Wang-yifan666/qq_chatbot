"""Persona RAG（v0.1）：夜子人格语料 → 本地索引 → 动态过滤 → 检索 → rerank → diversity。

职责（只做检索这一件事，不负责组装 Prompt）：
1. 加载已构建好的本地索引（NumPy 归一化矩阵 + JSONL metadata + index_config.json）；
2. 构造运行时 Query（当前用户消息 + 最近少量相关群聊上下文，不写死关系）；
3. 对整个矩阵计算 cosine similarity，取较大的候选集；
4. 运行时基础过滤：rag_quality / spoiler_level / romance_specific / intimacy_level
   （rag_candidate 只是上一阶段标注脚本的派生结果，只用于 debug 对比，
   绝不作为过滤条件）；
5. rerank：semantic × relation_factor × quality_factor × plot_factor + topic_bonus；
6. diversity 去重（文本近似 / embedding 近似 / source 去重）；
7. 返回 PersonaReference 列表（数据类，不把原始 JSON 到处传）。

任何失败都降级为 []（Bot 正常回答，绝不让 embedding 故障使 Bot 掉线）。

索引结构（scripts/build_persona_rag.py 生成）：
    data/persona_rag/embeddings.npy     (N, D) float32，已归一化
    data/persona_rag/metadata.jsonl     每行一个 DialogueUnit + retrieval_text
    data/persona_rag/index_config.json  embedding_model / embedding_dimension /
                                        corpus_count / build_time / schema_version
"""

import difflib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nonebot import logger

from services import redact_secrets
from services.embedding_backend import EmbeddingBackend
from services.embedding_backend import get_embedding_backend

# ==========================================================================
# 配置（环境变量，非法值 → warning + 安全默认值；改 .env 需重启生效）
# ==========================================================================

DEFAULTS = {
    "PERSONA_RAG_ENABLED": True,
    "PERSONA_RAG_CORPUS": "data/persona_processed/yako_processed.jsonl",
    "PERSONA_RAG_INDEX_DIR": "data/persona_rag",
    "PERSONA_RAG_CANDIDATE_K": 24,
    "PERSONA_RAG_TOP_K": 4,
    "PERSONA_RAG_MIN_SCORE": 0.38,
    "PERSONA_RAG_MIN_QUALITY": 0.65,
    "PERSONA_RAG_MAX_SPOILER_LEVEL": 0,
    "PERSONA_RAG_MAX_CHARS": 800,
    "PERSONA_RAG_CONTEXT_MAX_MESSAGES": 3,
    "PERSONA_RAG_TOPIC_BONUS": 0.05,
}

# 索引 schema 版本：build 脚本写，加载时校验（不匹配 → 禁用 RAG + 明确错误）
INDEX_SCHEMA_VERSION = 1

# 检索到的最小语义相关性阈值是模型相关的：默认值 0.38 是针对
# BAAI/bge-small-zh-v1.5 的初版经验值（实测：相关命中 final≥0.40，
# 无关技术问题约 0.32~0.36，阈值卡在两者之间），可经 PERSONA_RAG_MIN_SCORE 调整。
# 注意：阈值作用于 rerank 之后的 final_score（≤ semantic score）。


def _env_bool(name: str) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw == "":
        return bool(DEFAULTS[name])
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    logger.warning("[PERSONA RAG] {}={} 不是合法布尔值，使用默认值 {}", name, raw, DEFAULTS[name])
    return bool(DEFAULTS[name])


def _env_int(name: str, min_value: int, max_value: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(DEFAULTS[name])
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[PERSONA RAG] {}={} 不是合法整数，使用默认值 {}", name, raw, DEFAULTS[name])
        return int(DEFAULTS[name])
    if not (min_value <= value <= max_value):
        logger.warning(
            "[PERSONA RAG] {}={} 超出范围 [{}, {}]，使用默认值 {}",
            name,
            value,
            min_value,
            max_value,
            DEFAULTS[name],
        )
        return int(DEFAULTS[name])
    return value


def _env_float(name: str, min_value: float, max_value: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return float(DEFAULTS[name])
    try:
        value = float(raw)
    except ValueError:
        logger.warning("[PERSONA RAG] {}={} 不是合法数字，使用默认值 {}", name, raw, DEFAULTS[name])
        return float(DEFAULTS[name])
    if not (min_value <= value <= max_value):
        logger.warning(
            "[PERSONA RAG] {}={} 超出范围 [{}, {}]，使用默认值 {}",
            name,
            value,
            min_value,
            max_value,
            DEFAULTS[name],
        )
        return float(DEFAULTS[name])
    return value


# 进程启动时解析一次
PERSONA_RAG_ENABLED = _env_bool("PERSONA_RAG_ENABLED")
PERSONA_RAG_CORPUS = (os.getenv("PERSONA_RAG_CORPUS") or DEFAULTS["PERSONA_RAG_CORPUS"]).strip()
PERSONA_RAG_INDEX_DIR = (os.getenv("PERSONA_RAG_INDEX_DIR") or DEFAULTS["PERSONA_RAG_INDEX_DIR"]).strip()
PERSONA_RAG_CANDIDATE_K = _env_int("PERSONA_RAG_CANDIDATE_K", 5, 100)
PERSONA_RAG_TOP_K = _env_int("PERSONA_RAG_TOP_K", 1, 8)
PERSONA_RAG_MIN_SCORE = _env_float("PERSONA_RAG_MIN_SCORE", 0.0, 1.0)
PERSONA_RAG_MIN_QUALITY = _env_float("PERSONA_RAG_MIN_QUALITY", 0.0, 1.0)
PERSONA_RAG_MAX_SPOILER_LEVEL = _env_int("PERSONA_RAG_MAX_SPOILER_LEVEL", 0, 3)
PERSONA_RAG_MAX_CHARS = _env_int("PERSONA_RAG_MAX_CHARS", 100, 4000)
PERSONA_RAG_CONTEXT_MAX_MESSAGES = _env_int("PERSONA_RAG_CONTEXT_MAX_MESSAGES", 0, 3)
PERSONA_RAG_TOPIC_BONUS = _env_float("PERSONA_RAG_TOPIC_BONUS", 0.0, 0.05)

# rerank 池大小：先取 rerank Top-N 再做 diversity（池大于 TOP_K 才有去重空间）
DIVERSITY_POOL = 16

# diversity 阈值（第一版简单规则，后续实测调整）
DEDUP_TEXT_SIM = 0.92  # difflib 字符串相似度：超过视为“高度近似”
DEDUP_EMB_SIM = 0.93  # 已选中的 embedding 点积：超过视为近乎重复
MAX_REFS_PER_SOURCE = 1  # 最终结果中每个 source 文件最多 1 条（防止相邻台词占满 Top-K）

# --------------------------------------------------------------------------
# 关系规则（不是最终神圣参数，集中在此便于实测调整）
# 核心原则：Relationship 改变“允许什么人格状态被优先检索”，
# 而不是“切换成另一个人格”。
# --------------------------------------------------------------------------

VALID_RELATIONSHIPS = ("stranger", "acquaintance", "familiar", "close")

# 普通模式（close ≠ 恋人）允许的最大 intimacy_level，双保险：
# 即使某条数据因标注误差 romance_specific=false，intimacy_level >= 3 也排除。
MAX_INTIMACY = {
    "stranger": 0,
    "acquaintance": 1,
    "familiar": 2,
    "close": 2,
}

# relation_stage → 加权系数（不是过滤）。未知 stage 防御性回落 0.5。
RELATION_WEIGHTS = {
    "stranger": {
        "hostile": 1.00,
        "guarded": 1.00,
        "neutral": 0.75,
        "accustomed": 0.35,
        "trusting": 0.10,
        "vulnerable": 0.05,
    },
    "acquaintance": {
        "hostile": 0.35,
        "guarded": 0.75,
        "neutral": 1.00,
        "accustomed": 0.85,
        "trusting": 0.35,
        "vulnerable": 0.10,
    },
    "familiar": {
        "hostile": 0.10,
        "guarded": 0.35,
        "neutral": 0.75,
        "accustomed": 1.00,
        "trusting": 0.85,
        "vulnerable": 0.35,
    },
    "close": {
        "hostile": 0.05,
        "guarded": 0.20,
        "neutral": 0.60,
        "accustomed": 0.90,
        "trusting": 1.00,
        "vulnerable": 0.75,
    },
}
DEFAULT_STAGE_WEIGHT = 0.5

# plot_specific 不硬过滤：标注可能把可泛化场景误标为剧情专属，
# rerank 小幅降权即可。核心剧透仍由 spoiler_level 硬过滤。
PLOT_FACTOR = 0.90

# quality_factor = 0.7 + 0.3 * rag_quality：不让 rag_quality 完全支配语义相关性
QUALITY_FACTOR_BASE = 0.7
QUALITY_FACTOR_SCALE = 0.3

# --------------------------------------------------------------------------
# topic 小加分：tag match 只是辅助信号，绝不能压过 semantic similarity。
# TOPIC_ALIASES：候选 topic 标签 → 查询文本中可能出现的触发词。
# "other" / "unknown" 是标注里的兜底标签，不参与加分。
# --------------------------------------------------------------------------

TOPIC_ALIASES = {
    "book": ["小说", "书", "阅读", "读书", "看书", "文学", "图书馆", "文库", "novel", "book", "reading", "library", "literature"],
    "magic_book": ["魔法书", "魔法之书", "魔法", "magic book"],
    "grief": ["难过", "难受", "伤心", "痛苦", "低落", "想哭", "sad"],
    "comfort": ["安慰", "难受", "难过", "累", "撑不住", "低落"],
    "greeting": ["在吗", "在不在", "你好", "hello", "嗨"],
    "teasing": ["可爱", "喜欢", "夸", "调戏", "逗"],
    "fear": ["害怕", "恐惧", "怕", "scared"],
    "trust": ["信任", "相信", "trust"],
    "privacy": ["隐私", "秘密", "private"],
    "school": ["学校", "上学", "作业", "考试", "老师", "同学"],
    "food": ["吃", "饭", "饿", "料理", "食物"],
    "request": ["帮我", "帮个忙", "麻烦", "求"],
    "refusal": ["拒绝", "不要", "不想"],
    "conflict": ["吵架", "冲突", "生气", "吵"],
    "daily_chat": ["天气", "今天", "日常", "在干嘛", "在做什么"],
}

# 不参与 topic 加分的兜底标签
_TOPIC_NOOP = {"other", "unknown"}


@dataclass
class PersonaReference:
    """注入 Prompt 的角色语料参考（只携带必要字段，不传原始 JSON）。"""

    id: str
    text: str
    persona_note: str
    relation_stage: str
    intimacy_level: int
    emotion: list[str]
    topics: list[str]
    score: float  # rerank 后的 final_score


# ==========================================================================
# Query 构造
# ==========================================================================


def _truncate(text: str, limit: int) -> str:
    """按字符数截断（超限补省略号）。"""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def build_query_text(
    question: str,
    history: list,
    max_context_messages: int = PERSONA_RAG_CONTEXT_MAX_MESSAGES,
    max_chars: int = PERSONA_RAG_MAX_CHARS,
) -> str:
    """构造运行时 Query 文本（不写入 relationship —— 关系留给 rerank）。

    结构：
        当前用户消息：{question}
        最近相关群聊上下文：
        - {最近第 1 条}
        - {最近第 2 条}
        - {最近第 3 条}

    - history 只取最近 max_context_messages 条“真正相关”的消息：
      跳过机器人自己的回复、过短噪音（<2 字符）；最多 3 条，
      绝不用完整 20 条群聊历史；
    - 总字符数受 max_chars 约束（PERSONA_RAG_MAX_CHARS，默认 800），
      超限时截断上下文而不是问题本身。
    """
    question = (question or "").strip()
    head = "当前用户消息：\n" + _truncate(question, 400)
    if max_context_messages <= 0:
        return _truncate(head, max_chars)

    # 最近在前（倒序遍历 history，history 本身不含当前问题）
    context_lines: list[str] = []
    for message in reversed(list(history or [])):
        role = getattr(message, "role", "user")
        content = (getattr(message, "content", "") or "").strip()
        if role != "user":  # 跳过机器人自己的回复
            continue
        if not content:
            continue
        if content == question:
            continue
        if len(content) < 2:  # 过短噪音（“啊”“哦”等）
            continue
        context_lines.append(_truncate(content, 200))
        if len(context_lines) >= max_context_messages:
            break

    parts = [head]
    budget = max_chars
    budget -= len(head)
    if context_lines and budget > 20:
        parts.append("最近相关群聊上下文：")
        budget -= len(parts[-1])
        used = 0
        for line in context_lines:
            prefix = "- "
            remaining = budget - used - len(prefix)
            if remaining <= 0:
                break
            clipped = _truncate(line, remaining)
            parts.append(prefix + clipped)
            used += len(prefix) + len(clipped)
    return "\n".join(parts)


# ==========================================================================
# topic 加分
# ==========================================================================


def _topic_bonus(query_lower: str, topics: list[str]) -> float:
    """候选 topics 命中查询触发词时的少量加分（≤ PERSONA_RAG_TOPIC_BONUS）。

    每个 topic 命中加 TOPIC_BONUS/2，最多计 2 个 topic（即封顶 TOPIC_BONUS）。
    tag match 永远只是辅助信号，不会压过 semantic similarity。
    """
    topics = [t for t in (topics or []) if t and t not in _TOPIC_NOOP]
    if not topics or not query_lower:
        return 0.0
    matched = 0
    for topic in topics:
        aliases = TOPIC_ALIASES.get(topic)
        if not aliases:
            continue
        if any(keyword.lower() in query_lower for keyword in aliases):
            matched += 1
            if matched >= 2:
                break
    return PERSONA_RAG_TOPIC_BONUS * matched / 2.0


# ==========================================================================
# 索引加载与一致性校验
# ==========================================================================


class _LoadedIndex:
    """已加载并通过一致性校验的索引 + 预计算的过滤/加权向量。"""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
        config: dict,
        base_mask: np.ndarray,
        intimacy_masks: dict[str, np.ndarray],
        relation_weight_vectors: dict[str, np.ndarray],
        quality_weights: np.ndarray,
        plot_weights: np.ndarray,
    ):
        self.embeddings = embeddings
        self.metadata = metadata
        self.config = config
        self.base_mask = base_mask
        self.intimacy_masks = intimacy_masks
        self.relation_weight_vectors = relation_weight_vectors
        self.quality_weights = quality_weights
        self.plot_weights = plot_weights

    @property
    def count(self) -> int:
        return len(self.metadata)


def _normalize_model_name(name: str) -> str:
    """模型名一致性对比用的轻量归一化（取最后一段路径）。"""
    return (name or "").strip().replace("\\", "/").rstrip("/").split("/")[-1].lower()


def _load_index(index_dir: str, backend: EmbeddingBackend) -> _LoadedIndex | None:
    """加载索引并做一致性校验；任何不一致 → ERROR 日志 + 返回 None（禁用 RAG）。

    校验项：
    - index_config.json 存在且可解析，schema_version 匹配；
    - embeddings.npy 维度 == index_config.embedding_dimension == 当前模型维度；
    - metadata.jsonl 行数 == embeddings 行数 == index_config.corpus_count；
    - index_config.embedding_model 与当前模型名（归一化后）一致
      （模型不同 → 向量空间不同，检索结果无意义，必须重建索引）。
    """
    root = Path(index_dir)
    config_path = root / "index_config.json"
    emb_path = root / "embeddings.npy"
    meta_path = root / "metadata.jsonl"

    if not config_path.is_file() or not emb_path.is_file() or not meta_path.is_file():
        logger.error(
            "[PERSONA RAG] 索引不完整（缺 embeddings.npy / metadata.jsonl / index_config.json 之一）："
            "{}。请运行 python scripts/build_persona_rag.py 构建。Persona RAG 已禁用。",
            index_dir,
        )
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path, allow_pickle=False)
        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = [json.loads(line) for line in fh if line.strip()]
    except Exception as exc:
        logger.error(
            "[PERSONA RAG] 索引读取失败：{}。请重新构建。Persona RAG 已禁用。{}: {}",
            index_dir,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None

    if int(config.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
        logger.error(
            "[PERSONA RAG] 索引 schema_version={} 与程序期望的 {} 不一致，"
            "请重新运行 python scripts/build_persona_rag.py。Persona RAG 已禁用。",
            config.get("schema_version"),
            INDEX_SCHEMA_VERSION,
        )
        return None

    expected_dim = int(config.get("embedding_dimension", -1))
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        logger.error(
            "[PERSONA RAG] embeddings.npy 维度 {} 与 index_config.json 的 {} 不一致，"
            "请重建索引。Persona RAG 已禁用。",
            embeddings.shape if embeddings.ndim == 2 else embeddings.shape,
            expected_dim,
        )
        return None

    try:
        backend_dim = backend.dimension()
    except Exception as exc:
        logger.error(
            "[PERSONA RAG] 当前 embedding 模型不可用，Persona RAG 已禁用：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return None
    if expected_dim != backend_dim:
        logger.error(
            "[PERSONA RAG] 索引维度 {} 与当前模型 {} 输出维度 {} 不一致，"
            "索引与模型不匹配，请重建索引。Persona RAG 已禁用。",
            expected_dim,
            backend.model_name,
            backend_dim,
        )
        return None

    expected_count = int(config.get("corpus_count", -1))
    if embeddings.shape[0] != len(metadata) or len(metadata) != expected_count:
        logger.error(
            "[PERSONA RAG] 数量不一致：embeddings={} metadata={} config={}，"
            "索引损坏，请重建。Persona RAG 已禁用。",
            embeddings.shape[0],
            len(metadata),
            expected_count,
        )
        return None

    if _normalize_model_name(config.get("embedding_model", "")) != _normalize_model_name(backend.model_name):
        logger.error(
            "[PERSONA RAG] 索引由模型 {} 构建，当前模型为 {}，向量空间不同，"
            "检索结果无意义。请改回原模型或重新构建索引。Persona RAG 已禁用。",
            config.get("embedding_model"),
            backend.model_name,
        )
        return None

    # ---- 预计算过滤 / 加权向量（每次检索免去逐行 Python 过滤） ----
    count = len(metadata)
    intimacy_levels = np.asarray(
        [int(m.get("intimacy_level") or 0) for m in metadata], dtype=np.int64
    )
    rag_quality = np.clip(
        np.asarray([float(m.get("rag_quality") or 0.0) for m in metadata]), 0.0, 1.0
    )
    spoiler = np.asarray([int(m.get("spoiler_level") or 0) for m in metadata], dtype=np.int64)
    romance = np.asarray([bool(m.get("romance_specific")) for m in metadata])
    plot = np.asarray([bool(m.get("plot_specific")) for m in metadata])
    has_text = np.asarray([bool((m.get("text") or "").strip()) for m in metadata])
    is_yako = np.asarray([(m.get("speaker") or "") == "夜子" for m in metadata])
    stages = [str(m.get("relation_stage") or "") for m in metadata]

    base_mask = (
        (rag_quality >= PERSONA_RAG_MIN_QUALITY)
        & (spoiler <= PERSONA_RAG_MAX_SPOILER_LEVEL)
        & (~romance)
        & has_text
        & is_yako
    )
    intimacy_masks = {
        rel: intimacy_levels <= max_level for rel, max_level in MAX_INTIMACY.items()
    }
    relation_weight_vectors = {
        rel: np.asarray(
            [RELATION_WEIGHTS[rel].get(stage, DEFAULT_STAGE_WEIGHT) for stage in stages],
            dtype=np.float32,
        )
        for rel in RELATION_WEIGHTS
    }
    quality_weights = (QUALITY_FACTOR_BASE + QUALITY_FACTOR_SCALE * rag_quality).astype(np.float32)
    plot_weights = np.where(plot, PLOT_FACTOR, 1.0).astype(np.float32)

    logger.info(
        "[PERSONA RAG] 索引加载成功：{} 条 corpus，dim={}，模型 {}",
        count,
        expected_dim,
        backend.model_name,
    )
    return _LoadedIndex(
        embeddings=embeddings.astype(np.float32, copy=False),
        metadata=metadata,
        config=config,
        base_mask=base_mask,
        intimacy_masks=intimacy_masks,
        relation_weight_vectors=relation_weight_vectors,
        quality_weights=quality_weights,
        plot_weights=plot_weights,
    )


# 索引 / 后端单例（惰性加载 + 线程安全）
_index: _LoadedIndex | None = None
_index_lock = threading.Lock()
_index_load_attempted = False


def _get_index() -> _LoadedIndex | None:
    """惰性加载索引；加载失败后每次仍会重试（构建好索引无需重启 Bot）。"""
    global _index, _index_load_attempted
    with _index_lock:
        if _index is None and not _index_load_attempted:
            _index_load_attempted = True
            try:
                backend = get_embedding_backend()
                _index = _load_index(PERSONA_RAG_INDEX_DIR, backend)
            except Exception as exc:
                logger.error(
                    "[PERSONA RAG] 索引初始化失败，Persona RAG 已禁用：{}: {}",
                    type(exc).__name__,
                    redact_secrets(str(exc)),
                )
                _index = None
        return _index


def warmup() -> bool:
    """Bot 启动时预热：加载模型 + 索引（在线程中调用；失败返回 False，绝不抛出）。"""
    if not PERSONA_RAG_ENABLED:
        logger.info("[PERSONA RAG] PERSONA_RAG_ENABLED=false，已跳过预热")
        return False
    try:
        index = _get_index()
        return index is not None
    except Exception as exc:
        logger.error(
            "[PERSONA RAG] warmup 失败（Bot 不受影响）：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


def _debug_enabled() -> bool:
    """debug 开关实时读环境变量（脚本里可在 import 后再设置）。"""
    return (os.getenv("PERSONA_RAG_DEBUG") or "").strip().lower() in ("1", "true", "yes", "on")


# ==========================================================================
# diversity 去重
# ==========================================================================

_WS_RE = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WS_RE.sub("", (text or "").strip())


def _diversity_select(
    index: _LoadedIndex,
    ordered_rows: np.ndarray,
    final_scores: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """从 rerank 排序后的候选池中做简单多样性选择。

    规则（第一版，不用复杂 MMR）：
    - text 完全重复（去空白后相同）→ 跳过；
    - text 与已选中高度近似（difflib ≥ DEDUP_TEXT_SIM）→ 跳过；
    - embedding 与已选中近似（点积 ≥ DEDUP_EMB_SIM）→ 跳过；
    - 同一 source 文件已选中一条 → 跳过（防止相邻台词占满 Top-K）。
    """
    selected: list[tuple[int, float]] = []
    selected_texts: list[str] = []
    selected_embs: list[np.ndarray] = []
    selected_sources: set[str] = set()

    for row, score in zip(ordered_rows.tolist(), final_scores.tolist()):
        if len(selected) >= top_k:
            break
        meta = index.metadata[row]
        text = _norm_text(meta.get("text") or "")
        source_file = str((meta.get("source") or {}).get("file") or "")

        if text in selected_texts:
            continue
        if source_file and source_file in selected_sources:
            continue
        emb = index.embeddings[row]
        if any(float(emb @ prev) >= DEDUP_EMB_SIM for prev in selected_embs):
            continue
        if any(difflib.SequenceMatcher(None, text, prev).ratio() >= DEDUP_TEXT_SIM for prev in selected_texts):
            continue

        selected.append((row, score))
        selected_texts.append(text)
        selected_embs.append(emb)
        selected_sources.add(source_file)

    return selected


# ==========================================================================
# 主检索入口
# ==========================================================================


def retrieve(
    question: str,
    relationship: str = "stranger",
    history: list | None = None,
) -> list[PersonaReference]:
    """检索夜子人格语料参考（同步函数；QQ 链路中请用 asyncio.to_thread 调用）。

    - 返回 [] 的情况：RAG 未启用 / 索引不可用 / 无超过阈值的相关样本。
      调用方不需要 try/except（本函数内部已兜底），但建议仍按规范包一层。
    - relationship 非法时防御性回落 stranger。
    """
    if not PERSONA_RAG_ENABLED:
        return []

    rel = relationship if relationship in VALID_RELATIONSHIPS else "stranger"

    try:
        index = _get_index()
        if index is None:
            return []

        backend = get_embedding_backend()
        query_text = build_query_text(question, history or [])
        query_vec = backend.encode([query_text])[0]

        # 1) 整个矩阵计算 cosine（embeddings 已归一化 → dot = cosine）
        scores = index.embeddings @ query_vec

        # 2) 基础过滤（硬规则）：不合格行置 -inf，保证候选集只来自合格语料
        eligible = index.base_mask & index.intimacy_masks[rel]
        masked = np.where(eligible, scores, -np.inf)

        # 3) 第一阶段召回：较大的候选集
        k = min(PERSONA_RAG_CANDIDATE_K, index.count)
        if k <= 0 or not np.any(eligible):
            return []
        # argpartition 取 top-k（-inf 自动沉底）
        if k < index.count:
            partitioned = np.argpartition(-masked, k - 1)[:k]
            valid = partitioned[np.isfinite(masked[partitioned])]
        else:
            valid = np.flatnonzero(np.isfinite(masked))
        semantic = scores[valid]
        order = np.argsort(-semantic)
        candidates = valid[order]
        semantic = semantic[order]

        # 4) rerank（乘法可解释，第一版保持简单）
        rel_weights = index.relation_weight_vectors[rel][candidates]
        quality = index.quality_weights[candidates]
        plot = index.plot_weights[candidates]
        final = semantic * rel_weights * quality * plot

        query_lower = query_text.lower()
        for i, row in enumerate(candidates.tolist()):
            final[i] += _topic_bonus(query_lower, index.metadata[row].get("topics") or [])

        # 5) 相似度阈值：宁可 [] 也不硬塞无关语录
        pool_order = np.argsort(-final)
        pool_rows = candidates[pool_order]
        pool_scores = final[pool_order]
        keep = pool_scores >= PERSONA_RAG_MIN_SCORE
        pool_rows = pool_rows[keep][:DIVERSITY_POOL]
        pool_scores = pool_scores[keep][:DIVERSITY_POOL]

        # 6) diversity 去重 → Top-K
        selected = _diversity_select(index, pool_rows, pool_scores, PERSONA_RAG_TOP_K)

        refs: list[PersonaReference] = []
        for row, score in selected:
            meta = index.metadata[row]
            refs.append(
                PersonaReference(
                    id=str(meta.get("id") or row),
                    text=(meta.get("text") or "").strip(),
                    persona_note=(meta.get("persona_note") or "").strip(),
                    relation_stage=str(meta.get("relation_stage") or ""),
                    intimacy_level=int(meta.get("intimacy_level") or 0),
                    emotion=[str(e) for e in (meta.get("emotion") or [])],
                    topics=[str(t) for t in (meta.get("topics") or [])],
                    score=float(round(score, 4)),
                )
            )

        _log_retrieval(query_text, rel, question, index, candidates, semantic, final,
                       pool_rows, pool_scores, selected)
        return refs
    except Exception as exc:  # 绝不向上抛：任何故障 → 空结果
        logger.exception(
            "[PERSONA RAG] retrieve 失败（降级为无参考，Bot 正常回答）：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return []


def _log_retrieval(
    query_text: str,
    rel: str,
    question: str,
    index: _LoadedIndex,
    candidates: np.ndarray,
    semantic: np.ndarray,
    final: np.ndarray,
    pool_rows: np.ndarray,
    pool_scores: np.ndarray,
    selected: list[tuple[int, float]],
) -> None:
    """PERSONA_RAG_DEBUG=true 时的调试日志。

    只输出 query / 关系 / 数量 / 分数 / 标签与截断摘要，
    绝不把大量完整原作台词写进生产日志。
    """
    if not _debug_enabled():
        return
    eligible = int(np.sum(index.base_mask & index.intimacy_masks[rel]))
    logger.info(
        "[PERSONA RAG DEBUG] query={} relationship={} corpus={} eligible={} candidates={} pool={} selected={}",
        _truncate(query_text.replace("\n", " ⏎ "), 120),
        rel,
        index.count,
        eligible,
        len(candidates),
        len(pool_rows),
        len(selected),
    )
    for row, score in selected:
        meta = index.metadata[row]
        logger.info(
            "[PERSONA RAG DEBUG] id={} final={} relation_stage={} intimacy={} quality={} topics={} text={}",
            meta.get("id"),
            round(score, 4),
            meta.get("relation_stage"),
            meta.get("intimacy_level"),
            meta.get("rag_quality"),
            (meta.get("topics") or [])[:6],
            _truncate(meta.get("text") or "", 40),
        )
    for i in range(min(PERSONA_RAG_CANDIDATE_K, len(candidates))):
        row = int(candidates[i])
        meta = index.metadata[row]
        logger.info(
            "[PERSONA RAG DEBUG] cand id={} semantic={} final={} stage={} q={} rag_candidate={}",
            meta.get("id"),
            round(float(semantic[i]), 4),
            round(float(final[i]), 4),
            meta.get("relation_stage"),
            meta.get("rag_quality"),
            meta.get("rag_candidate"),
        )
