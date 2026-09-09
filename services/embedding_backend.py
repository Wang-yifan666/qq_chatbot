"""Embedding Backend（v0.1）：业务无关的文本向量编码后端。

设计目标：
- 业务逻辑（Persona RAG / 以后可能的其他检索）只依赖 EmbeddingBackend 这个
  简单接口，不与具体模型绑定；
- 模型由环境变量 PERSONA_RAG_EMBEDDING_MODEL 指定，可替换；
- 进程内只加载一次：第一次调用 encode() 时完成惰性单例初始化
  （也可通过 warmup() 在 Bot 启动时预热），绝不为每条 QQ 消息重新加载；
- 输出统一 L2 归一化（cosine = dot product），dtype 为 float32。

第一版后端：sentence-transformers（CPU 推理）。
默认模型 BAAI/bge-small-zh-v1.5：中文语义检索效果较好、体积小（约 100MB）、
CPU 可运行。更换模型 = 改环境变量 + 重新运行 scripts/build_persona_rag.py。
"""

import os
import threading

import numpy as np
from nonebot import logger

from services import redact_secrets

# 默认 embedding 模型（环境变量 PERSONA_RAG_EMBEDDING_MODEL 可覆盖）
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 单次 encode 的最大文本数（防御性上限，避免异常调用打爆内存）
_MAX_BATCH_SIZE = 256


def _env_model_name() -> str:
    """读取模型名环境变量：空 / 空白 → 默认模型。"""
    name = (os.getenv("PERSONA_RAG_EMBEDDING_MODEL") or "").strip()
    return name or DEFAULT_EMBEDDING_MODEL


def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；零向量行保持零向量（避免除零）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class EmbeddingBackend:
    """文本 → 归一化向量。内部包装 sentence-transformers，惰性加载。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()  # 保护首次加载（double-checked）

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        """模型只加载一次；并发首次调用也只会加载一次。加载失败抛出异常，
        由调用方决定降级方式（Persona RAG 会降级为不启用）。

        加载策略：先尝试本地缓存（local_files_only，离线可用、秒开）；
        缓存不存在时再走网络下载（尊重 HF_ENDPOINT 镜像环境变量）。
        """
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                # 显式 CPU：QQ Bot 主链路不应抢占 GPU 显存
                try:
                    self._model = SentenceTransformer(
                        self.model_name, device="cpu", local_files_only=True
                    )
                    logger.info(
                        "[EMBEDDING] embedding 模型已从本地缓存加载：{}", self.model_name
                    )
                except Exception as exc:
                    logger.info(
                        "[EMBEDDING] 本地缓存未命中（{}），联网下载模型 {}…"
                        "（网络受限可设置 HF_ENDPOINT=https://hf-mirror.com）",
                        type(exc).__name__,
                        self.model_name,
                    )
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                    logger.info("[EMBEDDING] embedding 模型加载完成：{}", self.model_name)
            return self._model

    # ------------------------------------------------------------------
    # 接口
    # ------------------------------------------------------------------
    def dimension(self) -> int:
        """向量维度（不触发前向推理）。用于索引一致性校验。"""
        model = self._ensure_loaded()
        dim = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(dim):
            return int(dim())
        # 兜底：极小批量前向一次求维度
        return int(self.encode(["维度探测"])).reshape(1, -1).shape[1]

    def encode(self, texts: list[str]) -> np.ndarray:
        """把文本列表编码为归一化 float32 矩阵 (N, D)；空列表返回 (0, D)。"""
        if not texts:
            return np.zeros((0, self.dimension()), dtype=np.float32)
        model = self._ensure_loaded()
        batch = [str(t) for t in texts]
        embeddings = model.encode(
            batch,
            batch_size=min(len(batch), 64),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        # 双保险：即使模型配置未归一化也强制归一化（cosine = dot 的前提）
        return normalize_vectors(matrix)


# ----------------------------------------------------------------------
# 进程级单例（模型只加载一次）
# ----------------------------------------------------------------------
_instance: EmbeddingBackend | None = None
_instance_lock = threading.Lock()


def get_embedding_backend(model_name: str | None = None) -> EmbeddingBackend:
    """获取全局唯一 EmbeddingBackend。

    model_name=None 时使用环境变量；不同名字会替换实例（一般不应发生，
    因为换模型需要重建索引，索引一致性校验会拦截）。
    """
    global _instance
    name = (model_name or _env_model_name()).strip() or DEFAULT_EMBEDDING_MODEL
    with _instance_lock:
        if _instance is None or _instance.model_name != name:
            _instance = EmbeddingBackend(name)
        return _instance


def warmup_embedding_backend() -> bool:
    """Bot 启动时预热：加载模型（失败返回 False，绝不抛出）。"""
    try:
        backend = get_embedding_backend()
        backend._ensure_loaded()
        logger.info("[EMBEDDING] warmup 完成：{} (dim={})", backend.model_name, backend.dimension())
        return True
    except Exception as exc:
        logger.error(
            "[EMBEDDING] 模型加载失败（Persona RAG 将不可用，Bot 不受影响）：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False
