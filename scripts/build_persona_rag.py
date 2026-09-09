"""构建 Persona RAG 索引（v0.1）。

用法（在项目根目录执行）：
    python scripts/build_persona_rag.py

或显式指定参数：
    python scripts/build_persona_rag.py --corpus data/persona_processed/yako_processed.jsonl \
                                        --out data/persona_rag \
                                        --model BAAI/bge-small-zh-v1.5 \
                                        --batch-size 32

功能：
1. 读取标注语料 JSONL（只读，绝不修改原始 processed JSONL）；
2. 为每个 DialogueUnit 构造检索专用文本 retrieval_text
   （人格反应 persona_note 放在最前 —— 目标是“夜子在类似情况下会如何反应”，
   而不是普通剧情文本搜索；source file / line / chapter / id / source_route
   属于 metadata，绝不进 embedding）；
3. context 只取最后 1~3 条必要前文并限制字符，避免 embedding 被剧情细节淹没；
4. 批量编码 → L2 归一化 → 保存：
       embeddings.npy      (N, D) float32
       metadata.jsonl      原始字段 + retrieval_text / retrieval_context
       index_config.json   embedding_model / embedding_dimension / corpus_count /
                           build_time / schema_version
5. 运行时不重算：Bot 只加载索引。语料更新后重新运行本脚本即可。

注意：corpus 是版权数据（原作台词），data/persona_processed/ 与 data/persona_rag/
均已被 .gitignore 忽略，禁止提交。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（scripts/ 的上一级），保证从任何 cwd 运行都能解析相对路径和 services 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 与 services/persona_rag.py 保持一致（构建/加载必须使用同一版本）
INDEX_SCHEMA_VERSION = 1

# retrieval_text 中 context 的预算：最后最多 3 条、单条最多 80 字符、总计最多 240 字符
CONTEXT_MAX_ITEMS = 3
CONTEXT_ITEM_MAX_CHARS = 80
CONTEXT_TOTAL_MAX_CHARS = 240

# 编码批次大小（CPU 推理，太大内存/耗时都不可控）
DEFAULT_BATCH_SIZE = 32


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def build_retrieval_context(unit: dict) -> tuple[list[str], str]:
    """从 DialogueUnit.context 取“最后 1~3 条必要前文”。

    返回 (实际进入 embedding 的前文列表, 拼好的前文文本块)。
    - 只取 context 列表的最后 CONTEXT_MAX_ITEMS 条；
    - 每条拼成 “speaker: text” 并截断；跳过空文本；
    - 总计不超过 CONTEXT_TOTAL_MAX_CHARS。
    """
    raw_context = unit.get("context") or []
    picked: list[str] = []
    budget = CONTEXT_TOTAL_MAX_CHARS
    for item in reversed(raw_context):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        speaker = (item.get("speaker") or "").strip()
        line = _truncate(f"{speaker}：{text}" if speaker else text, CONTEXT_ITEM_MAX_CHARS)
        if len(line) > budget:
            line = _truncate(line, budget)
        picked.append(line)
        budget -= len(line)
        if budget <= 0 or len(picked) >= CONTEXT_MAX_ITEMS:
            break
    picked.reverse()  # 恢复时间顺序
    block = "\n".join(f"- {line}" for line in picked)
    return picked, block


def build_retrieval_text(unit: dict) -> tuple[str, str]:
    """构造检索专用文本 retrieval_text。

    字段顺序（persona_note 必须靠前）：
        人格反应：{persona_note}
        话题：{topics}
        回应方式：{response_intent}
        情绪：{emotion}
        人际状态：{relation_stage}
        必要前文：{最近少量 context}
        夜子回答：{text}

    source file / line / chapter / source_route / id 等属于 metadata，
    绝不放进 embedding。
    """
    def _join(values, sep: str = ", ") -> str:
        return sep.join(str(v) for v in (values or []) if str(v).strip())

    picked_context, context_block = build_retrieval_context(unit)

    parts: list[str] = []
    persona_note = _truncate(unit.get("persona_note") or "", 500)
    if persona_note:
        parts.append(f"人格反应：{persona_note}")
    topics = _join(unit.get("topics"))
    if topics:
        parts.append(f"话题：{topics}")
    intent = (unit.get("response_intent") or "").strip()
    if intent:
        parts.append(f"回应方式：{intent}")
    emotion = _join(unit.get("emotion"))
    if emotion:
        parts.append(f"情绪：{emotion}")
    stage = (unit.get("relation_stage") or "").strip()
    if stage:
        parts.append(f"人际状态：{stage}")
    if context_block:
        parts.append(f"必要前文：\n{context_block}")
    text = (unit.get("text") or "").strip()
    if text:
        parts.append(f"夜子回答：{_truncate(text, 300)}")

    retrieval_text = "\n".join(parts).strip()
    return retrieval_text, context_block


def read_corpus(corpus_path: Path) -> list[dict]:
    """逐行读取 JSONL；坏行跳过并计数（绝不因此中断构建）。"""
    units: list[dict] = []
    broken = 0
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                unit = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败，已跳过", file=sys.stderr)
                continue
            units.append(unit)
    if broken:
        print(f"[WARN] 共跳过 {broken} 行无法解析的记录", file=sys.stderr)
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Persona RAG 本地索引")
    parser.add_argument(
        "--corpus",
        default=os.getenv("PERSONA_RAG_CORPUS") or "data/persona_processed/yako_processed.jsonl",
        help="标注语料 JSONL 路径（默认取环境变量 PERSONA_RAG_CORPUS）",
    )
    parser.add_argument(
        "--out",
        default=os.getenv("PERSONA_RAG_INDEX_DIR") or "data/persona_rag",
        help="索引输出目录（默认取环境变量 PERSONA_RAG_INDEX_DIR）",
    )
    parser.add_argument(
        "--model",
        default=(os.getenv("PERSONA_RAG_EMBEDDING_MODEL") or "").strip() or "BAAI/bge-small-zh-v1.5",
        help="embedding 模型名（默认取环境变量 PERSONA_RAG_EMBEDDING_MODEL）",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="编码批次大小")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if not corpus_path.is_file():
        print(f"[ERROR] corpus 不存在：{corpus_path}", file=sys.stderr)
        return 1

    print(f"[BUILD] corpus: {corpus_path}")
    print(f"[BUILD] index dir: {out_dir}")
    print(f"[BUILD] embedding model: {args.model}（首次运行需下载，约 100MB；"
          f"网络受限时可用 HF_ENDPOINT=https://hf-mirror.com）")

    # 惰性单例：模型只加载一次
    from services.embedding_backend import get_embedding_backend

    backend = get_embedding_backend(args.model)

    print("[BUILD] 读取语料...")
    units = read_corpus(corpus_path)

    # 只索引 speaker=夜子 且 text 非空的行（其余跳过，计数日志）
    rows: list[dict] = []
    skipped = {"not_yako": 0, "empty_text": 0}
    for unit in units:
        if (unit.get("speaker") or "") != "夜子":
            skipped["not_yako"] += 1
            continue
        if not (unit.get("text") or "").strip():
            skipped["empty_text"] += 1
            continue
        retrieval_text, context_block = build_retrieval_text(unit)
        # 复制原始记录作为 metadata（保留全部原始字段，绝不修改原 JSONL），
        # 追加机器生成字段。
        meta = dict(unit)
        meta["retrieval_text"] = retrieval_text
        meta["retrieval_context"] = context_block
        rows.append(meta)
    print(f"[BUILD] 语料 {len(units)} 行：跳过非夜子 {skipped['not_yako']}，"
          f"跳过空台词 {skipped['empty_text']}，实际索引 {len(rows)} 行")

    if not rows:
        print("[ERROR] 没有可索引的语料", file=sys.stderr)
        return 1

    # 批量编码（构造好 retrieval_text 后统一喂给模型）
    print(f"[BUILD] 编码 {len(rows)} 条 retrieval_text（batch={args.batch_size}）...")
    texts = [row["retrieval_text"] for row in rows]
    matrices: list = []
    for start in range(0, len(texts), args.batch_size):
        chunk = texts[start : start + args.batch_size]
        matrices.append(backend.encode(chunk))
        done = min(start + args.batch_size, len(texts))
        print(f"[BUILD]   进度 {done}/{len(texts)}")

    import numpy as np

    embeddings = np.vstack(matrices).astype(np.float32)
    # 双保险归一化（cosine = dot 的前提）
    from services.embedding_backend import normalize_vectors

    embeddings = normalize_vectors(embeddings)

    dim = backend.dimension()
    assert embeddings.shape[1] == dim, f"编码维度 {embeddings.shape[1]} 与模型声明 {dim} 不一致"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 原子写入：先写临时文件再 replace，避免写到一半被 Bot 读到
    # （np.save 会自动补 .npy 后缀，所以临时名以 .tmp 结尾即可）
    emb_tmp = out_dir / "embeddings.tmp"
    meta_tmp = out_dir / "metadata.jsonl.tmp"
    config_tmp = out_dir / "index_config.json.tmp"
    np.save(emb_tmp, embeddings)
    with meta_tmp.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            row = dict(row)
            row["index_ordinal"] = i
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    config = {
        "embedding_model": backend.model_name,
        "embedding_dimension": int(dim),
        "corpus_count": len(rows),
        "build_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "schema_version": INDEX_SCHEMA_VERSION,
    }
    config_tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    os.replace(out_dir / "embeddings.tmp.npy", out_dir / "embeddings.npy")
    os.replace(meta_tmp, out_dir / "metadata.jsonl")
    os.replace(config_tmp, out_dir / "index_config.json")

    print(f"[BUILD] 完成：{len(rows)} 条语料 → {out_dir}")
    print(f"[BUILD]   embeddings.npy     shape=({embeddings.shape[0]}, {embeddings.shape[1]})")
    print(f"[BUILD]   metadata.jsonl     {len(rows)} 行")
    print(f"[BUILD]   index_config.json  {json.dumps(config, ensure_ascii=False)}")
    print("[BUILD] 验证命令：python scripts/test_persona_rag.py \"在吗\" --relationship stranger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
