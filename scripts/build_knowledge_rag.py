"""构建知识库 RAG 索引：本地资料文档 → 可检索的参考资料索引。

用法（在项目根目录执行）：
    python scripts/build_knowledge_rag.py
    python scripts/build_knowledge_rag.py --source data/knowledge --out data/knowledge_index
    python scripts/build_knowledge_rag.py --chunk-chars 600 --overlap 120

流程：
    扫描资料目录 → 抽取正文（复用 services/perception 的解析器）
      → 分块（services.knowledge_rag.split_into_chunks）
      → embedding（复用 Persona RAG 的同一个模型后端）
      → 原子写出 embeddings.npy / chunks.jsonl / index_config.json

支持格式：pdf / docx / pptx / xlsx / md / markdown / txt
（抽取器与 QQ 群里发文件时用的是**同一套**，所以本地构建与线上读取行为一致。）

注意：
- 资料是版权 / 隐私数据，data/knowledge/ 与 data/knowledge_index/ 均已 gitignore；
- 只读取资料目录，绝不修改原始文件；
- 换 embedding 模型必须重建索引（向量空间不同）。
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

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.knowledge_rag import INDEX_SCHEMA_VERSION  # noqa: E402
from services.knowledge_rag import split_into_chunks  # noqa: E402

# 扩展名 → 解析器（与 services/perception/parsers.py 对应）
PARSERS = {
    ".pdf": "parse_pdf",
    ".docx": "parse_docx",
    ".pptx": "parse_pptx",
    ".xlsx": "parse_xlsx",
    ".md": "parse_text",
    ".markdown": "parse_text",
    ".txt": "parse_text",
}

# 单份资料抽取正文的字符上限（防止一份巨型文档把索引撑爆）
PER_FILE_MAX_CHARS = 400_000


def extract_text(path: Path) -> tuple[str, str]:
    """抽取一份资料的正文；返回 (文本, 说明)。失败返回 ("", 原因)。"""
    import services.perception.parsers as parsers

    func_name = PARSERS.get(path.suffix.lower())
    if func_name is None:
        return "", f"不支持的格式 {path.suffix}"
    func = getattr(parsers, func_name, None)
    if func is None:
        return "", f"解析器缺失 {func_name}"
    try:
        result = func(str(path), PER_FILE_MAX_CHARS)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if not result.ok:
        return "", result.note or "解析失败"
    return result.text, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="构建知识库 RAG 索引")
    parser.add_argument(
        "--source",
        default=os.getenv("KNOWLEDGE_SOURCE_DIR") or "data/knowledge",
        help="资料目录（默认取环境变量 KNOWLEDGE_SOURCE_DIR）",
    )
    parser.add_argument(
        "--out",
        default=os.getenv("KNOWLEDGE_INDEX_DIR") or "data/knowledge_index",
        help="索引输出目录（默认取环境变量 KNOWLEDGE_INDEX_DIR）",
    )
    parser.add_argument("--model", default="", help="embedding 模型（默认与 Persona RAG 同一个）")
    parser.add_argument("--chunk-chars", type=int, default=0, help="每块字符数（默认取环境变量）")
    parser.add_argument("--overlap", type=int, default=-1, help="块间重叠字符数（默认取环境变量）")
    parser.add_argument("--batch-size", type=int, default=32, help="编码批次大小")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if not source_dir.is_dir():
        source_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ERROR] 资料目录不存在，已为你创建：{source_dir}", file=sys.stderr)
        print(f"        请把资料放进去（支持 {' / '.join(sorted(PARSERS))}）后重跑本脚本。", file=sys.stderr)
        return 1

    from services import knowledge_rag

    chunk_chars = args.chunk_chars or knowledge_rag.KNOWLEDGE_CHUNK_CHARS
    overlap = knowledge_rag.KNOWLEDGE_CHUNK_OVERLAP if args.overlap < 0 else args.overlap

    print(f"[BUILD] 资料目录 : {source_dir}")
    print(f"[BUILD] 索引目录 : {out_dir}")
    print(f"[BUILD] 分块参数 : chunk_chars={chunk_chars} overlap={overlap}")

    files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in PARSERS and not p.name.startswith(".")
    )
    if not files:
        print(f"[ERROR] 资料目录里没有支持的文档（{'/'.join(sorted(PARSERS))}）", file=sys.stderr)
        return 1

    print(f"[BUILD] 发现 {len(files)} 份资料，开始抽取正文…")
    chunks: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for path in files:
        rel = path.relative_to(source_dir).as_posix()
        text, note = extract_text(path)
        if not text.strip():
            skipped.append((rel, note or "正文为空"))
            print(f"    [!] 跳过 {rel}：{note or '正文为空'}")
            continue
        pieces = split_into_chunks(text, chunk_chars=chunk_chars, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append({"text": piece, "source_file": rel, "chunk_index": i})
        print(f"    [√] {rel}：{len(text)} 字 → {len(pieces)} 块")

    if not chunks:
        print("[ERROR] 没有任何可用片段（全部资料抽取失败或为空）", file=sys.stderr)
        return 1

    print(f"[BUILD] 共 {len(chunks)} 块，开始编码…")
    from services.embedding_backend import get_embedding_backend
    from services.embedding_backend import normalize_vectors

    backend = get_embedding_backend(args.model.strip() or None)
    texts = [c["text"] for c in chunks]
    matrices = []
    for start in range(0, len(texts), args.batch_size):
        matrices.append(backend.encode(texts[start : start + args.batch_size]))
        print(f"[BUILD]   进度 {min(start + args.batch_size, len(texts))}/{len(texts)}")

    import numpy as np

    embeddings = normalize_vectors(np.vstack(matrices).astype(np.float32))
    dim = backend.dimension()
    assert embeddings.shape[1] == dim, f"编码维度 {embeddings.shape[1]} 与模型声明 {dim} 不一致"

    # 原子写入：先写临时文件再 replace，避免 Bot 读到写了一半的索引
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_tmp = out_dir / "embeddings.tmp"
    chunks_tmp = out_dir / "chunks.jsonl.tmp"
    config_tmp = out_dir / "index_config.json.tmp"

    np.save(emb_tmp, embeddings)
    with chunks_tmp.open("w", encoding="utf-8") as fh:
        for row in chunks:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    config = {
        "embedding_model": backend.model_name,
        "embedding_dimension": int(dim),
        "chunk_count": len(chunks),
        "source_files": sorted({c["source_file"] for c in chunks}),
        "chunk_chars": int(chunk_chars),
        "overlap": int(overlap),
        "build_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "schema_version": INDEX_SCHEMA_VERSION,
    }
    config_tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    os.replace(out_dir / "embeddings.tmp.npy", out_dir / "embeddings.npy")
    os.replace(chunks_tmp, out_dir / "chunks.jsonl")
    os.replace(config_tmp, out_dir / "index_config.json")

    print(f"\n[BUILD] 完成：{len(chunks)} 块 ← {len(files) - len(skipped)} 份资料 → {out_dir}")
    print(f"[BUILD]   embeddings.npy  shape=({embeddings.shape[0]}, {embeddings.shape[1]})")
    print(f"[BUILD]   模型            {backend.model_name}")
    if skipped:
        print(f"[BUILD]   跳过 {len(skipped)} 份：")
        for rel, why in skipped:
            print(f"             {rel}：{why}")
    print('\n[BUILD] 调阈值 / 看效果：python scripts/test_knowledge_rag.py "你的问题"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
