"""调阈值 / 检查知识库检索效果（不启动 Bot、不发 QQ 消息）。

用法：
    python scripts/test_knowledge_rag.py "笔试什么时候交"
    python scripts/test_knowledge_rag.py "笔试什么时候交" --top-k 5
    python scripts/test_knowledge_rag.py "笔试什么时候交" --min-score 0.0   # 看全部候选分数

用途：定 KNOWLEDGE_MIN_SCORE。
- 打印每条命中的分数、来源文件、片段编号与前 120 字；
- `--min-score 0.0` 可以看清「无关问题」的分数落在哪，从而把阈值卡在
  「相关问题」与「无关问题」之间。
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import knowledge_rag  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索效果测试")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--top-k", type=int, default=0, help="返回条数（默认取环境变量）")
    parser.add_argument("--min-score", type=float, default=-1.0, help="相似度阈值（默认取环境变量）")
    parser.add_argument("--show-block", action="store_true", help="同时打印注入 Prompt 的原文块")
    args = parser.parse_args()

    print(f"[INFO] 索引目录 : {knowledge_rag.resolve_index_dir()}")
    print(f"[INFO] 资料目录 : {knowledge_rag.resolve_source_dir()}")
    print(f"[INFO] 环境阈值 : top_k={knowledge_rag.KNOWLEDGE_TOP_K} "
          f"min_score={knowledge_rag.KNOWLEDGE_MIN_SCORE} "
          f"max_chars={knowledge_rag.KNOWLEDGE_MAX_CHARS} "
          f"max_per_file={knowledge_rag.KNOWLEDGE_MAX_PER_FILE}")

    if not knowledge_rag.index_available():
        print("[ERROR] 索引不可用：请先运行 python scripts/build_knowledge_rag.py", file=sys.stderr)
        return 1

    chunks = knowledge_rag.retrieve(
        args.query,
        top_k=args.top_k or None,
        min_score=None if args.min_score < 0 else args.min_score,
    )
    print(f"\n[QUERY] {args.query}")
    print(f"[HIT] {len(chunks)} 条\n")
    for i, c in enumerate(chunks, start=1):
        print(f"  {i}. score={c.score:.4f}  {c.source_file}  第 {c.chunk_index + 1} 段")
        print(f"     {c.text[:120].replace(chr(10), ' ')}")
        print()

    if args.show_block:
        print("=" * 70)
        print(knowledge_rag.build_knowledge_block(chunks))
        print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
