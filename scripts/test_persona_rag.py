"""Persona RAG 本地检索测试（v0.1）。

在接入 QQ Bot 之前直接检查检索质量。示例：

    python scripts/test_persona_rag.py "在吗" --relationship stranger
    python scripts/test_persona_rag.py "最近有什么小说推荐吗" --relationship familiar
    python scripts/test_persona_rag.py "今天有点难受" --relationship close
    python scripts/test_persona_rag.py "你好可爱" --relationship acquaintance
    python scripts/test_persona_rag.py "STM32 的 DMA 怎么配置" --relationship familiar

支持一次跑多个查询（模型只加载一次）：
    python scripts/test_persona_rag.py "在吗" "你好可爱" --relationship familiar

可选参数：
    --relationship {stranger,acquaintance,familiar,close}  默认 stranger
    --context "模拟的群聊上下文"                            可重复，最多 3 条进入 query
    --debug                                                输出候选集明细
    --top-k N                                              覆盖 PERSONA_RAG_TOP_K
    --min-score F                                          覆盖 PERSONA_RAG_MIN_SCORE

输出：每条命中的 score / relation / intimacy / persona_note / 台词预览。
"""

import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（scripts/ 的上一级），保证从任何 cwd 运行都能导入 services 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import persona_rag
from services.persona_rag import PersonaReference


def print_refs(query: str, relationship: str, history: list) -> None:
    print(f"\n{'=' * 72}")
    print(f"Query      : {query}")
    print(f"Relationship: {relationship}")

    refs = persona_rag.retrieve(query, relationship, history)

    print(f"结果: {len(refs)} 条参考")
    if not refs:
        print("（无参考注入：Persona RAG 返回空，Prompt 将不注入角色语料）")
        return
    for i, ref in enumerate(refs, start=1):
        print(f"\n参考 {i}:")
        print(f"  score={ref.score}")
        print(f"  relation={ref.relation_stage}  intimacy={ref.intimacy_level}")
        print(f"  emotion={ref.emotion}")
        print(f"  topics={ref.topics}")
        print(f"  persona_note: {ref.persona_note}")
        print(f"  text preview: {ref.text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Persona RAG 本地检索测试")
    parser.add_argument("queries", nargs="+", help="一个或多个查询文本")
    parser.add_argument(
        "--relationship",
        default="stranger",
        choices=list(persona_rag.VALID_RELATIONSHIPS),
        help="模拟的 QQ 关系等级（默认 stranger）",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="模拟群聊上下文（可重复，最多 3 条进入 query）",
    )
    parser.add_argument("--debug", action="store_true", help="输出候选集调试明细")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 PERSONA_RAG_TOP_K")
    parser.add_argument("--min-score", type=float, default=None, help="覆盖 PERSONA_RAG_MIN_SCORE")
    args = parser.parse_args()

    if args.debug:
        os.environ["PERSONA_RAG_DEBUG"] = "true"
    if args.top_k is not None:
        persona_rag.PERSONA_RAG_TOP_K = args.top_k
    if args.min_score is not None:
        persona_rag.PERSONA_RAG_MIN_SCORE = args.min_score

    print(f"[INFO] PERSONA_RAG_ENABLED={persona_rag.PERSONA_RAG_ENABLED}")
    print(f"[INFO] index_dir={persona_rag.PERSONA_RAG_INDEX_DIR}")
    print(
        f"[INFO] 参数: candidate_k={persona_rag.PERSONA_RAG_CANDIDATE_K} "
        f"top_k={persona_rag.PERSONA_RAG_TOP_K} "
        f"min_score={persona_rag.PERSONA_RAG_MIN_SCORE} "
        f"min_quality={persona_rag.PERSONA_RAG_MIN_QUALITY} "
        f"max_spoiler={persona_rag.PERSONA_RAG_MAX_SPOILER_LEVEL} "
        f"max_chars={persona_rag.PERSONA_RAG_MAX_CHARS}"
    )

    # 预检查索引可用性（触发一次加载，把清晰错误打出来）
    index = persona_rag._get_index()
    if index is None:
        print(
            "\n[FAIL] Persona RAG 索引不可用：请先运行 python scripts/build_persona_rag.py",
            file=sys.stderr,
        )
        return 2

    Message = namedtuple("Message", ["role", "content"])
    history = [Message(role="user", content=line) for line in args.context[:3]]

    for query in args.queries:
        print_refs(query, args.relationship, history)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
