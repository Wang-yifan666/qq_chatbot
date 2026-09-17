"""夜子人格行为 eval（v0.8）：把关系/好感/情境喂进真实 Prompt，人工阅读模型输出。

为什么不是 pytest 断言
----------------------
人格的正确性无法用字符串断言衡量：“才没有”和“知道就行了”都可能是对的，
取决于关系、场合与说话的人。把它写成 assert 只会逼出“为了让测试通过而措辞”的
反效果。因此本脚本：

1. **默认（无参数）只做结构检查**：为 12 个 case 构造真实 Prompt（不联网、不花钱），
   校验画像注入、关系/好感标签、仲裁纪律是否都在位，并打印一个对照矩阵。
   它可以安全地放进 CI。
2. **`--live` 才真正调用模型**：把每个 case 的回答打印出来（或写进文件），
   由人对照 case 里的 expected_traits / forbidden_traits 阅读判断。

用法：
    python scripts/eval_persona_behavior.py                 # 结构检查 + 关系对照表
    python scripts/eval_persona_behavior.py --case case_12  # 只看某个 case
    python scripts/eval_persona_behavior.py --live          # 真实调用模型并输出回答
    python scripts/eval_persona_behavior.py --live --out eval_out.md

注意：`--live` 会真实消耗 API 额度，且需要 .env 中配置好 API Key。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from services.context_store import ChatMessage  # noqa: E402
from services.interaction_profile import build_interaction_profile  # noqa: E402
from services.prompt_builder import CurrentUser  # noqa: E402
from services.prompt_builder import ScheduledEvent  # noqa: E402
from services.prompt_builder import build_messages  # noqa: E402
from services.trigger_intensity import assess_trigger  # noqa: E402

CASES_FILE = PROJECT_ROOT / "tests" / "persona_cases.json"

# 每个 case 用一个稳定的假 user_id（不接触真实数据库）
DEFAULT_USER_ID = 2001

# 输出卫生检查：这些内容一旦出现在最终文本里，就是“会直接发到群里”的严重故障。
# 真实事故：eval 时模型把工具调用以文本形式吐了出来（<tool_calls> / <invoke>），
# 说明“模型把工具协议写进正文”这条风险必须在验收时被显式检查，而不是靠人眼扫过。
FORBIDDEN_OUTPUT_PATTERNS = (
    "<tool_calls>",
    "</tool_calls>",
    "<invoke",
    "</invoke>",
    "<function_calls>",
    "```xml",
    "web_search",
)


def output_hygiene_problems(text: str) -> list[str]:
    """检查最终文本是否可以安全直接发到群里。"""
    problems: list[str] = []
    lowered = (text or "").lower()
    for pattern in FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.lower() in lowered:
            problems.append(f"输出泄漏了工具调用/协议片段：{pattern}")
    if (text or "").strip().startswith("```"):
        problems.append("输出以代码块开头（不应把整条回复包进代码块）")
    return problems


def load_cases() -> list[dict]:
    payload = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    return payload["cases"]


def _history(case: dict) -> list[ChatMessage]:
    """把 case 的历史行变成 ChatMessage。

    history_speaker:
      - "current"（默认）：全部历史都由当前用户发出——适合"反复纠缠"这类
        必须能看出"同一个人在刷同一句话"的场景；
      - "mixed"：用户与夜子交替——适合普通多轮对话。
    """
    speaker = case.get("history_speaker", "current")
    messages: list[ChatMessage] = []
    for index, line in enumerate(case.get("history") or [], start=1):
        if speaker == "mixed":
            is_user = index % 2 == 1
        else:
            is_user = True
        messages.append(
            ChatMessage(
                id=index,
                group_id=900,
                user_id=DEFAULT_USER_ID if is_user else 3001,
                nickname="群友" if is_user else "夜子",
                role="user" if is_user else "assistant",
                content=line,
                created_at="2026-09-16 01:00:00",
            )
        )
    return messages


def build_case_messages(case: dict, relationship: str, affection: str, user_id: int = DEFAULT_USER_ID):
    """按 case 的 mode 构造真实 messages（与线上链路同一套 build_messages）。"""
    profile = build_interaction_profile(relationship, affection)
    mode = case.get("mode", "direct")
    history = _history(case)
    poke_count = int(case.get("poke_sequence") or 1)
    trigger = assess_trigger(
        case.get("message") or "",
        profile,
        recent_poke_count=poke_count,
        mode=mode,
        history=history,
        user_id=user_id,
    )

    if mode == "scheduled":
        event_type = case.get("scheduled_event", "morning_greeting")
        return build_messages(
            None,
            relationship,
            [],
            history,
            "",
            runtime_state="【runtime】2026-09-16 07:10（周三）",
            conversation_mode="scheduled",
            scheduled_event=ScheduledEvent(event_type, "2026-09-16 07:10:00", "07:10"),
            interaction_profile=profile,
        )

    if mode == "poke":
        return build_messages(
            current_user=CurrentUser(user_id=user_id, display_name="群友"),
            relationship=relationship,
            memories=[],
            history=history,
            question="",
            runtime_state="【runtime】2026-09-16 20:00",
            conversation_mode="poke",
            poke_back=True,
            interaction_profile=profile,
            recent_poke_count=poke_count,
            trigger=trigger,
        )

    return build_messages(
        current_user=CurrentUser(user_id=user_id, display_name="群友"),
        relationship=relationship,
        memories=[],
        history=history,
        question=case.get("message") or "",
        runtime_state="【runtime】2026-09-16 20:00",
        interaction_profile=profile,
        trigger=trigger,
    )


def case_trigger(case: dict, relationship: str, affection: str, user_id: int = DEFAULT_USER_ID):
    """该 case 在该关系下的强度判定（供打印与报告）。"""
    return assess_trigger(
        case.get("message") or "",
        build_interaction_profile(relationship, affection),
        recent_poke_count=int(case.get("poke_sequence") or 1),
        mode=case.get("mode", "direct"),
        history=_history(case),
        user_id=user_id,
    )


def case_variants(case: dict) -> list[tuple[str, str]]:
    """返回该 case 需要跑的 (relationship, affection) 组合。"""
    variants = case.get("variants")
    if variants:
        return [(item["relationship"], item["affection"]) for item in variants]
    return [(case.get("relationship", "stranger"), case.get("affection", "normal"))]


def structural_check(case: dict) -> list[str]:
    """结构检查：Prompt 是否真的带上了这一轮该有的可信状态。"""
    problems: list[str] = []
    for relationship, affection in case_variants(case):
        messages = build_case_messages(case, relationship, affection)
        system = messages[0]["content"]
        # scheduled 模式没有 current_user / 没有关系状态：它不注入画像块，
        # 相关边界由核心人格与定时任务指令负责。
        if case.get("mode", "direct") != "scheduled":
            if f"relationship={relationship} / affection={affection}" not in system:
                problems.append(f"缺少画像块：{relationship}+{affection}")
            if "照样要认真回答" not in system:
                problems.append("画像块缺少“不覆盖事实”的边界说明")
        if "记得 ≠ 必须提" not in system:
            problems.append("缺少上下文仲裁纪律")
        # direct / poke 必须带本轮强度（v0.9：有原因时充分表现的前提）
        if case.get("mode", "direct") in ("direct", "poke"):
            if "intensity: " not in system:
                problems.append("缺少 Trigger Intensity 强度块")
        if case.get("mode", "direct") == "direct":
            if not any("current_user_id" in str(m.get("content", "")) for m in messages):
                problems.append("direct 模式缺少 current_user_id")
    return problems


def print_case_summary(case: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"{case['id']}｜{case['title']}")
    print(f"{'-' * 78}")
    if case.get("variants"):
        for relationship, affection in case_variants(case):
            profile = build_interaction_profile(relationship, affection)
            trigger = case_trigger(case, relationship, affection)
            print(
                f"  {relationship:<12}+{affection:<13} "
                f"access={profile.access_privilege:<18} "
                f"def={profile.defensiveness:<7} "
                f"init={profile.initiative:<18} "
                f"care={profile.care_expression:<10} "
                f"→ trigger={trigger.category}/{trigger.intensity}"
            )
    else:
        relationship, affection = case_variants(case)[0]
        profile = build_interaction_profile(relationship, affection)
        trigger = case_trigger(case, relationship, affection)
        print(f"  关系/好感：{relationship} + {affection}（mode={case.get('mode', 'direct')}）")
        print(f"  强度：{trigger.category} / {trigger.intensity}（上限 {trigger.ceiling}）｜{trigger.reason}")
        print(
            f"  画像：access={profile.access_privilege} def={profile.defensiveness} "
            f"interrupt={profile.interruption_tolerance} init={profile.initiative} "
            f"care={profile.care_expression} disclose={profile.personal_disclosure} "
            f"history={profile.history_callback} tease={profile.teasing_style} "
            f"soften={profile.conflict_softening} positive={profile.positive_expression}"
        )
    print(f"  消息：{case.get('message') or '（主动发言）'}")
    print(f"  期望：{' / '.join(case.get('expected_traits', []))}")
    print(f"  禁止：{' / '.join(case.get('forbidden_traits', []))}")


async def run_live(cases: list[dict], out_path: Path | None) -> int:
    from services.llm_client import ask_with_fallback

    blocks: list[str] = ["# 夜子人格行为 eval（live 输出）", ""]
    hygiene_failures = 0
    for case in cases:
        print_case_summary(case)
        blocks.append(f"## {case['id']}｜{case['title']}")
        blocks.append("")
        blocks.append(f"消息：`{case.get('message') or '（主动发言）'}`")
        blocks.append("")
        for relationship, affection in case_variants(case):
            messages = build_case_messages(case, relationship, affection)
            trigger = case_trigger(case, relationship, affection)
            answer, provider = await ask_with_fallback(messages, tools=None)
            text = (answer or "（模型未返回内容）").strip()
            problems = output_hygiene_problems(text)
            marker = "  [FAIL]" if problems else ""
            print(f"  [{relationship}+{affection}] ({trigger.intensity}) ({provider}) {text}{marker}")
            for problem in problems:
                hygiene_failures += 1
                print(f"    - {problem}")
            blocks.append(
                f"**{relationship} + {affection}**（强度 `{trigger.intensity}` / {provider}）："
            )
            blocks.append("")
            blocks.append("> " + text.replace("\n", "\n> "))
            blocks.append("")
            if problems:
                blocks.append(f"- ⚠ 输出卫生问题：{'；'.join(problems)}")
                blocks.append("")
        blocks.append(f"- 期望：{'；'.join(case.get('expected_traits', []))}")
        blocks.append(f"- 禁止：{'；'.join(case.get('forbidden_traits', []))}")
        blocks.append("")
        blocks.append("---")
        blocks.append("")

    if out_path is not None:
        out_path.write_text("\n".join(blocks), encoding="utf-8")
        print(f"\n[OK] 已写出 {out_path}")

    if hygiene_failures:
        print(f"\n[FAIL] 输出卫生检查发现 {hygiene_failures} 处问题（见上）")
        return 1
    print("\n[OK] 输出卫生检查通过（无工具协议泄漏 / 无整条代码块包裹）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="夜子人格行为 eval")
    parser.add_argument("--case", default=None, help="只跑某个 case id（前缀匹配）")
    parser.add_argument("--live", action="store_true", help="真实调用模型（消耗 API 额度）")
    parser.add_argument("--out", default=None, help="live 模式下把结果写入文件")
    args = parser.parse_args()

    cases = load_cases()
    if args.case:
        cases = [case for case in cases if case["id"].startswith(args.case)]
        if not cases:
            print(f"[FAIL] 没有匹配的 case：{args.case}", file=sys.stderr)
            return 2

    print(f"载入 {len(cases)} 个 case（{CASES_FILE.name}）")

    failures = 0
    for case in cases:
        print_case_summary(case)
        problems = structural_check(case)
        if problems:
            failures += 1
            print(f"  [FAIL] {'；'.join(problems)}")
        else:
            print("  [OK] 结构检查通过（画像 / 仲裁纪律 / 信状态均在位）")

    print(f"\n结构检查：{len(cases) - failures}/{len(cases)} 通过")
    if failures:
        return 1

    if not args.live:
        print("\n（未加 --live：只做了结构检查。加 --live 才会真实调用模型供人工阅读。）")
        return 0

    out_path = Path(args.out) if args.out else None
    return asyncio.run(run_live(cases, out_path))


if __name__ == "__main__":
    raise SystemExit(main())
