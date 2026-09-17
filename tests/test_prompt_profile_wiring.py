"""Prompt 元数据接线测试（v0.8）：InteractionProfile / Context Arbitration。

本文件验证的是**接线正确性**，不是措辞：
- profile 是否真的进入 SYSTEM（可信状态区），而不是被丢进 DATA；
- profile 缺失时是否与旧行为完全一致（向后兼容）；
- 画像是否覆盖全部 20 种组合、且每种组合都能安全渲染；
- CONTEXT_ARBITRATION_RULES / MEMORY_RULES 是否包含关键纪律；
- untrusted 边界没有被本次改造破坏。
"""

import json
import sys
from pathlib import Path

import pytest

import services.prompt_builder as prompt_builder
from services.context_arbitration import CONTEXT_ARBITRATION_RULES
from services.context_arbitration import content_tokens
from services.context_arbitration import has_substantive_content
from services.context_arbitration import should_inject_personal_memory
from services.interaction_profile import AFFECTION_LEVELS
from services.interaction_profile import RELATIONSHIP_LEVELS
from services.interaction_profile import build_interaction_profile
from services.prompt_builder import AFFECTION_RULES
from services.prompt_builder import MEMORY_RULES
from services.prompt_builder import RELATIONSHIP_RULES
from services.prompt_builder import STATIC_SYSTEM_PROMPT
from services.prompt_builder import CurrentUser
from services.prompt_builder import build_messages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_persona_behavior as eval_mod  # noqa: E402


def _direct(**kwargs) -> list[dict]:
    defaults = dict(
        current_user=CurrentUser(user_id=1001, display_name="小明"),
        relationship="familiar",
        memories=[],
        history=[],
        question="在吗",
        runtime_state="RUNTIME",
    )
    defaults.update(kwargs)
    return build_messages(**defaults)


# 判定“画像块是否被注入”的标记：用程序生成的那一行，而不是 “Interaction Profile”
# 这个短语本身 —— persona.txt 会正常提到这套机制的名字。
PROFILE_MARKER = "relationship=familiar / affection="
ANY_PROFILE_MARKER = " / affection="


class TestProfileReachesSystem:
    def test_profile_block_is_in_system_message(self):
        profile = build_interaction_profile("familiar", "very_distant")
        messages = _direct(interaction_profile=profile)
        system = messages[0]["content"]
        assert PROFILE_MARKER + "very_distant" in system
        assert "interruption_tolerance=very_low" in system
        assert "社交准入（access_privilege=" in system

    def test_profile_is_not_leaked_into_untrusted_data(self):
        profile = build_interaction_profile("close", "very_close")
        messages = _direct(interaction_profile=profile)
        data_message = messages[1]["content"]
        assert ANY_PROFILE_MARKER not in data_message
        assert "access_privilege" not in data_message

    def test_profile_absent_keeps_legacy_system(self):
        """不传 profile 时 SYSTEM 与 v0.7 完全一致（不含程序生成的画像块）。"""
        system = _direct()[0]["content"]
        assert ANY_PROFILE_MARKER not in system
        assert "access_privilege=" not in system
        assert system.startswith(prompt_builder.CORE_PERSONA)
        assert "relationship: familiar" in system

    def test_every_combination_renders_safely(self):
        for relationship in RELATIONSHIP_LEVELS:
            for affection in AFFECTION_LEVELS:
                profile = build_interaction_profile(relationship, affection)
                system = _direct(
                    relationship=relationship, interaction_profile=profile
                )[0]["content"]
                assert f"relationship={relationship} / affection={affection}" in system
                # 画像块必须始终带着“不覆盖事实”的边界说明
                assert "照样要认真回答" in system

    def test_profile_block_comes_before_persona_rag_refs(self):
        """画像必须排在“检索到的语料参考”之前，避免被原句带偏。

        注意标记选择：PERSONA_RAG_RULES 里也会出现“夜子表达与反应参考”字样，
        因此这里用实际参考条目的结构标记，而不是那句说明。
        """

        class _Ref:
            persona_note = "被搭话时先判断对方想干什么"
            relation_stage = "stranger"
            emotion = ["警戒"]
            text = "有事就说。"

        profile = build_interaction_profile("stranger", "normal")
        system = _direct(interaction_profile=profile, persona_refs=[_Ref()])[0]["content"]
        assert system.index("【Interaction Profile（程序生成") < system.index("参考 1：")
        assert system.index("【当前请求可信状态（程序生成") < system.index(
            "【Interaction Profile（程序生成"
        )


class TestPokeProfileWiring:
    def _poke(self, **kwargs) -> list[dict]:
        defaults = dict(
            current_user=CurrentUser(user_id=1001, display_name="小明"),
            relationship="familiar",
            memories=[],
            history=[],
            question="",
            runtime_state="RUNTIME",
            conversation_mode="poke",
        )
        defaults.update(kwargs)
        return build_messages(**defaults)

    def test_poke_carries_profile_and_repeat_count(self):
        profile = build_interaction_profile("close", "very_close")
        system = self._poke(interaction_profile=profile, recent_poke_count=4)[0]["content"]
        assert "conversation_mode: poke" in system
        assert "recent_poke_count: 4" in system
        assert "access_privilege=trusted_exception" in system

    def test_poke_repeat_count_defaults_to_first_time(self):
        system = self._poke()[0]["content"]
        assert "recent_poke_count: 1" in system

    def test_poke_repeat_count_is_clamped_at_minimum(self):
        system = self._poke(recent_poke_count=0)[0]["content"]
        assert "recent_poke_count: 1" in system

    def test_poke_without_profile_keeps_legacy_shape(self):
        system = self._poke()[0]["content"]
        assert ANY_PROFILE_MARKER not in system
        joined = "\n".join(
            m["content"] if isinstance(m["content"], str) else "" for m in self._poke()
        )
        assert "current_user_id: 1001" in joined


class TestStaticRulesContainNewDiscipline:
    @pytest.mark.parametrize(
        "needle",
        [
            "记得 ≠ 必须提",
            "current_message > active_topic > relevant_memory > unrelated_history",
        ],
    )
    def test_arbitration_discipline_present(self, needle):
        assert needle in STATIC_SYSTEM_PROMPT

    def test_arbitration_rules_are_in_static_prompt(self):
        assert CONTEXT_ARBITRATION_RULES in STATIC_SYSTEM_PROMPT

    def test_relationship_rules_separate_familiarity_from_liking(self):
        assert "熟悉 ≠ 喜欢" in RELATIONSHIP_RULES
        assert "好感 ≠ 权限" in RELATIONSHIP_RULES
        assert "熟悉的冷" in RELATIONSHIP_RULES

    def test_affection_rules_are_not_a_warmth_dial(self):
        assert "主观倾向" in AFFECTION_RULES
        assert "不是" in AFFECTION_RULES
        assert "突然变得热情或温柔" in AFFECTION_RULES

    def test_memory_rules_forbid_reporting_memories(self):
        assert "记得 ≠ 必须提" in MEMORY_RULES
        assert "不要追问、不要试探" in MEMORY_RULES

    def test_scheduled_instructions_have_no_hardcoded_user(self):
        """定时问候不得硬编码任何具体 QQ 号（close 的唯一来源是 CLOSE_USER_ID）。

        断言「不出现 6 位以上连续数字」而不是某个具体号码：
        既覆盖了历史上硬编码 close 用户 QQ 的回归，也不把真实 QQ 号写进测试。
        """
        import re

        for name, instruction in prompt_builder.SCHEDULED_EVENT_INSTRUCTIONS.items():
            assert not re.search(r"\d{6,}", instruction), f"{name} 里出现了硬编码数字 ID"
            assert "QQ 号" not in instruction

    def test_morning_greeting_forbids_daily_roll_call(self):
        morning = prompt_builder.SCHEDULED_EVENT_INSTRUCTIONS["morning_greeting"]
        assert "不要罗列昨天的话题" in morning
        assert "不要每天固定点名同一个人" in morning


class TestContextArbitrationDecisions:
    @pytest.mark.parametrize(
        "message",
        [
            "",
            "？",
            "在吗",
            "在不在",
            "嗯",
            "哈哈哈哈",
            "好累",
            "谢谢",
            "我去睡了",
            "草",
            "。。。",
            "hi",
        ],
    )
    def test_filler_messages_do_not_pull_personal_memory(self, message):
        assert should_inject_personal_memory(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "VLA 在 C 里面到底怎么实现",
            "帮我看看这段 C++ 为什么崩了",
            "在吗，我的 Lean4 那个证明崩了",
            "我最近看了一本挺有意思的小说",
            "我又在配 VS",
            "表还没签字",
        ],
    )
    def test_substantive_messages_keep_personal_memory(self, message):
        assert should_inject_personal_memory(message) is True

    def test_content_tokens_drop_single_characters(self):
        assert content_tokens("嗯") == []
        assert "Ubuntu" in content_tokens("我喜欢 Ubuntu Mono")

    def test_has_substantive_content_accepts_any_real_token(self):
        assert has_substantive_content("嗯，那个表") is True


class TestOutputHygiene:
    """模型输出会被原样发到 QQ 群：协议片段泄漏必须被规则与 eval 双重拦住。

    真实事故：v0.8 live eval 中 case_06 把工具调用以文本形式吐了出来
    （`<tool_calls>` / `<invoke name="web_search">`），这段文本会直接发到群里。
    """

    def test_security_rules_forbid_protocol_leakage(self):
        rules = prompt_builder.SECURITY_RULES
        assert "输出卫生" in rules
        assert "原样发送到 QQ 群" in rules
        assert "<tool_calls>" in rules
        assert "正在搜索" in rules

    def test_hygiene_rule_is_in_every_mode(self):
        for mode_kwargs in (
            {},
            {"conversation_mode": "poke"},
            {"conversation_mode": "ambient", "ambient_context": "片段"},
        ):
            messages = _direct(**mode_kwargs)
            assert "输出卫生" in messages[0]["content"]

    @pytest.mark.parametrize(
        "text",
        [
            '<tool_calls>\n<invoke name="web_search">',
            "```xml\n<invoke name=\"web_search\">\n```",
            "让我先调用一下 web_search 工具",
        ],
    )
    def test_eval_hygiene_checker_catches_leaks(self, text):
        assert eval_mod.output_hygiene_problems(text)

    @pytest.mark.parametrize(
        "text",
        [
            "在。什么事。",
            "`new[]` 配的是 `delete[]`。\n\n```cpp\nchar* p = new char[10];\n```",
            "没干什么。有事？",
        ],
    )
    def test_eval_hygiene_checker_accepts_normal_replies(self, text):
        assert eval_mod.output_hygiene_problems(text) == []


class TestTrustBoundaryUnchanged:
    """本次改造不允许放松任何既有的信任分区。"""

    def test_untrusted_nickname_stays_out_of_system(self):
        profile = build_interaction_profile("stranger", "normal")
        injected = "UNTRUSTED-NICKNAME-TOKEN"
        messages = _direct(
            current_user=CurrentUser(user_id=1001, display_name=injected),
            interaction_profile=profile,
        )
        system = messages[0]["content"]
        # SYSTEM 只能出现程序生成的画像行，绝不能出现用户可控的显示名
        assert injected not in system
        assert ANY_PROFILE_MARKER in system
        assert injected in messages[1]["content"]

    def test_display_name_is_json_escaped_in_data(self):
        messages = _direct(
            current_user=CurrentUser(user_id=1001, display_name='a"b'),
            interaction_profile=build_interaction_profile("stranger", "normal"),
        )
        payload = json.loads(messages[1]["content"].split("\n", 1)[1])
        assert payload["current_user"]["display_name"] == 'a"b'

    def test_current_message_is_last(self):
        profile = build_interaction_profile("familiar", "normal")
        messages = _direct(question="在吗", interaction_profile=profile)
        assert messages[-1]["content"].endswith("当前消息：\n在吗")
