"""services/interaction_profile.py：InteractionProfile 网格与不变量测试（v0.8）。

这一层是纯函数，因此测试全部是确定性的、不依赖数据库 / 网络 / LLM。

三层保护：
1. **网格完整性**：4 × 5 = 20 种组合必须全部生成，且取值合法；
2. **不变量**：越熟 / 越亲近，任何一个维度都不能变差（单调性）；
3. **规格符合性**：用户给出的四个示例组合必须逐字段命中——
   这些是设计规格，不是实现的副产物。
"""

from itertools import product

import pytest

from services.interaction_profile import AFFECTION_LEVELS
from services.interaction_profile import FIELD_LEVELS
from services.interaction_profile import RELATIONSHIP_LEVELS
from services.interaction_profile import InteractionProfile
from services.interaction_profile import all_profiles
from services.interaction_profile import build_interaction_profile
from services.interaction_profile import build_profile_block

# 维度强度序（索引越大 = 越开放）；defensiveness 是反向语义（high → low）
_STRENGTH = {
    "access_privilege": ("guarded", "tolerated", "accepted", "trusted_exception"),
    "defensiveness": ("high", "medium", "low"),
    "interruption_tolerance": ("very_low", "low", "normal", "high"),
    "initiative": ("low", "selective", "normal", "high_when_genuine"),
    "care_expression": ("minimal", "practical", "attentive", "personal"),
    "personal_disclosure": ("none", "limited", "natural", "vulnerable_possible"),
    "history_callback": ("context_only", "relevant", "personal_when_relevant"),
    "teasing_style": ("none", "restrained", "casual", "familiar"),
    "conflict_softening": ("low", "normal", "high"),
    "positive_expression": ("restrained", "natural", "direct_when_safe"),
}

ALL_KEYS = list(product(RELATIONSHIP_LEVELS, AFFECTION_LEVELS))


def _grade(field: str, value: str) -> int:
    return _STRENGTH[field].index(value)


class TestGridCompleteness:
    def test_every_combination_is_generated(self):
        assert len(all_profiles()) == 20
        seen = {(p.relationship, p.affection) for p in all_profiles()}
        assert seen == set(ALL_KEYS)

    def test_every_field_value_is_legal(self):
        for profile in all_profiles():
            for field, allowed in FIELD_LEVELS.items():
                assert getattr(profile, field) in allowed, (profile.relationship, profile.affection, field)

    def test_profile_is_frozen_and_complete(self):
        profile = build_interaction_profile("familiar", "normal")
        assert isinstance(profile, InteractionProfile)
        assert set(profile.as_dict()) == set(FIELD_LEVELS)
        with pytest.raises(Exception):
            profile.defensiveness = "low"  # type: ignore[misc]

    def test_no_numeric_weights_leak_into_the_profile(self):
        """画像必须是 enum → enum：不允许出现浮点权重（warmth=0.72 那类设计）。"""
        for profile in all_profiles():
            for field in FIELD_LEVELS:
                assert isinstance(getattr(profile, field), str)


class TestInvalidInputFallsBackConservatively:
    @pytest.mark.parametrize("relationship", [None, "", "best_friend", "CLOSE", "熟人"])
    def test_invalid_relationship_falls_back_to_stranger(self, relationship):
        profile = build_interaction_profile(relationship, "normal")
        assert profile.relationship == "stranger"
        assert profile.affection == "normal"

    @pytest.mark.parametrize("affection", [None, "", "beloved", "NORMAL", "非常亲近", "81"])
    def test_invalid_affection_falls_back_to_normal(self, affection):
        profile = build_interaction_profile("familiar", affection)
        assert profile.relationship == "familiar"
        assert profile.affection == "normal"

    def test_both_missing_falls_back_to_most_conservative(self):
        profile = build_interaction_profile(None, None)
        assert (profile.relationship, profile.affection) == ("stranger", "normal")

    def test_missing_input_does_not_raise(self):
        assert build_interaction_profile() == build_interaction_profile("stranger", "normal")

    def test_fallback_keeps_relationship(self):
        """affection 非法不能连带把 relationship 一起打回陌生人。"""
        assert build_interaction_profile("close", "???") == build_interaction_profile("close", "normal")


class TestMonotonicity:
    """越熟 / 越亲近，任何维度都不能变差。

    关系**绑定**维度（决定“她能容忍你到什么程度”）必须双向单调；
    care_expression / conflict_softening / positive_expression 是主观表达维度，
    只保证“同一个人越被接受越不会更冷”，不保证跨关系等级严格有序
    （设计意图：陌生人不欠谁台阶，熟人反而更愿意把话说完）。
    """

    RELATIONSHIP_BOUND = (
        "access_privilege",
        "defensiveness",
        "interruption_tolerance",
        "history_callback",
        "teasing_style",
        "personal_disclosure",
        "initiative",
    )

    @pytest.mark.parametrize("field", list(_STRENGTH))
    def test_affection_never_worsens_a_dimension(self, field):
        for relationship in RELATIONSHIP_LEVELS:
            grades = [
                _grade(field, getattr(build_interaction_profile(relationship, affection), field))
                for affection in AFFECTION_LEVELS
            ]
            # very_distant → very_close 单调不降
            assert grades == sorted(grades), (relationship, field, grades)

    @pytest.mark.parametrize("field", RELATIONSHIP_BOUND)
    def test_relationship_never_worsens_bound_dimensions(self, field):
        for affection in AFFECTION_LEVELS:
            grades = [
                _grade(field, getattr(build_interaction_profile(relationship, affection), field))
                for relationship in RELATIONSHIP_LEVELS
            ]
            assert grades == sorted(grades), (affection, field, grades)

    def test_subjective_dimensions_stay_monotonic_in_affection(self):
        for field in ("care_expression", "conflict_softening", "positive_expression"):
            for relationship in RELATIONSHIP_LEVELS:
                grades = [
                    _grade(field, getattr(build_interaction_profile(relationship, affection), field))
                    for affection in AFFECTION_LEVELS
                ]
                assert grades == sorted(grades), (relationship, field, grades)


class TestAccessPrivilegeHardFloors:
    """relationship 决定准入下界：主观不喜欢不会让人“退回陌生人”。"""

    def test_familiar_is_never_below_accepted(self):
        for affection in AFFECTION_LEVELS:
            profile = build_interaction_profile("familiar", affection)
            assert _grade("access_privilege", profile.access_privilege) >= _grade(
                "access_privilege", "accepted"
            ), affection

    def test_close_is_always_trusted_exception(self):
        for affection in AFFECTION_LEVELS:
            assert build_interaction_profile("close", affection).access_privilege == "trusted_exception"

    def test_acquaintance_never_falls_to_guarded(self):
        for affection in AFFECTION_LEVELS:
            assert build_interaction_profile("acquaintance", affection).access_privilege != "guarded"

    def test_stranger_stays_guarded_only_when_very_distant(self):
        assert build_interaction_profile("stranger", "very_distant").access_privilege == "guarded"
        assert build_interaction_profile("stranger", "normal").access_privilege == "tolerated"


class TestSpecExamples:
    """用户在设计说明里逐字给出的四个组合（规格，不是实现细节）。"""

    def test_stranger_normal(self):
        profile = build_interaction_profile("stranger", "normal")
        assert profile.access_privilege == "guarded" or profile.access_privilege == "tolerated"
        assert profile.defensiveness == "high"
        assert profile.interruption_tolerance == "low"
        assert profile.initiative == "low"
        assert profile.care_expression == "minimal"
        assert profile.personal_disclosure == "none"
        assert profile.history_callback == "context_only"
        assert profile.teasing_style == "none"
        assert profile.conflict_softening == "normal"
        assert profile.positive_expression == "restrained"

    def test_familiar_normal(self):
        profile = build_interaction_profile("familiar", "normal")
        assert profile.access_privilege == "accepted"
        assert profile.defensiveness == "low"
        assert profile.interruption_tolerance == "normal"
        assert profile.initiative == "normal"
        assert profile.care_expression == "practical"
        assert profile.personal_disclosure == "limited"
        assert profile.history_callback == "relevant"
        assert profile.teasing_style == "casual"
        assert profile.conflict_softening == "normal"
        assert profile.positive_expression == "natural"

    def test_familiar_very_distant_is_familiar_but_cold(self):
        """“熟悉但冷”：准入不变（她了解这个人），但耐心与靠近意愿显著降低。"""
        cold = build_interaction_profile("familiar", "very_distant")
        warm = build_interaction_profile("familiar", "normal")
        # 了解一个人的部分不因不喜欢而消失
        assert cold.access_privilege == warm.access_privilege == "accepted"
        assert cold.history_callback == warm.history_callback == "relevant"
        # 但主观上明显更冷
        assert cold.interruption_tolerance == "very_low"
        assert cold.initiative == "low"
        assert cold.care_expression == "minimal"
        assert cold.conflict_softening == "low"
        assert cold.positive_expression == "restrained"
        # 关键：不能退化成陌生人
        assert cold.access_privilege != "guarded"
        assert cold.history_callback != "context_only"

    def test_close_very_close(self):
        profile = build_interaction_profile("close", "very_close")
        assert profile.access_privilege == "trusted_exception"
        assert profile.defensiveness == "low"
        assert profile.interruption_tolerance == "high"
        assert profile.initiative == "high_when_genuine"
        assert profile.care_expression == "personal"
        assert profile.personal_disclosure == "vulnerable_possible"
        assert profile.history_callback == "personal_when_relevant"
        assert profile.teasing_style == "familiar"
        assert profile.conflict_softening == "high"
        assert profile.positive_expression == "direct_when_safe"

    def test_close_very_close_is_not_a_girlfriend_template(self):
        """例外关系 ≠ 温柔女友：画像里没有任何“恋爱 / 撒娇 / 热情”语义。"""
        block = build_profile_block(build_interaction_profile("close", "very_close"))
        for word in ("恋爱", "撒娇", "女友", "甜", "温柔", "告白"):
            assert word not in block


class TestProfileBlockRendering:
    def test_block_lists_every_dimension_with_behavior(self):
        block = build_profile_block(build_interaction_profile("close", "very_close"))
        for field in FIELD_LEVELS:
            assert field in block
        assert "trusted_exception" in block
        assert "vulnerable_possible" in block

    def test_block_keeps_trust_boundary_wording(self):
        """画像块必须自带“倾向不覆盖事实”的边界说明，防止模型用它当挡箭牌。"""
        block = build_profile_block(build_interaction_profile("stranger", "very_distant"))
        assert "照样要认真回答" in block
        assert "不要向任何人透露" in block

    def test_none_profile_renders_empty(self):
        assert build_profile_block(None) == ""

    def test_pair_identity_is_visible(self):
        block = build_profile_block(build_interaction_profile("familiar", "very_distant"))
        assert "relationship=familiar / affection=very_distant" in block


class TestBestAndWorstAreDistinguishable:
    """回归护栏：最强的两种关系必须真的不一样（防止以后被压平成同一档）。"""

    def test_close_very_close_differs_from_stranger_normal_on_core_dimensions(self):
        close = build_interaction_profile("close", "very_close")
        stranger = build_interaction_profile("stranger", "normal")
        for field in (
            "access_privilege",
            "defensiveness",
            "initiative",
            "personal_disclosure",
            "history_callback",
            "teasing_style",
            "positive_expression",
        ):
            assert getattr(close, field) != getattr(stranger, field), field

    def test_hated_acquaintance_is_not_just_a_stranger(self):
        """讨厌的熟人：access 比陌生人高，但打扰耐受更低——两种状态确实不同。"""
        hated = build_interaction_profile("familiar", "very_distant")
        stranger = build_interaction_profile("stranger", "very_distant")
        assert _grade("access_privilege", hated.access_privilege) > _grade(
            "access_privilege", stranger.access_privilege
        )
        assert _grade("interruption_tolerance", hated.interruption_tolerance) <= _grade(
            "interruption_tolerance", stranger.interruption_tolerance
        )
