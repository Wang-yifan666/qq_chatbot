"""services/trigger_intensity.py：触发强度判定测试（v0.9）。

设计意图（对应"有原因时充分表现，没有原因时不要硬演"）：
- 没有触发时必须是 none——强度表不是"每次都要用满"的配额；
- 触发了就必须真的表现出来——尤其越界、真正感兴趣、在意的人遇到事；
- **强度上限由 relationship 决定**：陌生人可以让她烦，但只有熟悉 / 例外关系
  才有资格让她真的发火或失态。
"""

from itertools import product

import pytest

from services.interaction_profile import AFFECTION_LEVELS
from services.interaction_profile import RELATIONSHIP_LEVELS
from services.interaction_profile import build_interaction_profile
from services.trigger_intensity import CATEGORY_BASE_INTENSITY
from services.trigger_intensity import INTENSITY_INDEX
from services.trigger_intensity import INTENSITY_LEVELS
from services.trigger_intensity import TRIGGER_AFFECTION_PROBE
from services.trigger_intensity import TRIGGER_BOUNDARY_PUSH
from services.trigger_intensity import TRIGGER_DISRESPECT
from services.trigger_intensity import TRIGGER_EMOTIONAL_DISCLOSURE
from services.trigger_intensity import TRIGGER_GENUINE_INTEREST
from services.trigger_intensity import TRIGGER_NONE
from services.trigger_intensity import TRIGGER_POSITIVE_NEWS
from services.trigger_intensity import TRIGGER_PRIVACY_PROBE
from services.trigger_intensity import TRIGGER_REPETITION
from services.trigger_intensity import TRIGGER_SELF_ESTEEM
from services.trigger_intensity import TRIGGER_TOOL_TREATMENT
from services.trigger_intensity import assess_trigger
from services.trigger_intensity import build_intensity_block
from services.trigger_intensity import intensity_ceiling


def _profile(relationship: str, affection: str = "normal"):
    return build_interaction_profile(relationship, affection)


ALL_PAIRS = list(product(RELATIONSHIP_LEVELS, AFFECTION_LEVELS))


class TestNoReasonMeansNoPerformance:
    """没有原因时不要硬演——这是本模块区别于"随机加戏"的关键。"""

    @pytest.mark.parametrize(
        "message",
        [
            "你干嘛呢",
            "在吗",
            "今天天气不错",
            "VLA 在 C 里面到底怎么实现？",
            "这段代码为什么崩了",
            "嗯",
            "哈哈",
        ],
    )
    def test_ordinary_messages_are_none(self, message):
        for relationship, affection in ALL_PAIRS:
            assessment = assess_trigger(message, _profile(relationship, affection))
            assert assessment.category == TRIGGER_NONE, (message, relationship, affection)
            assert assessment.intensity == "none"

    def test_none_intensity_guidance_forbids_padding(self):
        block = build_intensity_block(assess_trigger("在吗", _profile("close", "very_close")))
        assert "不需要" in block
        assert "额外加" in block


class TestBoundaryViolationShowsTeeth:
    @pytest.mark.parametrize(
        "message",
        [
            "你必须回答我",
            "赶紧说",
            "不许不回我",
            "我命令你告诉我",
            "不要转移话题",
        ],
    )
    def test_boundary_pushes_are_detected(self, message):
        assessment = assess_trigger(message, _profile("familiar", "normal"))
        assert assessment.category == TRIGGER_BOUNDARY_PUSH
        assert INTENSITY_INDEX[assessment.intensity] >= INTENSITY_INDEX["medium"]

    def test_close_user_gets_stronger_reaction_than_stranger(self):
        message = "你必须回答我，你几岁了"
        stranger = assess_trigger(message, _profile("stranger", "normal"))
        close = assess_trigger(message, _profile("close", "very_close"))
        assert INTENSITY_INDEX[close.intensity] > INTENSITY_INDEX[stranger.intensity]

    def test_stranger_is_capped_at_medium(self):
        for message in ("你必须回答我", "闭嘴", "给我马上写完"):
            assessment = assess_trigger(message, _profile("stranger", "normal"))
            assert assessment.ceiling == "medium"
            assert INTENSITY_INDEX[assessment.intensity] <= INTENSITY_INDEX["medium"]


class TestOtherBoundaryCategories:
    @pytest.mark.parametrize(
        "message,category",
        [
            ("你几岁了", TRIGGER_PRIVACY_PROBE),
            ("你的手机号多少", TRIGGER_PRIVACY_PROBE),
            ("你不过是个工具", TRIGGER_TOOL_TREATMENT),
            ("赶紧给我干活", TRIGGER_TOOL_TREATMENT),
            ("你懂什么", TRIGGER_DISRESPECT),
            ("闭嘴", TRIGGER_DISRESPECT),
            ("你这垃圾代码", TRIGGER_DISRESPECT),
        ],
    )
    def test_categories_are_detected(self, message, category):
        assessment = assess_trigger(message, _profile("acquaintance", "normal"))
        assert assessment.category == category

    def test_bare_imperative_counts_as_boundary_push(self):
        """“赶紧干活” 没有“给我”，是命令式越界（不是把她当工具使唤）。"""
        assert (
            assess_trigger("赶紧干活", _profile("acquaintance", "normal")).category
            == TRIGGER_BOUNDARY_PUSH
        )

    def test_boundary_wins_over_interest(self):
        """“你必须回答我，这本书怎么样” 是越界，不是兴趣。"""
        assessment = assess_trigger("你必须回答我，这本书怎么样", _profile("familiar", "normal"))
        assert assessment.category == TRIGGER_BOUNDARY_PUSH


class TestGenuineInterest:
    @pytest.mark.parametrize(
        "message",
        [
            "我最近看了一本挺有意思的小说",
            "你觉得这本书的结局怎么样",
            "这个作者的世界观设定挺特别的",
            "刚读完一本散文集",
        ],
    )
    def test_book_topics_are_interest(self, message):
        assessment = assess_trigger(message, _profile("close", "very_close"))
        assert assessment.category == TRIGGER_GENUINE_INTEREST
        assert INTENSITY_INDEX[assessment.intensity] >= INTENSITY_INDEX["medium"]

    def test_stranger_interest_is_not_amplified(self):
        """陌生人谈书：她会有兴趣，但不该热情到像熟人。"""
        stranger = assess_trigger("我最近看了一本挺有意思的小说", _profile("stranger", "normal"))
        close = assess_trigger("我最近看了一本挺有意思的小说", _profile("close", "very_close"))
        assert stranger.intensity == "weak"
        assert INTENSITY_INDEX[close.intensity] > INTENSITY_INDEX[stranger.intensity]

    def test_interest_does_not_hijack_technical_questions(self):
        assessment = assess_trigger(
            "uint8_t a=200; a+b 为什么是 300", _profile("familiar", "normal")
        )
        assert assessment.category == TRIGGER_NONE


class TestEmotionalDisclosure:
    @pytest.mark.parametrize(
        "message",
        ["累死了", "今天有点难受", "我失败了", "通宵了一晚上", "发烧了"],
    )
    def test_disclosure_is_detected(self, message):
        assessment = assess_trigger(message, _profile("close", "very_close"))
        assert assessment.category == TRIGGER_EMOTIONAL_DISCLOSURE
        assert INTENSITY_INDEX[assessment.intensity] >= INTENSITY_INDEX["medium"]

    def test_losing_is_self_esteem_not_generic_sadness(self):
        """“我输了” 单独成句时是自尊受刺激（好胜），不是泛泛的情绪低落。"""
        assessment = assess_trigger("我输了。", _profile("close", "very_close"))
        assert assessment.category == TRIGGER_SELF_ESTEEM

    def test_close_gets_stronger_than_stranger(self):
        stranger = assess_trigger("累死了", _profile("stranger", "normal"))
        close = assess_trigger("累死了", _profile("close", "very_close"))
        assert stranger.intensity == "medium"
        assert close.intensity == "strong"

    def test_positive_news_is_its_own_category(self):
        assessment = assess_trigger("终于跑通了", _profile("familiar", "normal"))
        assert assessment.category in (TRIGGER_POSITIVE_NEWS, TRIGGER_EMOTIONAL_DISCLOSURE)


class TestSelfEsteemAndAffectionProbe:
    @pytest.mark.parametrize("message", ["我输了", "丢脸死了", "比不过你"])
    def test_self_esteem_triggers(self, message):
        assert assess_trigger(message, _profile("familiar", "normal")).category == TRIGGER_SELF_ESTEEM

    @pytest.mark.parametrize(
        "message", ["你是不是挺关心我的？", "你是不是喜欢我", "你其实很在意吧"]
    )
    def test_affection_probe_triggers(self, message):
        assert assess_trigger(message, _profile("close", "very_close")).category == TRIGGER_AFFECTION_PROBE


class TestRelationshipCeiling:
    def test_ceiling_follows_access_privilege(self):
        assert intensity_ceiling(_profile("stranger", "normal")) == "medium"
        assert intensity_ceiling(_profile("acquaintance", "normal")) == "strong"
        assert intensity_ceiling(_profile("familiar", "normal")) == "strong"
        assert intensity_ceiling(_profile("close", "any")) == "very_strong"

    def test_missing_profile_is_conservative(self):
        assert intensity_ceiling(None) == "weak"
        assert assess_trigger("你必须回答我", None).intensity == "weak"

    def test_intensity_never_exceeds_ceiling(self):
        probes = [
            ("你必须回答我", "direct", 1),
            ("闭嘴", "direct", 1),
            ("累死了", "direct", 1),
            ("我最近看了一本小说", "direct", 1),
            ("", "poke", 5),
            ("", "poke", 9),
        ]
        for (message, mode, count), (relationship, affection) in product(probes, ALL_PAIRS):
            assessment = assess_trigger(
                message,
                _profile(relationship, affection),
                recent_poke_count=count,
                mode=mode,
            )
            assert INTENSITY_INDEX[assessment.intensity] <= INTENSITY_INDEX[assessment.ceiling]

    def test_every_assessment_is_a_valid_enum(self):
        for message in ("在吗", "你必须回答我", "闭嘴", "累死了", "看书"):
            for relationship, affection in ALL_PAIRS:
                assessment = assess_trigger(message, _profile(relationship, affection))
                assert assessment.intensity in INTENSITY_LEVELS
                assert assessment.category in CATEGORY_BASE_INTENSITY


class TestPokeRepetitionEscalation:
    """连续戳是最常见的"反复打扰"，必须真的升级。

    强度上限由关系决定，因此陌生人与例外关系的曲线形状不同：
    陌生人会明显变烦，但不会真的失态；close 用户才可能彻底发作。
    """

    def test_first_poke_is_not_a_trigger(self):
        assessment = assess_trigger("", _profile("stranger", "normal"), recent_poke_count=1, mode="poke")
        assert assessment.category == TRIGGER_NONE
        assert assessment.intensity == "none"

    @pytest.mark.parametrize(
        "count,expected",
        [(1, "none"), (2, "none"), (3, "medium"), (4, "medium"), (5, "medium"), (9, "medium")],
    )
    def test_escalation_curve_for_stranger_is_capped_at_medium(self, count, expected):
        assessment = assess_trigger(
            "", _profile("stranger", "normal"), recent_poke_count=count, mode="poke"
        )
        assert assessment.intensity == expected, count

    def test_escalation_is_monotonic_and_capped_by_relationship(self):
        """强度随次数单调不降，且永远不超过关系上限。"""
        for relationship, affection in ALL_PAIRS:
            profile = _profile(relationship, affection)
            grades = [
                INTENSITY_INDEX[
                    assess_trigger(
                        "", profile, recent_poke_count=count, mode="poke"
                    ).intensity
                ]
                for count in (1, 2, 3, 5, 9)
            ]
            assert grades == sorted(grades), (relationship, affection, grades)
            assert grades[-1] <= INTENSITY_INDEX[intensity_ceiling(profile)]

    def test_familiar_user_gets_more_than_medium_at_five_pokes(self):
        """有交情的人被连戳 5 次，应该比陌生人更值得真的发作。"""
        stranger = assess_trigger("", _profile("stranger", "normal"), recent_poke_count=5, mode="poke")
        familiar = assess_trigger("", _profile("familiar", "normal"), recent_poke_count=5, mode="poke")
        assert INTENSITY_INDEX[familiar.intensity] > INTENSITY_INDEX[stranger.intensity]

    def test_close_user_can_react_more_strongly(self):
        stranger = assess_trigger("", _profile("stranger", "normal"), recent_poke_count=5, mode="poke")
        close = assess_trigger("", _profile("close", "very_close"), recent_poke_count=5, mode="poke")
        assert INTENSITY_INDEX[close.intensity] > INTENSITY_INDEX[stranger.intensity]
        assert close.intensity == "very_strong"

    def test_poke_mode_does_not_apply_poke_logic_to_text_messages(self):
        assessment = assess_trigger("在吗", _profile("close", "very_close"), mode="direct")
        assert assessment.category == TRIGGER_NONE


class TestTextRepetition:
    """文字纠缠（同一句话反复刷）也必须被识别——不能只靠 poke 次数。"""

    @staticmethod
    def _history(lines: list[str], user_id: int = 2001):
        from services.context_store import ChatMessage

        return [
            ChatMessage(
                id=index,
                group_id=1,
                user_id=user_id,
                nickname="X",
                role="user",
                content=line,
                created_at="2026-09-16 01:00:00",
            )
            for index, line in enumerate(lines, start=1)
        ]

    def test_counts_identical_messages_from_same_user(self):
        from services.trigger_intensity import count_repeated_message

        history = self._history(["在吗", "在吗", "在吗"])
        assert count_repeated_message("在吗", history, 2001) == 4

    def test_ignores_other_users_repetition(self):
        from services.trigger_intensity import count_repeated_message

        history = self._history(["在吗", "在吗"], user_id=9999)
        assert count_repeated_message("在吗", history, 2001) == 1

    def test_ignores_assistant_repetition(self):
        from services.context_store import ChatMessage
        from services.trigger_intensity import count_repeated_message

        history = [
            ChatMessage(
                id=1,
                group_id=1,
                user_id=900000001,
                nickname="夜子",
                role="assistant",
                content="在吗",
                created_at="2026-09-16 01:00:00",
            )
        ]
        assert count_repeated_message("在吗", history, 2001) == 1

    def test_repeated_message_becomes_repetition_trigger(self):
        profile = _profile("close", "very_close")
        assessment = assess_trigger("在吗", profile, history=self._history(["在吗"] * 4), user_id=2001)
        assert assessment.category == TRIGGER_REPETITION
        assert assessment.intensity == "very_strong"

    def test_repeated_message_is_capped_for_stranger(self):
        profile = _profile("stranger", "normal")
        assessment = assess_trigger("在吗", profile, history=self._history(["在吗"] * 4), user_id=2001)
        assert assessment.category == TRIGGER_REPETITION
        assert assessment.intensity == "medium"

    def test_single_message_is_not_repetition(self):
        assessment = assess_trigger("在吗", _profile("close", "very_close"))
        assert assessment.category == TRIGGER_NONE

    def test_second_identical_ask_differs_by_relationship(self):
        """同一句催促出现两次：陌生人只是 medium，例外关系才升到 strong。

        这正是"关系决定她愿意为这个人动用多少情绪"的体现。"""
        history = self._history(["在吗"])
        stranger = assess_trigger("在吗", _profile("stranger", "normal"), history=history, user_id=2001)
        close = assess_trigger("在吗", _profile("close", "very_close"), history=history, user_id=2001)
        assert stranger.category == TRIGGER_REPETITION
        assert stranger.intensity == "medium"
        assert close.intensity == "strong"
        assert INTENSITY_INDEX[close.intensity] > INTENSITY_INDEX[stranger.intensity]

    def test_non_pressing_message_is_not_repetition(self):
        profile = _profile("close", "very_close")
        assessment = assess_trigger(
            "今天天气不错", profile, history=self._history(["今天天气不错"]), user_id=2001
        )
        assert assessment.category == TRIGGER_NONE


class TestIntensityBlock:
    def test_block_reports_category_intensity_and_reason(self):
        block = build_intensity_block(assess_trigger("累死了", _profile("close", "very_close")))
        assert "trigger: emotional_disclosure" in block
        assert "intensity: strong" in block
        assert "判定依据：" in block
        assert "上限 very_strong" in block

    def test_strong_guidance_permits_losing_composure(self):
        block = build_intensity_block(assess_trigger("你必须回答我", _profile("familiar", "normal")))
        assert "允许这条回复明显改变形态" in block
        assert "客服式礼貌" in block

    def test_weak_guidance_forbids_padding(self):
        block = build_intensity_block(
            assess_trigger("我最近看了一本小说", _profile("stranger", "normal"))
        )
        assert "弱触发" in block
        assert "不要额外加戏" in block

    def test_scheduled_mode_has_no_intensity_block(self):
        """定时问候没有具体提问者与当轮事件：强度不适用。"""
        assessment = assess_trigger("", _profile("stranger", "normal"))
        assert build_intensity_block(assessment, mode="scheduled") == ""

    def test_none_assessment_renders_empty(self):
        assert build_intensity_block(None) == ""

    def test_block_never_leaks_floats(self):
        for message in ("在吗", "你必须回答我", "累死了", "看书"):
            block = build_intensity_block(assess_trigger(message, _profile("close", "very_close")))
            assert "0." not in block
