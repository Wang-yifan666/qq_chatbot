"""services/relationship_service.py：确定性关系升级规则测试（v0.3.1）。"""

import pytest

from services.relationship_service import ACQUAINTANCE_THRESHOLD
from services.relationship_service import FAMILIAR_THRESHOLD
from services.relationship_service import VALID_EFFECTIVE_LEVELS
from services.relationship_service import calculate_base_level


class TestCalculateBaseLevel:
    @pytest.mark.parametrize("count", [0, 1, 4])
    def test_below_acquaintance_is_stranger(self, count):
        assert calculate_base_level(count) == "stranger"

    @pytest.mark.parametrize("count", [5, 6, 19])
    def test_between_thresholds_is_acquaintance(self, count):
        assert calculate_base_level(count) == "acquaintance"

    @pytest.mark.parametrize("count", [20, 21, 10_000])
    def test_at_or_above_familiar_is_familiar(self, count):
        assert calculate_base_level(count) == "familiar"

    def test_boundary_values(self):
        assert calculate_base_level(4) == "stranger"
        assert calculate_base_level(5) == "acquaintance"
        assert calculate_base_level(19) == "acquaintance"
        assert calculate_base_level(20) == "familiar"

    def test_never_returns_close(self):
        # close 只能由 CLOSE_USER_ID 运行时派生，计数规则永远不产生 close
        for count in range(-5, 300):
            assert calculate_base_level(count) != "close"
        assert calculate_base_level(-1) == "stranger"
        assert calculate_base_level(2**31) == "familiar"

    def test_thresholds_are_ordered(self):
        assert 0 < ACQUAINTANCE_THRESHOLD < FAMILIAR_THRESHOLD

    def test_close_exists_only_in_effective_levels(self):
        assert "close" in VALID_EFFECTIVE_LEVELS


class TestCloseTargetDefaults:
    def test_no_close_user_in_test_environment(self):
        # tests/conftest.py 设置 CLOSE_USER_ID=""，导入期解析应为 None
        from services.relationship_service import CLOSE_USER_ID
        from services.relationship_service import is_close_target

        assert CLOSE_USER_ID is None
        assert not is_close_target(123)
        assert not is_close_target(0)
