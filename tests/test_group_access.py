"""services/group_access.py：解析与 fail-closed 语义测试（v0.3.1）。"""

import importlib
import os

import pytest

from services.group_access import ALLOW_ALL_GROUPS
from services.group_access import ALLOWED_GROUP_IDS
from services.group_access import is_group_allowed
from services.group_access import parse_allowed_group_ids


class TestParseAllowedGroupIds:
    @pytest.mark.parametrize("raw", [None, "", "   ", "\t  \n"])
    def test_empty_values_fail_closed(self, raw):
        assert parse_allowed_group_ids(raw) == (frozenset(), False)

    def test_star_allows_all(self):
        assert parse_allowed_group_ids("*") == (frozenset(), True)
        assert parse_allowed_group_ids("  *  ") == (frozenset(), True)

    def test_single_id(self):
        assert parse_allowed_group_ids("123") == (frozenset({123}), False)

    def test_two_ids(self):
        ids, allow_all = parse_allowed_group_ids("123,456")
        assert ids == frozenset({123, 456})
        assert allow_all is False

    def test_spaces_around_commas(self):
        ids, allow_all = parse_allowed_group_ids("123, 456")
        assert ids == frozenset({123, 456})
        assert allow_all is False

    def test_empty_segments_ignored(self):
        ids, allow_all = parse_allowed_group_ids("123,,456")
        assert ids == frozenset({123, 456})
        assert allow_all is False

    def test_duplicate_ids_deduplicated(self):
        ids, allow_all = parse_allowed_group_ids("123,123, 123")
        assert ids == frozenset({123})
        assert allow_all is False

    @pytest.mark.parametrize(
        "raw",
        ["0", "-1", "abc", "123,abc", "12.5", "1e3", "123, 0", "*,123", "123,*", "+1"],
    )
    def test_invalid_values_raise_value_error(self, raw):
        with pytest.raises(ValueError):
            parse_allowed_group_ids(raw)

    def test_error_message_does_not_leak_config_value(self):
        with pytest.raises(ValueError) as excinfo:
            parse_allowed_group_ids("123,abc")
        message = str(excinfo.value)
        assert "123" not in message
        assert "abc" not in message


class TestIsGroupAllowed:
    def test_allow_all(self, monkeypatch):
        monkeypatch.setattr("services.group_access.ALLOW_ALL_GROUPS", True)
        monkeypatch.setattr("services.group_access.ALLOWED_GROUP_IDS", frozenset())
        assert is_group_allowed(123)
        assert is_group_allowed(999999)
        assert is_group_allowed(0)

    def test_only_listed_groups(self, monkeypatch):
        monkeypatch.setattr("services.group_access.ALLOW_ALL_GROUPS", False)
        monkeypatch.setattr("services.group_access.ALLOWED_GROUP_IDS", frozenset({123, 456}))
        assert is_group_allowed(123)
        assert is_group_allowed(456)
        assert not is_group_allowed(789)
        assert not is_group_allowed(0)
        assert not is_group_allowed(-1)

    def test_fail_closed_when_empty(self, monkeypatch):
        monkeypatch.setattr("services.group_access.ALLOW_ALL_GROUPS", False)
        monkeypatch.setattr("services.group_access.ALLOWED_GROUP_IDS", frozenset())
        assert not is_group_allowed(123)
        assert not is_group_allowed(0)

    def test_module_defaults_match_conftest_env(self):
        # tests/conftest.py 设置 ALLOWED_GROUP_IDS="111, 222"
        assert ALLOW_ALL_GROUPS is False
        assert ALLOWED_GROUP_IDS == frozenset({111, 222})


class TestImportTimeValidation:
    def test_invalid_env_raises_at_import_time(self, monkeypatch):
        import services.group_access as group_access

        monkeypatch.setenv("ALLOWED_GROUP_IDS", "111,abc")
        with pytest.raises(ValueError):
            importlib.reload(group_access)
        # 手动恢复环境并重载，保证模块与后续测试处于合法状态
        # （importlib.reload 复用同一模块对象，插件绑定的函数引用不受影响）
        monkeypatch.undo()
        importlib.reload(group_access)
        assert group_access.is_group_allowed(111)
        assert not group_access.is_group_allowed(333)
