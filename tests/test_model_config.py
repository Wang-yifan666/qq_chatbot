"""模型配置基线测试（v0.5 → v0.7）：默认 deepseek-flash + alias 归一化。"""

from services.deepseek import DEFAULT_MODEL as DEEPSEEK_DEFAULT_MODEL
from services.llm_client import VISION_CAPABLE_MODELS
from services.llm_client import provider_supports_vision
from services.model_registry import is_vision_capable
from services.model_registry import normalize_model_name
from services.zhipu import DEFAULT_MODEL as GLM_DEFAULT_MODEL


def test_deepseek_default_model_is_flash():
    # conftest 不设置 DEEPSEEK_MODEL → 默认必须是当前推荐的 deepseek-flash
    assert DEEPSEEK_DEFAULT_MODEL == "deepseek-flash"


def test_glm_default_unchanged():
    assert GLM_DEFAULT_MODEL == "glm-4.7-flash"


def test_vision_capable_models_contain_deepseek_flash():
    assert "deepseek-flash" in VISION_CAPABLE_MODELS
    # v0.7：能力表只存 canonical 名，alias 由 normalize_model_name 负责展开
    assert "deepseek-v4-flash-vision-exp" not in VISION_CAPABLE_MODELS


def test_default_deepseek_is_vision_capable():
    assert provider_supports_vision("deepseek", None) is True


class TestModelAliasNormalization:
    """v0.7：alias 归一化集中实现，不再多个文件各自维护一份判断逻辑。"""

    def test_aliases_map_to_canonical(self):
        assert normalize_model_name("deepseek-v4-flash") == "deepseek-flash"
        assert normalize_model_name("deepseek-v4-flash-vision-exp") == "deepseek-flash"
        assert normalize_model_name("deepseek-flash") == "deepseek-flash"
        assert normalize_model_name("  DEEPSEEK-V4-FLASH  ") == "deepseek-flash"

    def test_unknown_and_empty_models_are_not_rewritten(self):
        assert normalize_model_name("deepseek-chat") == "deepseek-chat"
        assert normalize_model_name("glm-4.7-flash") == "glm-4.7-flash"
        assert normalize_model_name(None) == ""
        assert normalize_model_name("") == ""

    def test_aliases_are_vision_capable(self):
        assert is_vision_capable("deepseek-v4-flash-vision-exp") is True
        assert is_vision_capable("deepseek-flash") is True
        assert is_vision_capable("deepseek-chat") is False

    def test_deepseek_model_env_alias_is_normalized(self, monkeypatch):
        """DEEPSEEK_MODEL 写历史 alias 时也必须被归一化成 canonical 名。"""
        import importlib

        import services.deepseek as deepseek

        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp")
        importlib.reload(deepseek)
        try:
            assert deepseek.DEFAULT_MODEL == "deepseek-flash"
        finally:
            monkeypatch.undo()
            importlib.reload(deepseek)
