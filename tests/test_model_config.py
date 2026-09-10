"""模型配置基线测试（v0.5）：DeepSeek 默认模型 = deepseek-flash。"""

from services.deepseek import DEFAULT_MODEL as DEEPSEEK_DEFAULT_MODEL
from services.llm_client import VISION_CAPABLE_MODELS
from services.llm_client import provider_supports_vision
from services.zhipu import DEFAULT_MODEL as GLM_DEFAULT_MODEL


def test_deepseek_default_model_is_flash():
    # conftest 不设置 DEEPSEEK_MODEL → 默认必须是当前推荐的 deepseek-flash
    assert DEEPSEEK_DEFAULT_MODEL == "deepseek-flash"


def test_glm_default_unchanged():
    assert GLM_DEFAULT_MODEL == "glm-4.7-flash"


def test_vision_capable_models_contain_deepseek_flash():
    assert "deepseek-flash" in VISION_CAPABLE_MODELS
    assert "deepseek-v4-flash-vision-exp" not in VISION_CAPABLE_MODELS


def test_default_deepseek_is_vision_capable():
    assert provider_supports_vision("deepseek", None) is True
