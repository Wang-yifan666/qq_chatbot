"""services/llm_client.py：capability-aware fallback 测试（v0.5）。"""

import pytest

import services.llm_client as llm
from services.vision import VisionImage
from services.vision import attach_images_to_last_user_message


def _vision_messages() -> list[dict]:
    base = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "当前提问者 user_id=1\n当前消息：\n看这张图"},
    ]
    return attach_images_to_last_user_message(base, [VisionImage(url="http://x/1.jpg", detail="auto")])


class TestVisionCapability:
    def test_deepseek_flash_is_vision_capable(self):
        assert llm.provider_supports_vision("deepseek", None) is True
        assert llm.provider_supports_vision("deepseek", "deepseek-flash") is True
        assert llm.provider_supports_vision("deepseek", "DEEPSEEK-FLASH") is True

    def test_unknown_models_are_text_only(self):
        assert llm.provider_supports_vision("deepseek", "deepseek-chat") is False
        assert llm.provider_supports_vision("deepseek", "deepseek-v4-flash") is False
        assert llm.provider_supports_vision("zhipu", None) is False
        assert llm.provider_supports_vision("zhipu", "glm-4.7-flash") is False

    def test_effective_model(self, monkeypatch):
        monkeypatch.setattr(llm, "DEEPSEEK_DEFAULT_MODEL", "deepseek-flash")
        monkeypatch.setattr(llm, "GLM_DEFAULT_MODEL", "glm-4.7-flash")
        assert llm.effective_model("deepseek", None) == "deepseek-flash"
        assert llm.effective_model("deepseek", "deepseek-chat") == "deepseek-chat"
        assert llm.effective_model("zhipu", None) == "glm-4.7-flash"


class TestCandidates:
    def _setup(self, monkeypatch, primary, primary_model, fallback, fallback_model):
        monkeypatch.setattr(llm, "AI_PROVIDER", primary)
        monkeypatch.setattr(llm, "AI_MODEL", primary_model)
        monkeypatch.setattr(llm, "AI_FALLBACK", fallback)
        monkeypatch.setattr(llm, "AI_FALLBACK_MODEL", fallback_model)
        monkeypatch.setattr(llm, "DEEPSEEK_DEFAULT_MODEL", "deepseek-flash")
        monkeypatch.setattr(llm, "GLM_DEFAULT_MODEL", "glm-4.7-flash")

    def test_text_request_keeps_all_candidates(self, monkeypatch):
        self._setup(monkeypatch, "deepseek", None, "zhipu", None)
        assert llm._candidates(require_vision=False) == [("deepseek", None), ("zhipu", None)]

    def test_vision_request_filters_text_only_candidates(self, monkeypatch):
        self._setup(monkeypatch, "deepseek", None, "zhipu", None)
        assert llm._candidates(require_vision=True) == [("deepseek", None)]

    def test_vision_skips_text_only_primary_uses_capable_fallback(self, monkeypatch):
        # 主模型 deepseek-chat（text-only）→ 视觉请求直接跳过主、用 deepseek-flash 备用
        self._setup(monkeypatch, "deepseek", "deepseek-chat", "deepseek", "deepseek-flash")
        assert llm._candidates(require_vision=True) == [("deepseek", "deepseek-flash")]

    def test_vision_no_capable_candidate(self, monkeypatch):
        self._setup(monkeypatch, "zhipu", None, "zhipu", "glm-4.7-flash")
        assert llm._candidates(require_vision=True) == []


class TestAskWithFallbackRouting:
    async def test_text_request_unchanged_behavior(self, monkeypatch):
        """纯文本：主成功不再调备用；主失败降级备用（v0.4 行为不变）。"""
        calls: list[tuple[str, str | None]] = []

        async def fake_ask(provider, messages, model=None):
            calls.append((provider, model))
            return None if provider == "deepseek" else "fallback-answer"

        monkeypatch.setattr(llm, "ask", fake_ask)
        monkeypatch.setattr(llm, "AI_PROVIDER", "deepseek")
        monkeypatch.setattr(llm, "AI_MODEL", None)
        monkeypatch.setattr(llm, "AI_FALLBACK", "zhipu")
        monkeypatch.setattr(llm, "AI_FALLBACK_MODEL", None)

        answer, used = await llm.ask_with_fallback([{"role": "user", "content": "hi"}])
        assert answer == "fallback-answer"
        assert used == "zhipu"
        assert calls == [("deepseek", None), ("zhipu", None)]

    async def test_vision_never_sent_to_text_only_candidate(self, monkeypatch):
        """视觉请求绝不发给 text-only 候选：主是 chat 时只调 flash 备用。"""
        calls: list[tuple[str, str | None]] = []

        async def fake_ask(provider, messages, model=None):
            calls.append((provider, model))
            return "ok"

        monkeypatch.setattr(llm, "ask", fake_ask)
        monkeypatch.setattr(llm, "AI_PROVIDER", "deepseek")
        monkeypatch.setattr(llm, "AI_MODEL", "deepseek-chat")
        monkeypatch.setattr(llm, "AI_FALLBACK", "deepseek")
        monkeypatch.setattr(llm, "AI_FALLBACK_MODEL", "deepseek-flash")
        monkeypatch.setattr(llm, "DEEPSEEK_DEFAULT_MODEL", "deepseek-flash")

        answer, used = await llm.ask_with_fallback(_vision_messages(), require_vision=True)
        assert answer == "ok"
        assert used == "deepseek"
        assert calls == [("deepseek", "deepseek-flash")], "text-only 主模型不应被调用"

    async def test_vision_no_capable_candidate_returns_none(self, monkeypatch):
        calls: list = []

        async def fake_ask(provider, messages, model=None):
            calls.append((provider, model))
            return "should-not-happen"

        monkeypatch.setattr(llm, "ask", fake_ask)
        monkeypatch.setattr(llm, "AI_PROVIDER", "zhipu")
        monkeypatch.setattr(llm, "AI_MODEL", None)
        monkeypatch.setattr(llm, "AI_FALLBACK", "zhipu")
        monkeypatch.setattr(llm, "AI_FALLBACK_MODEL", None)

        answer, used = await llm.ask_with_fallback(_vision_messages(), require_vision=True)
        assert answer is None
        assert calls == []

    async def test_vision_all_candidates_fail_returns_none(self, monkeypatch):
        async def fake_ask(provider, messages, model=None):
            return None

        monkeypatch.setattr(llm, "ask", fake_ask)
        monkeypatch.setattr(llm, "AI_PROVIDER", "deepseek")
        monkeypatch.setattr(llm, "AI_MODEL", None)
        monkeypatch.setattr(llm, "AI_FALLBACK", "deepseek")
        monkeypatch.setattr(llm, "AI_FALLBACK_MODEL", "deepseek-flash")
        monkeypatch.setattr(llm, "DEEPSEEK_DEFAULT_MODEL", "deepseek-flash")

        answer, _ = await llm.ask_with_fallback(_vision_messages(), require_vision=True)
        assert answer is None
