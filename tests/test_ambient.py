"""services/ambient.py：AMBIENT 主动插话 MVP 测试（v0.4）。

全部 mock：不连 QQ、不调真实 LLM；防抖任务用短 sleep / 手动 gather 清理。
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

import services.ambient as ambient
from services.group_conversation import get_group_conversation_state

# 保证防抖测试不依赖真实 10 秒等待
ambient.AMBIENT_QUIET_SECONDS = 0.01


class FakeEvent:
    def __init__(self, group_id: int, user_id: int, self_id: int, text: str):
        self.group_id = group_id
        self.user_id = user_id
        self.self_id = self_id
        self._text = text
        self.plaintext_calls = 0

    def get_plaintext(self) -> str:
        self.plaintext_calls += 1
        return self._text


async def _drain_pending(group_id: int) -> None:
    """等待 / 清理某群残留的防抖任务，避免任务泄漏到其它测试。"""
    state = get_group_conversation_state(group_id)
    pending = state.ambient_pending_task
    if pending is not None:
        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass
    state.ambient_pending_task = None


class TestParseShouldReply:
    def test_true(self):
        assert ambient.parse_should_reply('{"should_reply": true}') is True

    def test_false(self):
        assert ambient.parse_should_reply('{"should_reply": false}') is False

    def test_fenced_json(self):
        text = '```json\n{"should_reply": true}\n```'
        assert ambient.parse_should_reply(text) is True

    def test_garbage_is_false(self):
        assert ambient.parse_should_reply("随便说点什么") is False
        assert ambient.parse_should_reply('{"should_reply": "yes"}') is False

    def test_empty_is_false(self):
        assert ambient.parse_should_reply(None) is False
        assert ambient.parse_should_reply("") is False


class TestDecisionMessages:
    def test_decision_prompt_structure(self):
        history = []
        messages = ambient.build_ambient_decision_messages(
            history, "STM32 的 DMA 怎么配", runtime_state="RUNTIME"
        )
        system = messages[0]["content"]
        assert system.startswith(ambient.STATIC_SYSTEM_PROMPT)
        assert "conversation_mode: ambient_decision" in system
        assert "should_reply" in system
        # 决策规则只问“该不该说话”，不含性格形容词指令
        for word in ("活泼", "傲娇", "毒舌", "可爱", "温柔"):
            assert word not in ambient.AMBIENT_DECISION_RULES
        # 没有伪造具体提问者（direct 专用格式不得出现）
        joined = "\n".join(m["content"] for m in messages)
        assert "current_user_id: " not in joined
        assert "当前提问者 user_id=" not in joined
        assert "STM32 的 DMA 怎么配" in joined


class TestAmbientGate:
    async def test_hourly_cap_blocks(self, monkeypatch):
        monkeypatch.setattr(ambient, "AMBIENT_MAX_PER_HOUR", 2)
        state = get_group_conversation_state(711)
        now = time.monotonic()
        state.ambient_sent_at = [now, now + 1.0]

        async def active(group_id, minutes):
            return False

        monkeypatch.setattr(ambient, "has_recent_bot_message", active)
        assert not await ambient._ambient_gate_allows(711)

    async def test_cooldown_blocks(self, monkeypatch):
        monkeypatch.setattr(ambient, "AMBIENT_MAX_PER_HOUR", 5)
        get_group_conversation_state(712).ambient_sent_at = []

        async def active(group_id, minutes):
            return True

        monkeypatch.setattr(ambient, "has_recent_bot_message", active)
        assert not await ambient._ambient_gate_allows(712)

    async def test_gate_allows_when_quiet(self, monkeypatch):
        monkeypatch.setattr(ambient, "AMBIENT_MAX_PER_HOUR", 5)
        get_group_conversation_state(713).ambient_sent_at = []

        async def active(group_id, minutes):
            return False

        monkeypatch.setattr(ambient, "has_recent_bot_message", active)
        assert await ambient._ambient_gate_allows(713)

    async def test_max_per_hour_zero_disables(self, monkeypatch):
        monkeypatch.setattr(ambient, "AMBIENT_MAX_PER_HOUR", 0)
        assert not await ambient._ambient_gate_allows(714)


class TestOnGroupMessageDebounce:
    async def test_too_short_message_not_scheduled(self, monkeypatch):
        monkeypatch.setattr(ambient, "AMBIENT_MIN_MESSAGE_CHARS", 4)
        scheduled: list[str] = []

        def fake_schedule(group_id, chunk):
            scheduled.append(chunk)

        monkeypatch.setattr(ambient, "_schedule_decision", fake_schedule)
        await ambient.on_group_message(FakeEvent(721, 1001, 999, "哈"))
        assert scheduled == []

    async def test_normal_message_scheduled(self, monkeypatch):
        monkeypatch.setattr(ambient, "AMBIENT_MIN_MESSAGE_CHARS", 4)
        scheduled: list[str] = []

        def fake_schedule(group_id, chunk):
            scheduled.append(chunk)

        monkeypatch.setattr(ambient, "_schedule_decision", fake_schedule)
        await ambient.on_group_message(FakeEvent(722, 1001, 999, "这个 DMA 配置到底怎么弄"))
        assert scheduled == ["这个 DMA 配置到底怎么弄"]

    async def test_new_message_cancels_previous_debounce(self):
        ambient._schedule_decision(723, "第一条消息")
        state = get_group_conversation_state(723)
        first = state.ambient_pending_task
        assert first is not None
        ambient._schedule_decision(723, "第二条消息")
        second = state.ambient_pending_task
        assert second is not first
        await asyncio.sleep(0.02)
        assert first.cancelled()
        await _drain_pending(723)


class TestAfterQuiet:
    async def test_decision_false_ends_silently(self, monkeypatch):
        monkeypatch.setattr(ambient, "_decide", _async_return(False))
        monkeypatch.setattr(ambient, "_ambient_gate_allows", _async_return(True))
        generated: list = []

        async def fake_generate(group_id, chunk):
            generated.append((group_id, chunk))

        monkeypatch.setattr(ambient, "_generate_and_send", fake_generate)
        await ambient._after_quiet(731, "触发片段")
        assert generated == []

    async def test_decision_true_generates(self, monkeypatch):
        monkeypatch.setattr(ambient, "_decide", _async_return(True))
        monkeypatch.setattr(ambient, "_ambient_gate_allows", _async_return(True))
        generated: list = []

        async def fake_generate(group_id, chunk):
            generated.append((group_id, chunk))

        monkeypatch.setattr(ambient, "_generate_and_send", fake_generate)
        await ambient._after_quiet(732, "触发片段")
        assert generated == [(732, "触发片段")]

    async def test_gate_blocks_before_decision(self, monkeypatch):
        decisions: list = []
        monkeypatch.setattr(ambient, "_ambient_gate_allows", _async_return(False))

        async def fake_decide(group_id, chunk):
            decisions.append(group_id)
            return True

        monkeypatch.setattr(ambient, "_decide", fake_decide)
        await ambient._after_quiet(733, "触发片段")
        assert decisions == []


class TestGenerateAndSend:
    async def test_full_pipeline_uses_ambient_mode(self, monkeypatch):
        captured: dict = {}
        sent: list = []
        saved: list = []

        async def fake_history(group_id, limit):
            return []

        monkeypatch.setattr(ambient, "get_recent_messages", fake_history)
        monkeypatch.setattr(ambient, "PERSONA_RAG_ENABLED", False)

        def fake_build(*args, **kwargs):
            captured.update(kwargs)
            captured["args"] = args
            return [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

        monkeypatch.setattr(ambient, "build_messages", fake_build)

        async def fake_llm(messages, tools=None):
            captured["tools"] = tools
            return ("这写法能跑，但设计得不好。", "deepseek")

        monkeypatch.setattr(ambient, "ask_with_fallback", fake_llm)

        bot = SimpleNamespace(self_id="999")
        monkeypatch.setattr(ambient, "get_onebot_bot", lambda: bot)

        async def send(bot, group_id, text):
            sent.append((group_id, text))
            return True

        monkeypatch.setattr(ambient, "send_group_message", send)

        async def save(group_id, self_id, text):
            saved.append((group_id, self_id, text))
            return True

        monkeypatch.setattr(ambient, "save_assistant_message", save)
        monkeypatch.setattr(ambient, "AMBIENT_MAX_PER_HOUR", 3)

        await ambient._generate_and_send(741, "触发片段")

        assert captured["conversation_mode"] == "ambient"
        assert captured["ambient_context"] == "触发片段"
        assert captured["args"][0] is None  # 没有伪造 current_user
        assert captured["tools"] is None  # AMBIENT 默认无工具
        assert sent == [(741, "这写法能跑，但设计得不好。")]
        assert saved == [(741, "999", "这写法能跑，但设计得不好。")]
        state = get_group_conversation_state(741)
        assert state.ambient_hourly_count(time.monotonic()) >= 1


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner
