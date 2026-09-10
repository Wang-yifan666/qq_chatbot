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


class TestTriggerSnippet:
    def _msg(self, role, content):
        return SimpleNamespace(role=role, content=content)

    def test_snippet_takes_recent_user_messages_only(self):
        history = [
            self._msg("user", "第一条"),
            self._msg("assistant", "机器人说的话不算"),
            self._msg("user", "第二条"),
            self._msg("user", "第三条"),
        ]
        assert ambient.build_trigger_snippet(history) == "第一条\n第二条\n第三条"

    def test_long_then_short_keeps_long_content(self):
        """A 长消息 + B「哈哈」：安静期从 B 重算，但片段仍包含 A 的内容。"""
        history = [
            self._msg("user", "今天老板让我改了五遍，人都麻了"),
            self._msg("user", "哈哈"),
        ]
        snippet = ambient.build_trigger_snippet(history)
        assert "今天老板让我改了五遍" in snippet
        assert "哈哈" in snippet

    def test_snippet_bounded(self):
        history = [self._msg("user", "x" * 300) for _ in range(5)]
        snippet = ambient.build_trigger_snippet(history)
        assert len(snippet) <= ambient.SNIPPET_MAX_CHARS


class TestOnGroupMessageDebounce:
    async def test_short_message_still_resets_timer(self, monkeypatch):
        """activity 语义：「哈哈」不单独作为 candidate，但必须重置 quiet timer。"""
        monkeypatch.setattr(ambient, "AMBIENT_MIN_MESSAGE_CHARS", 4)
        scheduled: list[int] = []

        def fake_schedule(group_id):
            scheduled.append(group_id)

        monkeypatch.setattr(ambient, "_schedule_decision", fake_schedule)
        await ambient.on_group_message(FakeEvent(721, 1001, 999, "哈"))
        assert scheduled == [721], "短消息也必须刷新 quiet timer"

    async def test_normal_message_scheduled(self, monkeypatch):
        scheduled: list[int] = []

        def fake_schedule(group_id):
            scheduled.append(group_id)

        monkeypatch.setattr(ambient, "_schedule_decision", fake_schedule)
        await ambient.on_group_message(FakeEvent(722, 1001, 999, "这个 DMA 配置到底怎么弄"))
        assert scheduled == [722]

    async def test_new_message_cancels_previous_debounce(self):
        ambient._schedule_decision(723)
        state = get_group_conversation_state(723)
        first = state.ambient_pending_task
        assert first is not None
        ambient._schedule_decision(723)
        second = state.ambient_pending_task
        assert second is not first
        await asyncio.sleep(0.02)
        assert first.cancelled()
        await _drain_pending(723)

    async def test_long_then_short_resets_timer(self, monkeypatch):
        """A 长消息 → 8 秒后 B「哈哈」：quiet timer 必须从 B 的时间重新计算。"""
        scheduled: list[int] = []

        def fake_schedule(group_id):
            scheduled.append(group_id)

        monkeypatch.setattr(ambient, "_schedule_decision", fake_schedule)
        await ambient.on_group_message(FakeEvent(724, 1001, 999, "今天老板让我改了五遍"))
        await ambient.on_group_message(FakeEvent(724, 1002, 999, "哈哈"))
        assert scheduled == [724, 724], "两条消息各自重置 timer"


class TestAfterQuiet:
    def _long_history(self):
        return [SimpleNamespace(role="user", content="这个 DMA 配置到底怎么弄")]

    async def test_short_snippet_stays_silent_without_llm(self, monkeypatch):
        """安静期结束后片段仍是低信息（只有“哈哈”）→ 不调 decision LLM。"""
        monkeypatch.setattr(ambient, "AMBIENT_MIN_MESSAGE_CHARS", 4)
        decisions: list = []

        async def fake_history(group_id, limit):
            return [SimpleNamespace(role="user", content="哈哈")]

        monkeypatch.setattr(ambient, "get_recent_messages", fake_history)

        async def fake_decide(group_id, snippet):
            decisions.append(snippet)
            return True

        monkeypatch.setattr(ambient, "_decide", fake_decide)
        await ambient._after_quiet(731)
        assert decisions == []

    async def test_decision_false_ends_silently(self, monkeypatch):
        monkeypatch.setattr(ambient, "_decide", _async_return(False))
        monkeypatch.setattr(ambient, "_ambient_gate_allows", _async_return(True))
        monkeypatch.setattr(
            ambient, "get_recent_messages", _async_return(self._long_history())
        )
        generated: list = []

        async def fake_generate(group_id, chunk):
            generated.append((group_id, chunk))

        monkeypatch.setattr(ambient, "_generate_and_send", fake_generate)
        await ambient._after_quiet(732)
        assert generated == []

    async def test_decision_true_generates(self, monkeypatch):
        monkeypatch.setattr(ambient, "_decide", _async_return(True))
        monkeypatch.setattr(ambient, "_ambient_gate_allows", _async_return(True))
        monkeypatch.setattr(
            ambient, "get_recent_messages", _async_return(self._long_history())
        )
        generated: list = []

        async def fake_generate(group_id, chunk):
            generated.append((group_id, chunk))

        monkeypatch.setattr(ambient, "_generate_and_send", fake_generate)
        await ambient._after_quiet(733)
        assert generated == [(733, "这个 DMA 配置到底怎么弄")]

    async def test_gate_blocks_before_decision(self, monkeypatch):
        decisions: list = []
        monkeypatch.setattr(ambient, "_ambient_gate_allows", _async_return(False))
        monkeypatch.setattr(
            ambient, "get_recent_messages", _async_return(self._long_history())
        )

        async def fake_decide(group_id, chunk):
            decisions.append(group_id)
            return True

        monkeypatch.setattr(ambient, "_decide", fake_decide)
        await ambient._after_quiet(734)
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
