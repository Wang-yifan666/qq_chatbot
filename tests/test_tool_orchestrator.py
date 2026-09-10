"""services/tool_orchestrator.py：工具白名单 / Schema 校验 / 调用循环测试（v0.3.1）。

所有搜索调用使用 mock，绝不访问真实网络。
"""

import pytest

from services.tool_orchestrator import MAX_TOOL_CALLS_PER_TURN
from services.tool_orchestrator import MAX_TOOL_ROUNDS
from services.tool_orchestrator import WEB_SEARCH_TOOL_SCHEMA
from services.tool_orchestrator import RawCompletion
from services.tool_orchestrator import run_with_tools
from services.tool_orchestrator import validate_web_search_args


class TestValidateWebSearchArgs:
    def test_valid_query(self):
        assert validate_web_search_args('{"query": "STM32 DMA"}') == "STM32 DMA"

    def test_valid_with_extra_fields(self):
        assert validate_web_search_args('{"query": "q", "x": 1}') == "q"

    def test_blank_arguments_invalid(self):
        assert validate_web_search_args("") is None
        assert validate_web_search_args("   ") is None

    def test_not_json_invalid(self):
        assert validate_web_search_args("not json") is None

    def test_json_not_object_invalid(self):
        assert validate_web_search_args('["query"]') is None
        assert validate_web_search_args('"query"') is None

    def test_query_not_string_invalid(self):
        assert validate_web_search_args('{"query": 123}') is None
        assert validate_web_search_args('{"query": null}') is None
        assert validate_web_search_args('{"query": ["a"]}') is None

    def test_missing_query_invalid(self):
        assert validate_web_search_args("{}") is None

    def test_empty_or_blank_query_invalid(self):
        assert validate_web_search_args('{"query": ""}') is None
        assert validate_web_search_args('{"query": "   "}') is None

    def test_exactly_250_chars_ok_over_rejected(self):
        q250 = "a" * 250
        assert validate_web_search_args('{"query": "' + q250 + '"}') == q250
        assert validate_web_search_args('{"query": "' + "a" * 251 + '"}') is None


def _call_fn(completions, calls):
    """构造注入给 run_with_tools 的 call_fn：依次弹出预设 RawCompletion。"""

    async def _fn(messages, tools):
        calls.append((messages, tools))
        return completions.pop(0) if completions else None

    return _fn


class TestRunWithTools:
    async def test_plain_answer_no_tools(self):
        calls = []
        call_fn = _call_fn([RawCompletion(content="答案", tool_calls=[])], calls)
        answer = await run_with_tools(
            call_fn, [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer == "答案"
        assert len(calls) == 1

    async def test_search_executed_with_mock(self, monkeypatch):
        searched = []

        async def fake_search(query):
            searched.append(query)
            return [{"title": "t", "url": "u", "snippet": "s"}]

        monkeypatch.setattr("services.tool_orchestrator.search", fake_search)

        completions = [
            RawCompletion(
                content=None,
                tool_calls=[{"id": "c1", "name": "web_search", "arguments": '{"query": "DeepSeek"}'}],
            ),
            RawCompletion(content="最终回答", tool_calls=[]),
        ]
        calls = []
        answer = await run_with_tools(
            _call_fn(completions, calls), [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer == "最终回答"
        assert searched == ["DeepSeek"]
        # 第二轮请求的消息里包含 role=tool 的结果
        second_messages = calls[1][0]
        assert any(m.get("role") == "tool" for m in second_messages)

    async def test_unknown_tool_rejected_without_search(self, monkeypatch):
        searched = []

        async def fake_search(query):
            searched.append(query)
            return []

        monkeypatch.setattr("services.tool_orchestrator.search", fake_search)

        completions = [
            RawCompletion(
                content=None,
                tool_calls=[{"id": "c1", "name": "delete_everything", "arguments": "{}"}],
            ),
            RawCompletion(content="ok", tool_calls=[]),
        ]
        calls = []
        answer = await run_with_tools(
            _call_fn(completions, calls), [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer == "ok"
        assert searched == []
        tool_msgs = [m for m in calls[1][0] if m.get("role") == "tool"]
        assert tool_msgs and "拒绝执行" in tool_msgs[0]["content"]

    async def test_invalid_arguments_produce_tool_error(self, monkeypatch):
        searched = []

        async def fake_search(query):
            searched.append(query)
            return []

        monkeypatch.setattr("services.tool_orchestrator.search", fake_search)

        completions = [
            RawCompletion(
                content=None,
                tool_calls=[{"id": "c1", "name": "web_search", "arguments": "not json"}],
            ),
            RawCompletion(content="ok", tool_calls=[]),
        ]
        calls = []
        answer = await run_with_tools(
            _call_fn(completions, calls), [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer == "ok"
        assert searched == []
        tool_msgs = [m for m in calls[1][0] if m.get("role") == "tool"]
        assert tool_msgs and "参数非法" in tool_msgs[0]["content"]

    async def test_tool_call_count_limit(self, monkeypatch):
        searched = []

        async def fake_search(query):
            searched.append(query)
            return [{"title": "t", "url": "u", "snippet": "s"}]

        monkeypatch.setattr("services.tool_orchestrator.search", fake_search)

        # 一轮 3 个 tool_calls，超过 MAX_TOOL_CALLS_PER_TURN 的第三个拒绝执行
        completions = [
            RawCompletion(
                content=None,
                tool_calls=[
                    {"id": "c1", "name": "web_search", "arguments": '{"query": "a"}'},
                    {"id": "c2", "name": "web_search", "arguments": '{"query": "b"}'},
                    {"id": "c3", "name": "web_search", "arguments": '{"query": "c"}'},
                ],
            ),
            RawCompletion(content="ok", tool_calls=[]),
        ]
        calls = []
        answer = await run_with_tools(
            _call_fn(completions, calls), [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer == "ok"
        assert searched == ["a", "b"]
        tool_msgs = [m for m in calls[1][0] if m.get("role") == "tool"]
        assert any("已达上限" in m["content"] for m in tool_msgs)

    async def test_round_limit_then_plain_final_call(self, monkeypatch):
        async def fake_search(query):
            return [{"title": "t", "url": "u", "snippet": "s"}]

        monkeypatch.setattr("services.tool_orchestrator.search", fake_search)

        completions = [
            RawCompletion(
                content=None,
                tool_calls=[{"id": "c1", "name": "web_search", "arguments": '{"query": "x"}'}],
            )
            for _ in range(MAX_TOOL_ROUNDS)
        ]
        completions.append(RawCompletion(content="最终", tool_calls=[]))
        calls = []
        answer = await run_with_tools(
            _call_fn(completions, calls), [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer == "最终"
        assert len(calls) == MAX_TOOL_ROUNDS + 1
        # 最后一次请求不提供任何工具
        assert calls[-1][1] is None

    async def test_call_fn_none_returns_none(self):
        async def call_fn(messages, tools):
            return None

        answer = await run_with_tools(
            call_fn, [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL_SCHEMA]
        )
        assert answer is None

    async def test_multimodal_user_content_preserved_through_tool_rounds(self, monkeypatch):
        """带图片的 user content 在工具轮次中绝不能丢失 / 字符串化（v0.5 vision + tools）。"""
        from services.vision import VisionImage
        from services.vision import attach_images_to_last_user_message

        searched = []

        async def fake_search(query):
            searched.append(query)
            return [{"title": "t", "url": "u", "snippet": "s"}]

        monkeypatch.setattr("services.tool_orchestrator.search", fake_search)

        multimodal = attach_images_to_last_user_message(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "当前消息：\n这个多少钱"},
            ],
            [VisionImage(url="http://x/1.jpg", detail="auto")],
        )
        completions = [
            RawCompletion(
                content=None,
                tool_calls=[{"id": "c1", "name": "web_search", "arguments": '{"query": "价格"}'}],
            ),
            RawCompletion(content="大概 199。", tool_calls=[]),
        ]
        seen: list[list[dict]] = []

        async def call_fn(messages, tools):
            seen.append([dict(m) for m in messages])
            return completions.pop(0) if completions else None

        answer = await run_with_tools(call_fn, multimodal, [WEB_SEARCH_TOOL_SCHEMA])
        assert answer == "大概 199。"
        assert searched == ["价格"]
        # 每一轮请求里，最初的 user 消息都仍保留 image_url block
        for messages in seen:
            user_messages = [m for m in messages if m.get("role") == "user"]
            assert user_messages
            original = user_messages[0]["content"]
            assert isinstance(original, list)
            assert any(part.get("type") == "image_url" for part in original)

    async def test_max_tool_calls_constant_is_positive(self):
        assert MAX_TOOL_CALLS_PER_TURN > 0
        assert MAX_TOOL_ROUNDS > 0
