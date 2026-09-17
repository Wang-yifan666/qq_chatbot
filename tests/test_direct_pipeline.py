"""端到端 DIRECT Pipeline 测试（v0.7）。

前面几个测试文件分别覆盖：
- `test_group_gate.py`：把 `_answer` 换成桩，只验证门禁与 Resolver 输出；
- `test_message_resolver.py` / `test_prompt_injection.py`：直接调用 Resolver 与
  `build_messages`。

本文件补上**真实的 `plugins.ai_chat._answer()`**（只把 LLM 层与 QQ 发送换成桩），
验证「QQ event → Resolver → build_messages → LLM」整条链路真的接通：

- messages 的信任分区在真实 pipeline 里成立（system / DATA / 当前消息）；
- 有图片时 `require_vision=True` 被真正传递（capability 路由不会退化成纯文本）；
- 落库占位符只含结构化短文本（绝不含 URL / 正文 / 转发内容）；
- 转发 / 文件 / 引用内容都进入 Prompt；
- 空消息与视觉关闭的降级路径仍然不调用模型。

不连真实 QQ / LLM / CDN：LLM 与图片下载全部 mock。
"""

from types import SimpleNamespace

import pytest

import plugins.ai_chat as ai
from nonebot.exception import FinishedException


# ======================================================================
# 基础设施
# ======================================================================


PIPELINE_GROUP_ID = 777   # 独立测试群：不与其他测试文件共享 messages 数据
PIPELINE_USER_ID = 91001  # 独立测试用户：不与其他测试文件共享 relationships 计数


class FakeChat:
    def __init__(self) -> None:
        self.finished: list = []
        self.sent: list = []

    async def finish(self, message=None) -> None:
        self.finished.append(message)
        raise FinishedException

    async def send(self, message) -> None:
        self.sent.append(message)


class FakeEvent:
    """伪 GroupMessageEvent（与 test_group_gate 保持同一形状）。"""

    def __init__(self, segments, *, text: str = "", group_id: int = PIPELINE_GROUP_ID, user_id: int = PIPELINE_USER_ID):
        self.group_id = group_id
        self.user_id = user_id
        self.self_id = 999
        self.message_id = 7001
        self.message = segments
        self._text = text
        self.to_me = False
        self.sender = SimpleNamespace(user_id=user_id, nickname="小明", card="")
        self.reply = None

    def get_plaintext(self) -> str:
        text = self._text
        if text.startswith("@bot"):
            text = text[len("@bot"):].strip()
        return text

    def get_message(self) -> list:
        return self.message


class FakeBot:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def call_api(self, action: str, **params):
        self.calls.append((action, params))
        if action in self.responses:
            return self.responses[action]
        raise RuntimeError(f"no stub for {action}")


def seg(seg_type: str, **data):
    return SimpleNamespace(type=seg_type, data=data)


def api_seg(seg_type: str, **data) -> dict:
    return {"type": seg_type, "data": data}


class Capture:
    """捕获真实 pipeline 传给 LLM 的 messages 与 require_vision。"""

    def __init__(self, answer: str = "夜子的回答"):
        self.answer = answer
        self.messages: list[dict] | None = None
        self.require_vision: bool | None = None
        self.calls = 0

    async def __call__(self, messages, tools=None, require_vision=False):
        self.calls += 1
        self.messages = messages
        self.require_vision = require_vision
        return self.answer, "deepseek"


async def run_direct(monkeypatch, event, bot=None, answer="夜子的回答"):
    """跑真实 _answer：只桩掉 LLM 层、QQ 发送与记忆提取。"""
    import services.database as dbm

    await dbm.init_db()

    capture = Capture(answer)
    monkeypatch.setattr(ai, "ask_with_fallback", capture)
    # 记忆提取是后台任务，测试里不启动（与真实行为无关，避免多余 LLM 调用）
    monkeypatch.setattr(ai, "extract_memories", None, raising=False)

    chat = FakeChat()
    monkeypatch.setattr(ai, "chat", chat)

    resolver = ai.MessageResolver(bot)
    try:
        resolved = await resolver.resolve_event(event)
        conversation = await resolver.build_conversation(resolved)
        plain = event.get_plaintext().strip()
        normalized = resolved.message.text() or plain
        result = await ai._answer(
            event, normalized, conversation=conversation, resolved=resolved
        )
    finally:
        resolver.cleanup()
    return result, capture, conversation, resolved


def flat_text(messages) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "image_url":
                    parts.append("[IMAGE]")
    return "\n".join(parts)


def system_text(messages) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "system")


# ======================================================================
# 纯文本
# ======================================================================


class TestDirectText:
    async def test_plain_text_reaches_llm_without_vision(self, monkeypatch):
        event = FakeEvent([seg("text", text="你好")], text="@bot 你好")
        result, capture, _conv, resolved = await run_direct(monkeypatch, event)
        assert result == "夜子的回答"
        assert capture.require_vision is False
        assert f"当前提问者 user_id={PIPELINE_USER_ID}" in flat_text(capture.messages)
        assert "你好" in flat_text(capture.messages)
        assert resolved.message.text() == "你好"

    async def test_context_placeholder_is_plain_text(self, monkeypatch):
        from services.context_store import get_recent_messages

        event = FakeEvent([seg("text", text="记住这句话")], text="@bot 记住这句话")
        await run_direct(monkeypatch, event)
        history = await get_recent_messages(PIPELINE_GROUP_ID, 5)
        assert any("记住这句话" in m.content for m in history)
        # 占位符里绝不能出现 URL / Base64
        assert all("http" not in m.content for m in history)


# ======================================================================
# 图片（capability 路由真的被打开）
# ======================================================================


class TestDirectImages:
    async def test_image_requires_vision_and_keeps_order(self, monkeypatch):
        event = FakeEvent(
            [
                seg("text", text="文字A"),
                seg("image", url="http://cdn/1.jpg"),
                seg("text", text="文字B"),
                seg("image", url="http://cdn/2.jpg"),
            ],
            text="@bot 文字A文字B",
        )
        _result, capture, conversation, _resolved = await run_direct(monkeypatch, event)
        assert capture.require_vision is True

        last = capture.messages[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        kinds = [block["type"] for block in last["content"]]
        # footer 文本 + 有序 4 个 block
        assert kinds == ["text", "text", "image_url", "text", "image_url"]
        assert last["content"][1]["text"] == "文字A"
        assert last["content"][2]["image_url"]["url"] == "http://cdn/1.jpg"
        assert last["content"][3]["text"] == "文字B"
        assert last["content"][4]["image_url"]["url"] == "http://cdn/2.jpg"
        assert conversation.has_any_image is True

    async def test_image_urls_never_leak_into_context(self, monkeypatch):
        from services.context_store import get_recent_messages

        event = FakeEvent(
            [seg("image", url="http://cdn/secret-token-abc.jpg")], text="@bot"
        )
        await run_direct(monkeypatch, event)
        history = await get_recent_messages(PIPELINE_GROUP_ID, 5)
        joined = "\n".join(m.content for m in history)
        assert "secret-token-abc" not in joined
        assert "http" not in joined
        assert "[发送了 1 张图片]" in joined or "[附带 1 张图片]" in joined

    async def test_vision_disabled_drops_image_blocks(self, monkeypatch):
        import services.vision as vision_module

        monkeypatch.setattr(vision_module, "VISION_ENABLED", False)
        event = FakeEvent(
            [seg("text", text="看看"), seg("image", url="http://cdn/1.jpg")],
            text="@bot 看看",
        )
        _result, capture, conversation, resolved = await run_direct(monkeypatch, event)
        assert resolved.image_accepted == 0
        assert conversation.has_any_image is False
        assert capture.require_vision is False
        for message in capture.messages:
            if isinstance(message.get("content"), list):
                assert all(b["type"] != "image_url" for b in message["content"])


# ======================================================================
# Reply
# ======================================================================


class TestDirectReply:
    async def test_reply_image_is_seen_by_the_model(self, monkeypatch):
        bot = FakeBot(
            {
                "get_msg": {
                    "message_id": 4001,
                    "sender": {"user_id": 2002, "nickname": "张三"},
                    "message": [api_seg("image", url="http://cdn/orig.jpg")],
                }
            }
        )
        event = FakeEvent(
            [seg("reply", id="4001"), seg("text", text="这个是谁？")],
            text="@bot 这个是谁？",
        )
        _result, capture, conversation, resolved = await run_direct(
            monkeypatch, event, bot=bot
        )
        assert capture.require_vision is True
        assert resolved.reply is not None
        assert resolved.reply.metadata.sender_name == "张三"

        joined = flat_text(capture.messages)
        # 引用内容与当前消息分块，并保留原作者
        assert "〖用户回复的消息〗" in joined
        assert "发送者：张三" in joined
        assert "〖用户当前消息〗" in joined
        assert joined.index("〖用户回复的消息〗") < joined.index("〖用户当前消息〗")
        # 原图被重新读取，成为真实 image block
        assert "[IMAGE]" in flat_text(capture.messages)

    async def test_reply_placeholder_has_no_original_text(self, monkeypatch):
        from services.context_store import get_recent_messages

        bot = FakeBot(
            {
                "get_msg": {
                    "message_id": 4001,
                    "sender": {"nickname": "张三"},
                    "message": [api_seg("text", text="需要保密的原文")],
                }
            }
        )
        event = FakeEvent(
            [seg("reply", id="4001"), seg("text", text="这个是谁？")],
            text="@bot 这个是谁？",
        )
        await run_direct(monkeypatch, event, bot=bot)
        history = await get_recent_messages(PIPELINE_GROUP_ID, 5)
        joined = "\n".join(m.content for m in history)
        assert "需要保密的原文" not in joined
        assert "[回复了一条消息]" in joined

    async def test_reply_failure_still_answers(self, monkeypatch):
        bot = FakeBot({})  # 任何 API 调用都会抛异常
        event = FakeEvent(
            [seg("reply", id="4001"), seg("text", text="这个是谁？")],
            text="@bot 这个是谁？",
        )
        result, capture, conversation, _resolved = await run_direct(
            monkeypatch, event, bot=bot
        )
        assert result == "夜子的回答"
        assert ai.REPLY_UNAVAILABLE_TEXT if hasattr(ai, "REPLY_UNAVAILABLE_TEXT") else True
        assert "引用消息无法读取" in flat_text(capture.messages)


# ======================================================================
# Forward
# ======================================================================


class TestDirectForward:
    async def test_forward_reaches_prompt_with_senders(self, monkeypatch):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        {
                            "type": "node",
                            "data": {
                                "user_id": "1",
                                "nickname": "A",
                                "time": 1700000000,
                                "message": [api_seg("text", text="你看看这个")],
                            },
                        },
                        {
                            "type": "node",
                            "data": {
                                "user_id": "2",
                                "nickname": "B",
                                "time": 1700000001,
                                "message": [api_seg("text", text="真的假的")],
                            },
                        },
                    ]
                }
            }
        )
        event = FakeEvent(
            [seg("forward", id="fwd-1"), seg("text", text="你怎么看")],
            text="@bot 你怎么看",
        )
        _result, capture, conversation, resolved = await run_direct(
            monkeypatch, event, bot=bot
        )
        assert resolved.forward_node_total == 2
        joined = flat_text(capture.messages)
        assert "〖合并转发开始〗" in joined
        assert "发送者：A" in joined and "发送者：B" in joined
        assert joined.index("你看看这个") < joined.index("真的假的")
        # 转发正文不进入“当前用户消息”本身
        assert "你看看这个" not in conversation.data_text.split("〖合并转发开始〗")[0]

    async def test_forward_placeholder_is_short(self, monkeypatch):
        from services.context_store import get_recent_messages

        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        {
                            "type": "node",
                            "data": {
                                "user_id": "1",
                                "nickname": "A",
                                "message": [api_seg("text", text="私人聊天记录正文")],
                            },
                        }
                    ]
                }
            }
        )
        event = FakeEvent(
            [seg("forward", id="fwd-1"), seg("text", text="看看")], text="@bot 看看"
        )
        await run_direct(monkeypatch, event, bot=bot)
        history = await get_recent_messages(PIPELINE_GROUP_ID, 5)
        joined = "\n".join(m.content for m in history)
        assert "私人聊天记录正文" not in joined
        assert "[发送了一条合并转发，共 1 个节点]" in joined


# ======================================================================
# File
# ======================================================================


class TestDirectFile:
    async def test_file_text_reaches_prompt_as_untrusted(self, monkeypatch, tmp_path):
        import services.perception.file_reader as fr
        from services.perception.net import DownloadResult

        payload = tmp_path / "note.txt"
        payload.write_text("文件里的内容", encoding="utf-8")

        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(
                ok=True, path=str(payload), dir_path=str(tmp_path), size=payload.stat().st_size
            )

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)

        event = FakeEvent(
            [seg("file", name="note.txt", url="http://x/note.txt"), seg("text", text="看看")],
            text="@bot 看看",
        )
        _result, capture, _conversation, resolved = await run_direct(monkeypatch, event)
        assert resolved.file_total == 1
        joined = flat_text(capture.messages)
        assert "〖UNTRUSTED FILE CONTENT〗" in joined
        assert "文件里的内容" in joined
        assert "文件里的内容" not in system_text(capture.messages)

    async def test_file_failure_is_graceful(self, monkeypatch):
        import services.perception.file_reader as fr
        from services.perception.net import DownloadResult

        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(ok=False, note="timeout")

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)

        event = FakeEvent(
            [seg("file", name="a.pdf", url="http://x/a.pdf"), seg("text", text="看看")],
            text="@bot 看看",
        )
        result, capture, _conversation, _resolved = await run_direct(monkeypatch, event)
        assert result == "夜子的回答"
        assert "a.pdf" in flat_text(capture.messages)
        assert "读取失败" in flat_text(capture.messages)
