"""统一 Message Resolver 测试（v0.7）：text / image / reply / forward / file。

全部使用伪造 OneBot 事件与 mock bot（get_msg / get_forward_msg / 文件下载），
绝不访问真实 QQ CDN、NapCat 或任何外部服务。

覆盖（与需求 35 对齐）：
- 普通：text / image / text+image（顺序保持）
- Reply：text / image / text+image / get_msg 失败 / 消息已删除 / 递归 / 深度上限 / 循环
- Forward：text / image / text+image / 多发送者 / 嵌套 / 深度上限 / 节点上限 /
  图片上限 / API 失败 / 循环
- File：txt（真实解析）/ 超大 / 类型不符 / 不支持
"""

from types import SimpleNamespace

import pytest

import services.perception.message_resolver as mr
import services.perception.multimodal_builder as mb
from services.perception.content import FileContent
from services.perception.content import ForwardContent
from services.perception.content import ImageContent
from services.perception.content import SystemNotice
from services.perception.content import TextContent
from services.perception.limits import ContentBudget
from services.perception.message_resolver import MessageResolver


# ======================================================================
# 伪造事件 / mock bot
# ======================================================================


def seg(seg_type: str, **data):
    """构造一个 MessageSegment（SimpleNamespace(type=..., data={...})）。"""
    return SimpleNamespace(type=seg_type, data=data)


def api_seg(seg_type: str, **data) -> dict:
    """构造 OneBot API 返回里的 segment dict（payload 是 dict，不是对象）。"""
    return {"type": seg_type, "data": data}


def event(segments=None, *, text: str = "", group_id: int = 111, user_id: int = 1001):
    segments = segments if segments is not None else []
    return SimpleNamespace(
        group_id=group_id,
        user_id=user_id,
        self_id=999,
        message_id=5001,
        message=segments,
        get_message=lambda: segments,
        get_plaintext=lambda: text,
        sender=SimpleNamespace(user_id=user_id, nickname="小明", card=""),
    )


def image(url: str = "http://cdn/x.jpg", **extra):
    return seg("image", url=url, **extra)


class FakeBot:
    """只实现 call_api 的伪 Bot：按 action 返回预置响应或抛异常。"""

    def __init__(self, responses: dict | None = None, errors: dict | None = None):
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls: list[tuple[str, dict]] = []

    async def call_api(self, action: str, **params):
        self.calls.append((action, params))
        if action in self.errors:
            raise self.errors[action]
        if action in self.responses:
            return self.responses[action]
        raise AssertionError(f"未预期的 API 调用：{action}")


def node(sender_id: str, name: str, segments: list, timestamp: int = 1700000000):
    """构造 OneBot v11 标准转发节点结构。"""
    return {
        "type": "node",
        "data": {
            "user_id": sender_id,
            "nickname": name,
            "time": timestamp,
            "message": segments,
        },
    }


def text_of(message) -> str:
    return message.text()


# ======================================================================
# 普通文本 / 图片
# ======================================================================


class TestPlainMessages:
    async def test_text_only(self):
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([seg("text", text="你好")], text="你好"))
        assert [item.type for item in resolved.message.items] == ["text"]
        assert resolved.message.text() == "你好"
        assert resolved.image_total == 0

    async def test_at_segment_is_not_part_of_text(self):
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(
            event([seg("at", qq="999"), seg("text", text="你好")], text="你好")
        )
        assert resolved.message.text() == "你好"

    async def test_image_only(self):
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([image("http://cdn/1.jpg")]))
        assert resolved.image_total == 1
        assert resolved.image_accepted == 1
        assert isinstance(resolved.message.items[0], ImageContent)
        assert resolved.message.items[0].image.url == "http://cdn/1.jpg"

    async def test_text_image_order_is_preserved(self):
        """核心不变量：文字A / 图片1 / 文字B / 图片2 绝不能重排。"""
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(
            event(
                [
                    seg("text", text="文字A"),
                    image("http://cdn/1.jpg"),
                    seg("text", text="文字B"),
                    image("http://cdn/2.jpg"),
                ]
            )
        )
        types = [item.type for item in resolved.message.items]
        assert types == ["text", "image", "text", "image"]
        assert resolved.message.items[0].text == "文字A"
        assert resolved.message.items[2].text == "文字B"
        assert resolved.message.items[1].image.url == "http://cdn/1.jpg"
        assert resolved.message.items[3].image.url == "http://cdn/2.jpg"

    async def test_multimodal_blocks_keep_order(self):
        """content blocks 与 items 顺序一致：text / image_url / text / image_url。"""
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(
            event(
                [
                    seg("text", text="文字A"),
                    image("http://cdn/1.jpg"),
                    seg("text", text="文字B"),
                    image("http://cdn/2.jpg"),
                ]
            )
        )
        content = await resolver.build_conversation(resolved)
        kinds = [block["type"] for block in content.items]
        assert kinds == ["text", "image_url", "text", "image_url"]
        assert content.items[0]["text"] == "文字A"
        assert content.items[2]["text"] == "文字B"
        assert content.has_any_image is True

    async def test_image_without_http_url_is_placeholdered(self):
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([image("")]))
        item = resolved.message.items[0]
        assert item.image is None
        assert "找不到可读取的地址" in item.placeholder
        assert resolved.image_accepted == 0

    async def test_image_file_field_is_not_treated_as_url(self):
        """`file` 字段是本机文件名 / QQ file id，绝不能当 URL 交给模型。"""
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([seg("image", file="broken.img")]))
        assert resolved.message.items[0].image is None
        assert resolved.image_accepted == 0

    async def test_unsupported_segment_becomes_system_notice(self):
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([seg("video", file="v.mp4")]))
        assert isinstance(resolved.message.items[0], SystemNotice)
        assert "视频" in resolved.message.items[0].text

    async def test_empty_event_is_empty_message(self):
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([], text=""))
        assert resolved.message.is_empty()
        assert resolved.image_total == 0

    async def test_plaintext_fallback_when_no_segments(self):
        """没有可枚举 segment 但 get_plaintext 有内容时，仍然走 AI pipeline。"""
        resolver = MessageResolver()
        resolved = await resolver.resolve_event(event([], text="只有纯文本"))
        assert not resolved.message.is_empty()
        assert resolved.message.text() == "只有纯文本"


# ======================================================================
# Reply
# ======================================================================


class TestReply:
    async def test_reply_text(self):
        bot = FakeBot(
            {
                "get_msg": {
                    "message_id": 4001,
                    "user_id": 2002,
                    "sender": {"user_id": 2002, "nickname": "张三"},
                    "message": [api_seg("text", text="原消息")],
                }
            }
        )
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("reply", id="4001"), seg("text", text="这个是谁？")])
        )
        assert resolved.reply is not None
        assert resolved.reply.text() == "原消息"
        assert resolved.reply.metadata.sender_name == "张三"
        assert resolved.reply.metadata.user_id == 2002
        # 当前消息仍然只有用户真正打出来的字
        assert resolved.message.text() == "这个是谁？"
        assert [call[0] for call in bot.calls] == ["get_msg"]

    async def test_reply_image_is_re_resolved_and_visible(self):
        """回复图片消息：bot 必须真正重新获取并看到原图。"""
        bot = FakeBot(
            {
                "get_msg": {
                    "message_id": 4001,
                    "sender": {"user_id": 2002, "nickname": "张三"},
                    "message": [
                        api_seg("image", url="http://cdn/orig.jpg")
                    ],
                }
            }
        )
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("reply", id="4001"), seg("text", text="这个是谁？")], text="这个是谁？")
        )
        reply_item = resolved.reply.items[0]
        assert isinstance(reply_item, ImageContent)
        assert reply_item.image is not None
        assert reply_item.image.url == "http://cdn/orig.jpg"

        content = await resolver.build_conversation(resolved)
        # 被回复消息的图片进入 image blocks，并被 〖用户回复的消息〗 包裹
        assert content.has_any_image is True
        assert len(content.reply_images) == 1
        assert mb.REPLY_MARKER in content.reply_text
        assert mb.REPLY_END_MARKER in content.reply_text
        assert "〖用户当前消息〗" in content.data_text
        assert content.data_text.index(mb.REPLY_MARKER) < content.data_text.index(
            "〖用户当前消息〗"
        )

    async def test_reply_text_plus_image(self):
        bot = FakeBot(
            {
                "get_msg": {
                    "message": [
                        api_seg("text", text="看看这个"),
                        api_seg("image", url="http://cdn/a.jpg"),
                    ],
                    "sender": {"nickname": "张三"},
                }
            }
        )
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("reply", id="4001"), seg("text", text="是什么？")])
        )
        types = [item.type for item in resolved.reply.items]
        assert types == ["text", "image"]
        assert resolved.reply.text() == "看看这个"

    async def test_reply_api_failure_degrades(self):
        bot = FakeBot(errors={"get_msg": RuntimeError("boom")})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("reply", id="4001"), seg("text", text="这个是谁？")])
        )
        assert resolved.reply is not None
        assert resolved.reply.items[0].text == mr.REPLY_UNAVAILABLE_TEXT

    async def test_reply_deleted_message_degrades(self):
        """消息已被删除 / 不存在：NapCat 返回 retcode!=0 → 结构化降级。"""
        bot = FakeBot(errors={"get_msg": SimpleNamespace(retcode=100)})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(event([seg("reply", id="4001")]))
        assert resolved.reply.items[0].text == mr.REPLY_UNAVAILABLE_TEXT

    async def test_reply_without_bot_degrades(self):
        resolver = MessageResolver(None)
        resolved = await resolver.resolve_event(event([seg("reply", id="4001")]))
        assert resolved.reply.items[0].text == mr.REPLY_UNAVAILABLE_TEXT

    async def test_reply_recursive(self):
        """A 回复 B、B 回复 C：递归解析出完整引用链。"""
        bot = FakeBot(
            {
                "get_msg": {
                    "message_id": 4001,
                    "sender": {"nickname": "B"},
                    "message": [
                        api_seg("reply", id="3001"),
                        api_seg("text", text="B 的回复"),
                    ],
                }
            }
        )
        # 第二次 get_msg（3001）返回不同内容 → 用状态化实现
        original = bot.call_api

        async def stateful(action: str, **params):
            if action == "get_msg" and params.get("message_id") == "3001":
                bot.calls.append((action, params))
                return {
                    "message_id": 3001,
                    "sender": {"nickname": "C"},
                    "message": [api_seg("text", text="C 的原消息")],
                }
            return await original(action, **params)

        bot.call_api = stateful
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(event([seg("reply", id="4001")]))
        assert resolved.reply.text() == "B 的回复"
        assert resolved.reply.reply is not None
        assert resolved.reply.reply.text() == "C 的原消息"
        assert resolved.reply.reply.metadata.sender_name == "C"

    async def test_reply_recursion_limit(self):
        """REPLY_MAX_DEPTH=2：第三层不再展开，而是给出结构化说明。"""
        calls: list[str] = []

        class ChainBot:
            async def call_api(self, action: str, **params):
                message_id = params.get("message_id")
                calls.append(message_id)
                # 每层都回复上一层：1000 → 1001 → 1002 → ...
                return {
                    "message_id": message_id,
                    "sender": {"nickname": f"用户{message_id}"},
                    "message": [
                        {
                            "type": "reply",
                            "data": {"id": str(int(message_id) + 1)},
                        },
                        api_seg("text", text=f"第 {message_id} 层"),
                    ],
                }

        resolver = MessageResolver(ChainBot(), reply_max_depth=2)
        resolved = await resolver.resolve_event(event([seg("reply", id="1000")]))
        # 最多解析两层，第三层不再调用 API（绝不无限展开）
        assert calls == ["1000", "1001"]
        assert resolved.reply.text() == "第 1000 层"
        assert resolved.reply.reply.text() == "第 1001 层"
        # 第三层是一条结构化说明（引用链过深），而不是继续请求
        assert resolved.reply.reply.reply.items[0].text == mr.REPLY_DEPTH_LIMIT_TEXT

    async def test_reply_cycle_detection(self):
        """A 回复 B、B 回复 A：重复 message_id 直接停止展开。"""
        class CycleBot:
            async def call_api(self, action: str, **params):
                message_id = params.get("message_id")
                other = "2001" if message_id == "2002" else "2002"
                return {
                    "message_id": message_id,
                    "sender": {"nickname": "循环"},
                    "message": [
                        api_seg("reply", id=other),
                        api_seg("text", text=f"节点 {message_id}"),
                    ],
                }

        resolver = MessageResolver(CycleBot(), reply_max_depth=5)
        resolved = await resolver.resolve_event(event([seg("reply", id="2001")]))
        assert resolved.reply.text() == "节点 2001"
        assert resolved.reply.reply.text() == "节点 2002"
        # 第三次遇到 2001 → 循环保护
        assert resolved.reply.reply.reply.items[0].text == mr.REPLY_CYCLE_TEXT


# ======================================================================
# Forward（合并转发）
# ======================================================================


class TestForward:
    async def _resolve_forward(self, bot, resolver=None, forward_data=None):
        resolver = resolver or MessageResolver(bot)
        segments = [seg("forward", **(forward_data or {"id": "fwd-1"}))]
        resolved = await resolver.resolve_event(event(segments))
        return resolver, resolved, resolved.message.items[0]

    async def test_forward_text_multi_user(self):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node("1", "A", [api_seg("text", text="你看看这个")]),
                        node("2", "B", [api_seg("text", text="真的假的")]),
                        node("1", "A", [api_seg("text", text="我也不知道")]),
                    ]
                }
            }
        )
        resolver, resolved, forward = await self._resolve_forward(bot)
        assert isinstance(forward, ForwardContent)
        assert forward.ok is True
        assert [item.sender_name for item in forward.nodes] == ["A", "B", "A"]
        assert [item.index for item in forward.nodes] == [1, 2, 3]
        assert forward.nodes[0].sender_id == "1"
        assert forward.nodes[1].timestamp == 1700000000

        content = await resolver.build_conversation(resolved)
        block = content.forward_block
        assert mb.FORWARD_START_MARKER in block
        assert mb.FORWARD_END_MARKER in block
        # 每个节点都保留“谁说了什么”
        assert "节点 1" in block and "发送者：A" in block
        assert "节点 2" in block and "发送者：B" in block
        assert block.index("你看看这个") < block.index("真的假的") < block.index("我也不知道")

    async def test_forward_image_becomes_real_image_block(self):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [
                                api_seg("text", text="你看看这个"),
                                api_seg("image", url="http://cdn/f1.jpg"),
                            ],
                        ),
                        node("2", "B", [api_seg("text", text="这谁啊？")]),
                    ]
                }
            }
        )
        resolver, resolved, forward = await self._resolve_forward(bot)
        node_items = forward.nodes[0].items
        assert [item.type for item in node_items] == ["text", "image"]
        assert node_items[1].image.url == "http://cdn/f1.jpg"

        content = await resolver.build_conversation(resolved)
        kinds = [block["type"] for block in content.items]
        # 真实 image block（不是字符串占位）
        assert "image_url" in kinds
        image_block = next(b for b in content.items if b["type"] == "image_url")
        assert image_block["image_url"]["url"] == "http://cdn/f1.jpg"
        assert content.has_any_image is True

    async def test_forward_inline_content_preferred_over_api(self):
        """segment 自带 content 时不得再调用 get_forward_msg。"""
        bot = FakeBot({})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event(
                [
                    seg(
                        "forward",
                        id="fwd-inline",
                        content=[node("1", "A", [api_seg("text", text="自带内容")])],
                    )
                ]
            )
        )
        assert bot.calls == []
        assert resolved.message.items[0].nodes[0].items[0].text == "自带内容"

    async def test_nested_forward(self):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [
                                {
                                    "type": "forward",
                                    "data": {
                                        "content": [
                                            node(
                                                "3",
                                                "C",
                                                [
                                                    {
                                                        "type": "text",
                                                        "data": {"text": "内层消息"},
                                                    }
                                                ],
                                            )
                                        ]
                                    },
                                }
                            ],
                        )
                    ]
                }
            }
        )
        resolver, resolved, forward = await self._resolve_forward(bot)
        inner = forward.nodes[0].items[0]
        assert isinstance(inner, ForwardContent)
        assert inner.nodes[0].sender_name == "C"

        content = await resolver.build_conversation(resolved)
        block = content.forward_block
        assert block.count(mb.FORWARD_START_MARKER) == 2
        assert block.count(mb.FORWARD_END_MARKER) == 2
        assert "内层消息" in block

    async def test_forward_max_depth(self):
        """FORWARD_MAX_DEPTH=1：嵌套一层后不再展开。"""
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [
                                {
                                    "type": "forward",
                                    "data": {
                                        "id": "inner",
                                        "content": [
                                            node(
                                                "2",
                                                "B",
                                                [
                                                    {
                                                        "type": "text",
                                                        "data": {"text": "太深了"},
                                                    }
                                                ],
                                            )
                                        ],
                                    },
                                }
                            ],
                        )
                    ]
                }
            }
        )
        resolver = MessageResolver(bot, forward_max_depth=1)
        _resolver, resolved, forward = await self._resolve_forward(bot, resolver)
        inner = forward.nodes[0].items[0]
        assert isinstance(inner, ForwardContent)
        assert inner.nodes == ()
        assert inner.note == mr.FORWARD_DEPTH_LIMIT_TEXT

    async def test_forward_node_limit(self):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(str(i), f"U{i}", [api_seg("text", text=f"第{i}条")])
                        for i in range(1, 11)
                    ]
                }
            }
        )
        resolver = MessageResolver(bot, forward_max_nodes=3)
        _resolver, resolved, forward = await self._resolve_forward(bot, resolver)
        assert len(forward.nodes) == 3
        # 超出时只加结构化说明，绝不报错
        assert "后续 7 条转发消息因上下文限制未展开" in resolved.notice

    async def test_forward_image_limit(self):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [
                                api_seg("image", url=f"http://cdn/{i}.jpg")
                                for i in range(5)
                            ],
                        )
                    ]
                }
            }
        )
        resolver = MessageResolver(bot, forward_max_images=2, forward_max_nodes=10)
        _resolver, resolved, forward = await self._resolve_forward(bot, resolver)
        loaded = [
            item for item in forward.nodes[0].items if item.image is not None
        ]
        assert len(loaded) == 2
        assert "该合并转发后续还有" in resolved.notice
        assert "张图片" in resolved.notice

    async def test_forward_file_limit(self, monkeypatch):
        """文件数量超限：不读取、只加结构化说明。"""
        monkeypatch.setattr(mr, "FORWARD_MAX_FILES", 0)
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [
                                api_seg("file", name="a.pdf", url="http://x/a.pdf"),
                            ],
                        )
                    ]
                }
            }
        )
        resolver = MessageResolver(bot, forward_max_files=0)
        _resolver, resolved, forward = await self._resolve_forward(bot, resolver)
        item = forward.nodes[0].items[0]
        assert isinstance(item, FileContent)
        assert item.ok is False
        assert "文件数量超过限制" in item.note

    async def test_forward_api_failure(self):
        bot = FakeBot(errors={"get_forward_msg": RuntimeError("boom")})
        _resolver, resolved, forward = await self._resolve_forward(bot)
        assert forward.ok is False
        assert forward.note == mr.FORWARD_UNAVAILABLE_TEXT
        # 聊天流程照常继续：当前消息结构完好
        assert resolved.message.items[0].type == "forward"

    async def test_forward_timeout(self, monkeypatch):
        import asyncio

        class SlowBot:
            async def call_api(self, action: str, **params):
                await asyncio.sleep(5)
                return {"messages": []}

        monkeypatch.setattr(mr, "API_TIMEOUT_SECONDS", 0.01)
        _resolver, resolved, forward = await self._resolve_forward(SlowBot())
        assert forward.note == mr.FORWARD_UNAVAILABLE_TEXT

    async def test_forward_cycle_detection(self):
        """A → B → A 的嵌套转发：相同 forward_id 直接停止递归。"""
        class CycleBot:
            async def call_api(self, action: str, **params):
                target = params.get("id")
                other = "b" if target == "a" else "a"
                return {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [
                                {"type": "forward", "data": {"id": other}},
                                api_seg("text", text=f"层 {target}"),
                            ],
                        )
                    ]
                }

        resolver = MessageResolver(CycleBot(), forward_max_depth=5)
        _resolver, resolved, forward = await self._resolve_forward(
            CycleBot(), resolver, forward_data={"id": "a"}
        )
        # 外层 a → 节点内嵌 forward b → b 的节点内再嵌 forward a（循环）
        inner_b = forward.nodes[0].items[0]
        assert isinstance(inner_b, ForwardContent)
        assert inner_b.forward_id == "b"
        inner_a = inner_b.nodes[0].items[0]
        assert isinstance(inner_a, ForwardContent)
        assert inner_a.forward_id == "a"
        assert inner_a.nodes == ()
        assert inner_a.note == mr.FORWARD_CYCLE_TEXT

    async def test_forward_text_budget_truncates(self):
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node(
                            "1",
                            "A",
                            [api_seg("text", text="很长的内容" * 100)],
                        )
                    ]
                }
            }
        )
        resolver = MessageResolver(bot, forward_max_text_chars=50)
        _resolver, resolved, forward = await self._resolve_forward(bot, resolver)
        text = forward.nodes[0].items[0].text
        assert len(text) <= 60
        assert "已截断" in text

    async def test_forward_absence_of_id_and_content(self):
        bot = FakeBot({})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(event([seg("forward")]))
        forward = resolved.message.items[0]
        assert forward.ok is False
        assert forward.note == mr.FORWARD_UNAVAILABLE_TEXT


# ======================================================================
# File（走真实 File Reader，下载用 monkeypatch 桩）
# ======================================================================


class TestFileSegments:
    async def test_file_segment_reads_text(self, monkeypatch, tmp_path):
        """file segment → 真实下载（桩）→ 真实 txt 解析。"""
        import services.perception.file_reader as fr
        from services.perception.net import DownloadResult

        payload = tmp_path / "note.txt"
        payload.write_text("这是文件正文", encoding="utf-8")

        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(
                ok=True,
                path=str(payload),
                dir_path=str(tmp_path),
                size=payload.stat().st_size,
            )

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)

        resolver = MessageResolver(FakeBot({}))
        resolved = await resolver.resolve_event(
            event(
                [
                    seg("file", name="note.txt", url="http://x/note.txt"),
                    seg("text", text="帮我看看"),
                ]
            )
        )
        item = next(i for i in resolved.message.items if i.type == "file")
        assert isinstance(item, FileContent)
        assert item.ok is True
        assert item.text == "这是文件正文"
        assert item.parser_type == "text"
        assert resolved.file_total == 1
        # 顺序保持：file 在前、text 在后
        assert [i.type for i in resolved.message.items] == ["file", "text"]

    async def test_file_without_url_degrades(self):
        resolver = MessageResolver(FakeBot({}))
        resolved = await resolver.resolve_event(
            event([seg("file", name="a.pdf", file_id="xyz")])
        )
        item = resolved.message.items[0]
        assert item.ok is False
        assert "a.pdf" in item.note

    async def test_file_type_mismatch_is_refused(self, monkeypatch, tmp_path):
        import services.perception.file_reader as fr
        from services.perception.net import DownloadResult

        payload = tmp_path / "homework.pdf"
        payload.write_bytes(b"MZ\x90\x00" + b"\x00" * 64)  # 实际是 exe

        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(ok=True, path=str(payload), dir_path=str(tmp_path), size=68)

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)

        resolver = MessageResolver(FakeBot({}))
        resolved = await resolver.resolve_event(
            event([seg("file", name="homework.pdf", url="http://x/homework.pdf")])
        )
        item = resolved.message.items[0]
        assert item.ok is False
        assert "不一致" in item.note


# ======================================================================
# 回归（v0.7.1）：真实适配器会在 matcher 之前删掉 reply segment
# ======================================================================

# 线上实测：@ 夜子 并「回复一条合并转发」时，解析日志一直是
# `reply=0 forward_nodes=0`，夜子只好回答“到我这儿是空的”。
# 根因：nonebot.adapters.onebot.v11.bot._check_reply() 在事件进入任何
# matcher 之前就已经调用 get_msg 取回被回复消息、写进 event.reply，并把
# reply segment 从 event.message 中删除。只扫 segment 的解析器因此永远
# 看不到引用，引用里的合并转发自然也不会被展开。
# 下面用**真实的适配器函数**复现这条链，避免测试与线上行为漂移。

# 注意：下面全部使用**合成 ID**（不复用任何真实 QQ 号 / 群号），
# 事件结构本身则严格按 NapCat 真实下发过的形态还原。

REPLIED_MESSAGE_ID = 50001
FORWARD_ID = "fwd-test-0001"
BOT_ID = 900000001
GROUP_ID = 111111111
USER_ID = 10001


def napcat_reply_event_payload() -> dict:
    """还原 NapCat 下发的原始群消息事件结构（reply + at + text）。"""
    return {
        "time": 1789561895,
        "self_id": BOT_ID,
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 50002,
        "group_id": GROUP_ID,
        "user_id": USER_ID,
        "message": [
            {"type": "reply", "data": {"id": str(REPLIED_MESSAGE_ID)}},
            {"type": "at", "data": {"qq": str(BOT_ID)}},
            {"type": "text", "data": {"text": " 看看这个"}},
        ],
        "raw_message": (
            f"[CQ:reply,id={REPLIED_MESSAGE_ID}][CQ:at,qq={BOT_ID}] 看看这个"
        ),
        "font": 0,
        "sender": {
            "user_id": USER_ID,
            "nickname": "测试用户",
            "card": "",
            "role": "member",
        },
        "anonymous": None,
    }


def napcat_replied_message_payload() -> dict:
    """被回复的那条消息：本身是一个合并转发。"""
    return {
        "message_id": REPLIED_MESSAGE_ID,
        "real_id": REPLIED_MESSAGE_ID,
        "time": 1789561876,
        "message_type": "group",
        "sender": {"user_id": USER_ID, "nickname": "测试用户", "card": ""},
        "message": [{"type": "forward", "data": {"id": FORWARD_ID}}],
    }


class StubGetMsgBot:
    """只实现适配器 _check_reply 需要的 get_msg。"""

    def __init__(self, replied_payload: dict):
        self.replied_payload = replied_payload
        self.calls: list[int] = []

    async def get_msg(self, message_id: int):
        self.calls.append(message_id)
        return self.replied_payload


class TestAdapterStripsReplySegment:
    async def test_adapter_really_deletes_reply_segment(self):
        """先把「适配器会删 segment」这个前提钉死，修复失效时能立刻发现。"""
        from nonebot.adapters.onebot.v11.bot import _check_reply
        from nonebot.adapters.onebot.v11.event import GroupMessageEvent

        ev = GroupMessageEvent.model_validate(napcat_reply_event_payload())
        assert [s.type for s in ev.message] == ["reply", "at", "text"]

        stub = StubGetMsgBot(napcat_replied_message_payload())
        await _check_reply(stub, ev)

        assert stub.calls == [REPLIED_MESSAGE_ID]
        # 适配器确实把 reply segment 删掉了，内容转存到 event.reply
        assert [s.type for s in ev.message] == ["at", "text"]
        assert ev.reply is not None

    async def test_resolver_reads_preloaded_reply_and_expands_forward(self):
        """修复点：即使 segment 已被删除，引用里的合并转发仍要被读到。"""
        from nonebot.adapters.onebot.v11.bot import _check_reply
        from nonebot.adapters.onebot.v11.event import GroupMessageEvent

        ev = GroupMessageEvent.model_validate(napcat_reply_event_payload())
        await _check_reply(StubGetMsgBot(napcat_replied_message_payload()), ev)

        # FakeBot 未预置 get_msg：若解析器重复调用 get_msg 会直接断言失败
        bot = FakeBot(
            {
                "get_forward_msg": {
                    "messages": [
                        node("1", "甲", [api_seg("text", text="夜子你看看这个")]),
                        node("2", "乙", [api_seg("text", text="真的假的")]),
                    ]
                }
            }
        )
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(ev)

        # 修复前：reply 为 None（日志里的 reply=0）
        assert resolved.reply is not None

        forward = resolved.reply.items[0]
        assert isinstance(forward, ForwardContent)
        assert forward.ok is True
        assert forward.forward_id == FORWARD_ID
        assert [n.sender_name for n in forward.nodes] == ["甲", "乙"]

        # build_conversation 必须把转发正文真正带进 prompt
        content = await resolver.build_conversation(resolved)
        block = content.reply_text
        assert mb.REPLY_MARKER in block
        assert mb.FORWARD_START_MARKER in block
        assert "夜子你看看这个" in block
        assert "真的假的" in block

        # 且没有为引用重复请求一次 get_msg
        assert [action for action, _ in bot.calls] == ["get_forward_msg"]


# ======================================================================
# 回归（v0.7.2）：群文件段没有 url，必须用 get_group_file_url 换下载直链
# ======================================================================

# 线上实测（探针直接问 NapCat 拿到的原始数据形态，此处用合成 ID 复现）：
#   {"type": "file",
#    "data": {"file": "problems.pdf",
#             "file_id": "/test-file-id",
#             "file_size": "884783"}}
# —— 没有任何 url 字段，因此文件永久停在
#   `[FILE] 无下载地址（source=message parser=none）`。
# 而 get_group_file_url 能返回可用直链；且群号选择有讲究：
# 实测「文件所属原群」会 retcode=1200（机器人不在那个群），只有
# 「收到转发的当前群」能拿到直链 —— 所以当前群必须优先。

FORWARD_ID_WITH_FILE = "fwd-test-0002"
SOURCE_GROUP_ID = 222222222
CURRENT_GROUP_ID = 111111111
FILE_ID = "/test-file-id"
FILE_URL = "https://example.invalid/ftn_handler/deadbeef/?fname=problems.pdf"


def forward_node_with_file() -> dict:
    """还原 NapCat 真实返回的、带群文件的转发节点。"""
    return {
        "type": "node",
        "data": {
            "user_id": "20001",
            "nickname": "转发者",
            "time": 1789396080,
            "group_id": SOURCE_GROUP_ID,
            "message": [
                {
                    "type": "file",
                    "data": {
                        "file": "problems.pdf",
                        "file_id": FILE_ID,
                        "file_size": "884783",
                    },
                }
            ],
        },
    }


class PerGroupFileBot:
    """按 group_id 决定 get_group_file_url 成败，复现真机的 1200 无权限行为。

    真机实测：当前群 ok / 文件所属原群 failed(retcode=1200)。
    """

    def __init__(self, ok_groups):
        self.ok_groups = set(ok_groups)
        self.calls: list[tuple[str, dict]] = []

    async def call_api(self, action: str, **params):
        self.calls.append((action, params))
        if action == "get_forward_msg":
            return {"messages": [forward_node_with_file()]}
        if action == "get_group_file_url":
            if params.get("group_id") in self.ok_groups:
                return {"url": FILE_URL}
            raise RuntimeError("retcode=1200 无权限")
        raise AssertionError(f"未预期的 API 调用：{action}")

    @property
    def url_calls(self) -> list[dict]:
        return [p for a, p in self.calls if a == "get_group_file_url"]


class TestForwardFileUrlExchange:
    def _prepare(self, monkeypatch, tmp_path):
        """准备好真实的 PDF 与假下载，返回 (downloaded, patch 上下文)。"""
        fitz = pytest.importorskip("fitz")
        import services.perception.file_reader as fr
        from services.perception.net import DownloadResult

        pdf_path = tmp_path / "problems.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # 用 ASCII：PyMuPDF 默认字体（Helvetica）无法编码中文，
        # 写中文进去提取出来会是乱码，与本用例要验证的东西无关。
        page.insert_text((72, 72), "problems.pdf body text")
        doc.save(str(pdf_path))
        doc.close()

        downloaded: list[str] = []

        async def fake_download(url, max_bytes, file_name=""):
            downloaded.append(url)
            return DownloadResult(
                ok=True,
                path=str(pdf_path),
                dir_path=str(tmp_path),
                size=pdf_path.stat().st_size,
            )

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)
        return downloaded

    async def test_current_group_is_tried_first(self, monkeypatch, tmp_path):
        """当前群优先——真机上只有它能拿到直链。"""
        downloaded = self._prepare(monkeypatch, tmp_path)

        bot = PerGroupFileBot(ok_groups={CURRENT_GROUP_ID})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("forward", id=FORWARD_ID_WITH_FILE)], group_id=CURRENT_GROUP_ID)
        )

        forward = resolved.message.items[0]
        assert isinstance(forward, ForwardContent)
        item = forward.nodes[0].items[0]
        assert isinstance(item, FileContent)
        assert item.ok is True
        assert "problems.pdf body text" in item.text
        # file_total 必须递归统计嵌套在转发节点里的文件（修复前恒为 0）
        assert resolved.file_total == 1

        assert bot.url_calls, "未调用 get_group_file_url，文件将永远无下载地址"
        # 第一次就该用当前群（不是文件所属的原群）
        assert bot.url_calls[0] == {
            "group_id": CURRENT_GROUP_ID,
            "file_id": FILE_ID,
        }
        # 当前群成功就不该再去试原群
        assert len(bot.url_calls) == 1
        assert downloaded == [FILE_URL]

    async def test_falls_back_to_own_group_when_current_group_denied(
        self, monkeypatch, tmp_path
    ):
        """当前群无权限时，仍要回退到文件所属原群再试一次。"""
        downloaded = self._prepare(monkeypatch, tmp_path)

        bot = PerGroupFileBot(ok_groups={SOURCE_GROUP_ID})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("forward", id=FORWARD_ID_WITH_FILE)], group_id=CURRENT_GROUP_ID)
        )

        item = resolved.message.items[0].nodes[0].items[0]
        assert isinstance(item, FileContent)
        assert item.ok is True
        assert downloaded == [FILE_URL]

        tried_groups = [p["group_id"] for p in bot.url_calls]
        assert tried_groups[0] == CURRENT_GROUP_ID
        assert SOURCE_GROUP_ID in tried_groups

    async def test_all_candidates_denied_degrades_to_file_id_only(self, monkeypatch):
        """所有群都无权限时：只降级成结构化提示，绝不抛异常、绝不漏掉这条消息。"""
        from services.perception.file_reader import read_file as real_read_file

        async def fake_download(url, max_bytes, file_name=""):  # pragma: no cover
            raise AssertionError("没有拿到直链时不应该发起下载")

        import services.perception.file_reader as fr

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)
        assert real_read_file is fr.read_file

        bot = PerGroupFileBot(ok_groups=set())
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("forward", id=FORWARD_ID_WITH_FILE)], group_id=CURRENT_GROUP_ID)
        )
        item = resolved.message.items[0].nodes[0].items[0]
        assert isinstance(item, FileContent)
        assert item.ok is False
        assert "problems.pdf" in item.note

    async def test_url_missing_everywhere_degrades_gracefully(self):
        """连 file_id 都拿不到时，只降级成结构化提示，绝不抛异常。"""
        bot = FakeBot({})
        resolver = MessageResolver(bot)
        resolved = await resolver.resolve_event(
            event([seg("file", name="no_id.pdf", file_size=100)], group_id=CURRENT_GROUP_ID)
        )
        item = resolved.message.items[0]
        assert isinstance(item, FileContent)
        assert item.ok is False
        assert "no_id.pdf" in item.note
        assert not [a for a, _ in bot.calls if a == "get_group_file_url"]
