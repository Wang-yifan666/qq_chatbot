"""群白名单 + handler 集成测试（v0.3.1）。

用构造的伪 OneBot 事件直接调用插件 handler（复用 scripts/test_group_access.py
的成熟做法，改写为 pytest 用例）：
- 未授权群：不读取消息正文、不回复、不调用 AI、不写
  messages / users / relationships / user_memories；
- 授权群：正常进入流程，AI 调用与回复通道用桩替代，
  绝不访问真实 QQ / LLM / 搜索服务。
"""

import asyncio
from types import SimpleNamespace

from nonebot.exception import FinishedException

class FakeEvent:
    """伪 OneBot GroupMessageEvent。

    get_plaintext 模拟 onebot v11 真实行为：去掉开头的 @机器人 本体；
    plaintext_calls / message_calls 用于断言“未授权群根本没读取正文 / 图片”。
    message_segments 模拟 event.get_message() 的 MessageSegment 列表
    （SimpleNamespace(type=..., data=...)）。
    """

    def __init__(
        self,
        group_id: int,
        user_id: int,
        self_id: int,
        text: str,
        segments: list | None = None,
    ):
        self.group_id = group_id
        self.user_id = user_id
        self.self_id = self_id
        self._text = text
        self._segments = segments if segments is not None else []
        self.plaintext_calls = 0
        self.message_calls = 0
        self.to_me = False
        self.sender = SimpleNamespace(user_id=user_id, nickname="小明", card="")

    def get_plaintext(self) -> str:
        self.plaintext_calls += 1
        text = self._text
        if text.startswith("@bot"):
            text = text[len("@bot") :].strip()
        return text

    def get_message(self) -> list:
        self.message_calls += 1
        return self._segments


class FakeChat:
    """替代 NoneBot Matcher 的 finish / send：记录回复，不真正发消息。

    finish 与真实 Matcher.finish 一致地抛 FinishedException
    （对照本仓库安装的 nonebot 2.5.0），保证 handler 内 finish 之后的
    代码在测试里同样不会执行。
    """

    def __init__(self) -> None:
        self.finished: list[str | None] = []
        self.sent: list[str] = []

    async def finish(self, message: str | None = None) -> None:
        self.finished.append(message)
        raise FinishedException

    async def send(self, message: str) -> None:
        self.sent.append(message)


async def _call_handler(handler, event, bot=None) -> None:
    """直接调用插件 handler；吞掉与 NoneBot 运行环境一致的 FinishedException。

    bot：v0.7 起 DIRECT handler 通过 NoneBot 依赖注入拿到 OneBot V11 Bot
    （只用于 Reply / Forward 的 get_msg / get_forward_msg 只读查询）。
    其它插件的 handler 只有 event 参数，因此这里按签名决定是否传 bot。
    """
    import inspect

    params = inspect.signature(handler).parameters
    try:
        if len(params) >= 2:
            await handler(event, bot)
        else:
            await handler(event)
    except FinishedException:
        pass


async def _count(table: str, where: str = "") -> int:
    from services.database import db_conn

    conn = db_conn()
    sql = f"SELECT COUNT(*) AS c FROM {table}"
    if where:
        sql += f" WHERE {where}"
    cursor = await conn.execute(sql)
    row = await cursor.fetchone()
    await cursor.close()
    return int(row["c"])


async def _gate_scenario() -> None:
    """完整门禁场景：未授权群零副作用 + 授权群正常流程（全部桩化）。

    对 `ai._answer` / `ai.chat` / `dg.debug` 的替换都用 monkeypatch，
    测试结束时自动还原：否则这些桩会泄漏到同一 session 里的其它测试文件
    （pytest 按文件顺序共用进程），让后面的端到端测试“看不到真实 pipeline”。
    """
    import nonebot

    nonebot.init()

    import pytest

    import services.database as dbm
    from plugins import ai_chat as ai
    from plugins import ambient as amb
    from plugins import context_recorder as cr
    from plugins import debug as dg

    await dbm.init_db()

    # --- 替换 AI 调用与回复通道（绝不打真实 API；测试结束自动还原） ---
    calls = {"answer": 0, "last_resolved": None, "last_conversation": None}

    async def fake_answer(
        event,
        normalized_text: str,
        conversation=None,
        resolved=None,
    ) -> str:
        """v0.7 桩：AI 层现在接收归一化文本 + 感知层结果，而不是原始图片列表。

        这里只做记录，不做任何解析——解析全部由 Message Resolver 在插件内完成，
        因此这些断言同时验证了“插件层不再自己解析 image segment”。
        """
        calls["answer"] += 1
        calls["last_question"] = normalized_text
        calls["last_resolved"] = resolved
        calls["last_conversation"] = conversation
        return "stub-answer"

    ai_chat = FakeChat()
    dg_chat = FakeChat()
    mp = pytest.MonkeyPatch()
    mp.setattr(ai, "_answer", fake_answer)
    mp.setattr(ai, "chat", ai_chat)
    mp.setattr(dg, "debug", dg_chat)
    try:
        await _gate_scenario_body(nonebot, ai, amb, cr, dg, ai_chat, dg_chat, calls)
    finally:
        mp.undo()


async def _gate_scenario_body(nonebot, ai, amb, cr, dg, ai_chat, dg_chat, calls) -> None:
    """门禁场景主体（patch 的安装 / 还原由 _gate_scenario 负责）。"""
    # ===== 未授权群（333）：零副作用 =====
    # @机器人 hello → 不读正文、不调 AI、不回复
    ev = FakeEvent(333, 1001, 999, "@bot hello")
    await _call_handler(ai.handle, ev)
    assert ev.plaintext_calls == 0, "未授权群 @ 消息不应被读取正文"
    assert calls["answer"] == 0, "未授权群不得触发 AI 调用"
    assert not ai_chat.finished and not ai_chat.sent, "未授权群不得收到任何回复"

    # 只 @机器人 → 也不回复「有什么想问我的？」
    ev = FakeEvent(333, 1001, 999, "@bot")
    await _call_handler(ai.handle, ev)
    assert not ai_chat.finished, "未授权群只 @ 也不得回复提示语"
    assert ev.plaintext_calls == 0

    # v0.5 视觉隐私边界：未授权群带图 @ → 图片 segment 根本不被读取
    ev = FakeEvent(
        333,
        1001,
        999,
        "@bot",
        segments=[SimpleNamespace(type="image", data={"url": "http://evil/1.jpg"})],
    )
    await _call_handler(ai.handle, ev)
    assert ev.plaintext_calls == 0, "未授权群不应读取正文"
    assert ev.message_calls == 0, "未授权群不得读取图片 segment（白名单在任何提取之前）"
    assert calls["answer"] == 0
    assert not ai_chat.finished

    # 普通消息 → 不写 messages / users / relationships / user_memories
    ev = FakeEvent(333, 1003, 999, "hello from unauthorized group")
    await _call_handler(cr.handle, ev)
    assert await _count("messages", "group_id = 333") == 0
    assert await _count("users", "user_id = 1003") == 0
    # 只断言“本测试涉及的用户”没有关系记录：relationships 是全进程共享表，
    # 其它测试文件（如 test_direct_pipeline）会为自己独立的测试用户写入计数，
    # 这里不能把别人的数据算成本次未授权群的副作用。
    assert await _count("relationships", "user_id IN (1001, 1002, 1003)") == 0
    assert await _count("user_memories") == 0

    # \debug memory set 携带敏感 value → 完全沉默，且正文未被读取
    ev = FakeEvent(333, 1001, 999, "\\debug memory set 1003 secret the-secret-value")
    await _call_handler(dg.handle, ev)
    assert ev.plaintext_calls == 0
    assert not dg_chat.finished and not dg_chat.sent

    # debug 匹配规则层：未授权群在读取正文之前就拒绝（正文连规则都进不去）
    ev = FakeEvent(333, 1001, 999, "\\debug status")
    assert not await dg._debug_rule(ev)
    assert ev.plaintext_calls == 0, "未授权群的命令文本不应被规则层读取"

    # ambient 插件门禁：未授权群不读正文、不进入 ambient 调度（不产生任何任务）
    import plugins.ambient as ambient_plugin

    ambient_calls: list = []

    async def fake_on_group_message(event):
        ambient_calls.append(event.group_id)

    ambient_plugin.on_group_message = fake_on_group_message
    ev = FakeEvent(333, 1001, 999, "unauthorized ambient chat")
    await _call_handler(ambient_plugin.handle, ev)
    assert ev.plaintext_calls == 0, "未授权群的 ambient 消息不应被读取正文"
    assert ambient_calls == []

    # 授权群 + AMBIENT_ENABLED=false（conftest 默认）：ambient 保持静默
    ev = FakeEvent(111, 1001, 999, "authorized ambient chat")
    await _call_handler(ambient_plugin.handle, ev)
    assert ev.plaintext_calls == 0, "AMBIENT_ENABLED=false 时不应读取正文"
    assert ambient_calls == []

    # ===== Scheduled cron 注册（真实插件；只 add_job 不启动 scheduler） =====
    nonebot.load_plugin("nonebot_plugin_apscheduler")
    import services.scheduled_tasks as st
    from nonebot_plugin_apscheduler import scheduler

    morning_task = st.ScheduledTask(
        "morning_greeting", "morning_greeting", True, 8, 0, frozenset({111}), False, 30, 10
    )
    night_task = st.ScheduledTask(
        "night_greeting", "night_greeting", True, 21, 0, frozenset({111, 222}), False, 30, 10
    )
    st.SCHEDULED_TASKS_ENABLED = True
    st.SCHEDULED_TASKS = {"morning_greeting": morning_task, "night_greeting": night_task}
    try:
        st.setup_scheduled_tasks(nonebot.get_driver())
        morning_job = scheduler.get_job("task:morning_greeting")
        assert morning_job is not None, "morning_greeting cron 应已注册"
        assert morning_job.coalesce is True
        assert morning_job.max_instances == 1
        assert morning_job.misfire_grace_time == 1800
        morning_fields = {f.name: list(f.expressions) for f in morning_job.trigger.fields}
        assert str(morning_fields["hour"][0]) == "8"
        assert str(morning_fields["minute"][0]) == "0"
        assert str(morning_job.trigger.timezone) == "Asia/Shanghai"
        # 注册表循环注册：night_greeting 同样被注册到自己的 cron
        night_job = scheduler.get_job("task:night_greeting")
        assert night_job is not None, "night_greeting cron 应已注册"
        night_fields = {f.name: list(f.expressions) for f in night_job.trigger.fields}
        assert str(night_fields["hour"][0]) == "21"
        assert str(night_fields["minute"][0]) == "0"
    finally:
        scheduler.remove_job("task:morning_greeting")
        scheduler.remove_job("task:night_greeting")
        st.SCHEDULED_TASKS_ENABLED = False
        st.SCHEDULED_TASKS = {}

    # ===== 授权群（111 / 222）：正常流程（AI 层桩替代） =====
    ev = FakeEvent(111, 1001, 999, "@bot hello")
    await _call_handler(ai.handle, ev)
    # v0.7：授权群读取正文三次——\ping 快速自检判定、Message Resolver 的
    # “无 segment 时纯文本兜底”、正常消息日志。真正的不变量是
    # “白名单通过之后才读正文”：未授权群的 plaintext_calls 仍然是 0。
    assert ev.plaintext_calls == 3
    assert calls["answer"] == 1
    assert ai_chat.finished == ["stub-answer"]

    ev = FakeEvent(222, 1002, 999, "@bot hi")
    await _call_handler(ai.handle, ev)
    assert calls["answer"] == 2, "第二个白名单群同样可用"

    # 白名单群只 @ → 保持 v0.1 行为（提示语，不调 API）
    ev = FakeEvent(111, 1001, 999, "@bot")
    await _call_handler(ai.handle, ev)
    assert ai_chat.finished[-1] == "有什么想问我的？"
    assert calls["answer"] == 2
    # @-only 的提示语也写进 Context（assistant 只写一次）：
    # 这样 DIRECT 刚结束时 has_recent_bot_message 生效，ambient 不会马上插话。
    assert await _count(
        "messages", "group_id = 111 AND role = 'assistant' AND content = '有什么想问我的？'"
    ) == 1

    # ===== DIRECT 取消 pending AMBIENT（含 @-only 提前返回路径） =====
    from services.group_conversation import get_group_conversation_state

    gstate = get_group_conversation_state(111)

    async def _pending():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

    # ambient 等待中，用户 @夜子 → pending 立即取消，只执行 DIRECT
    pending_ambient = asyncio.create_task(_pending())
    gstate.ambient_pending_task = pending_ambient
    ev = FakeEvent(111, 1001, 999, "@bot hello")
    await _call_handler(ai.handle, ev)
    await asyncio.sleep(0.02)  # 让 cancel 请求在事件循环里完成投递
    assert pending_ambient.cancelled(), "DIRECT 应取消 pending ambient"
    assert gstate.ambient_pending_task is None
    await asyncio.gather(pending_ambient, return_exceptions=True)

    # 只 @ 没有正文的提前返回路径同样取消 pending
    pending_ambient = asyncio.create_task(_pending())
    gstate.ambient_pending_task = pending_ambient
    ev = FakeEvent(111, 1001, 999, "@bot")
    await _call_handler(ai.handle, ev)
    await asyncio.sleep(0.02)
    assert pending_ambient.cancelled(), "@-only 路径也应取消 pending ambient"
    assert gstate.ambient_pending_task is None
    await asyncio.gather(pending_ambient, return_exceptions=True)

    # ===== v0.5 → v0.7 DIRECT 多模态（授权群；AI 层桩替代） =====
    import services.vision as vision_module
    from services.vision import VISION_ENABLED as _VI

    assert _VI is True  # conftest 默认开启
    # 纯文本 @ 行为与 v0.4 完全一致：不出现任何图片输入
    ev = FakeEvent(111, 1001, 999, "@bot hello")
    await _call_handler(ai.handle, ev)
    assert calls["last_question"] == "hello"
    assert calls["last_resolved"].image_accepted == 0
    assert calls["last_conversation"].has_any_image is False

    # @夜子 + 1 张图片 + 文字 → 进入 pipeline，图片随问题一起交给模型层
    ev = FakeEvent(
        111,
        1001,
        999,
        "@bot 你觉得这个怎么样",
        segments=[SimpleNamespace(type="image", data={"url": "http://x/1.jpg"})],
    )
    await _call_handler(ai.handle, ev)
    assert calls["last_question"] == "你觉得这个怎么样"
    assert calls["last_resolved"].image_total == 1
    assert calls["last_resolved"].image_accepted == 1
    assert calls["last_conversation"].has_any_image is True

    # @夜子 + 纯图片 → 不能回复旧的「有什么想问我的？」，必须进入 LLM pipeline
    ev = FakeEvent(
        111,
        1001,
        999,
        "@bot",
        segments=[SimpleNamespace(type="image", data={"url": "http://x/2.jpg"})],
    )
    await _call_handler(ai.handle, ev)
    assert ai_chat.finished[-1] == "stub-answer", "纯图片应进入 AI pipeline 而不是提示语"
    assert calls["last_question"] == ""
    assert calls["last_resolved"].image_total == 1
    assert calls["last_resolved"].image_accepted == 1

    # 多图：顺序保持、不超过 VISION_MAX_IMAGES（默认 4）
    ev = FakeEvent(
        111,
        1001,
        999,
        "@bot 比较一下这两张图",
        segments=[
            SimpleNamespace(type="image", data={"url": f"http://x/m{i}.jpg"})
            for i in range(6)
        ],
    )
    await _call_handler(ai.handle, ev)
    assert calls["last_resolved"].image_total == 6
    assert calls["last_resolved"].image_accepted == 4
    # 顺序保持：归一化 items 里的图片顺序与消息内顺序一致（绝不被重排）
    accepted_urls = [
        item.image.url
        for item in calls["last_resolved"].message.items
        if getattr(item, "type", "") == "image" and item.image is not None
    ]
    assert accepted_urls == [
        "http://x/m0.jpg",
        "http://x/m1.jpg",
        "http://x/m2.jpg",
        "http://x/m3.jpg",
    ]

    # 无有效 URL 的图片段被拒绝：有文字时照常走纯文本 pipeline
    ev = FakeEvent(
        111,
        1001,
        999,
        "@bot 看看这个",
        segments=[SimpleNamespace(type="image", data={"file": "broken.img"})],
    )
    await _call_handler(ai.handle, ev)
    assert calls["last_resolved"].image_total == 1
    assert calls["last_resolved"].image_accepted == 0
    assert calls["last_conversation"].has_any_image is False

    # VISION_ENABLED=false + 纯图片 → 稳定降级回复，且不进入 LLM pipeline
    # v0.7：Message Resolver 在实例化时读取 vision 模块的开关，
    # 因此这里显式改 vision.VISION_ENABLED（与生产代码修改的是同一处配置）。
    before = calls["answer"]
    ai.VISION_ENABLED = False
    vision_module.VISION_ENABLED = False
    ev = FakeEvent(
        111,
        1001,
        999,
        "@bot",
        segments=[SimpleNamespace(type="image", data={"url": "http://x/3.jpg"})],
    )
    await _call_handler(ai.handle, ev)
    assert ai_chat.finished[-1] == "我现在看不到图片。"
    assert calls["answer"] == before, "视觉关闭的纯图片请求不得调用模型"
    ai.VISION_ENABLED = True
    vision_module.VISION_ENABLED = True

    # VISION_ENABLED=false + 图片 + 文字 → 纯文本正常回答（行为稳定降级）
    ai.VISION_ENABLED = False
    vision_module.VISION_ENABLED = False
    ev = FakeEvent(
        111,
        1001,
        999,
        "@bot 上面这句话什么意思",
        segments=[SimpleNamespace(type="image", data={"url": "http://x/4.jpg"})],
    )
    await _call_handler(ai.handle, ev)
    assert calls["last_question"] == "上面这句话什么意思"
    assert calls["last_resolved"].image_accepted == 0
    assert ai_chat.finished[-1] == "stub-answer"
    ai.VISION_ENABLED = True
    vision_module.VISION_ENABLED = True

    # 白名单群普通消息 → 正常写入 messages + users（context_recorder 永不写关系/记忆）
    ev = FakeEvent(111, 1003, 999, "hello from allowed group")
    await _call_handler(cr.handle, ev)
    assert await _count("messages", "group_id = 111 AND content = 'hello from allowed group'") == 1
    assert await _count("users", "user_id = 1003") == 1
    assert await _count("relationships", "user_id IN (1001, 1002, 1003)") == 0
    assert await _count(
        "user_memories", "user_id IN (1001, 1002, 1003)"
    ) == 0

    # 数据库连接不在这里关闭：tests/conftest.py 的 session 级 autouse fixture
    # 会在同一事件循环上统一关闭（在这里 close_db 会永久锁死后续测试的懒恢复）。


async def test_unauthorized_groups_have_no_side_effects():
    """未授权群零副作用；授权群正常（全部桩化，不连真实 QQ / AI）。

    跑在 pytest-asyncio 的 session 事件循环上：nonebot.init() 与 SQLite
    连接都与其它异步测试共用同一 loop，session 结束统一关闭数据库。
    """
    await _gate_scenario()
