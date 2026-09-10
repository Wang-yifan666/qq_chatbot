"""群白名单 + handler 集成测试（v0.3.1）。

用构造的伪 OneBot 事件直接调用插件 handler（复用 scripts/test_group_access.py
的成熟做法，改写为 pytest 用例）：
- 未授权群：不读取消息正文、不回复、不调用 AI、不写
  messages / users / relationships / user_memories；
- 授权群：正常进入流程，AI 调用与回复通道用桩替代，
  绝不访问真实 QQ / LLM / 搜索服务。
"""

from types import SimpleNamespace

from nonebot.exception import FinishedException

class FakeEvent:
    """伪 OneBot GroupMessageEvent。

    get_plaintext 模拟 onebot v11 真实行为：去掉开头的 @机器人 本体；
    plaintext_calls 用于断言“未授权群根本没读取正文”。
    """

    def __init__(self, group_id: int, user_id: int, self_id: int, text: str):
        self.group_id = group_id
        self.user_id = user_id
        self.self_id = self_id
        self._text = text
        self.plaintext_calls = 0
        self.to_me = False
        self.sender = SimpleNamespace(user_id=user_id, nickname="小明", card="")

    def get_plaintext(self) -> str:
        self.plaintext_calls += 1
        text = self._text
        if text.startswith("@bot"):
            text = text[len("@bot") :].strip()
        return text


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


async def _call_handler(handler, event) -> None:
    """直接调用插件 handler；吞掉与 NoneBot 运行环境一致的 FinishedException。"""
    try:
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
    """完整门禁场景：未授权群零副作用 + 授权群正常流程（全部桩化）。"""
    import nonebot

    nonebot.init()

    import services.database as dbm
    from plugins import ai_chat as ai
    from plugins import ambient as amb
    from plugins import context_recorder as cr
    from plugins import debug as dg

    await dbm.init_db()

    # --- 替换 AI 调用与回复通道（绝不打真实 API） ---
    calls = {"answer": 0}

    async def fake_answer(event, question: str) -> str:
        calls["answer"] += 1
        return "stub-answer"

    ai._answer = fake_answer
    ai_chat = FakeChat()
    ai.chat = ai_chat
    dg_chat = FakeChat()
    dg.debug = dg_chat

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

    # 普通消息 → 不写 messages / users / relationships / user_memories
    ev = FakeEvent(333, 1003, 999, "hello from unauthorized group")
    await _call_handler(cr.handle, ev)
    assert await _count("messages", "group_id = 333") == 0
    assert await _count("users", "user_id = 1003") == 0
    assert await _count("relationships") == 0
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
    assert ev.plaintext_calls == 1
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

    # 白名单群普通消息 → 正常写入 messages + users（context_recorder 永不写关系/记忆）
    ev = FakeEvent(111, 1003, 999, "hello from allowed group")
    await _call_handler(cr.handle, ev)
    assert await _count("messages", "group_id = 111 AND content = 'hello from allowed group'") == 1
    assert await _count("users", "user_id = 1003") == 1
    assert await _count("relationships") == 0
    assert await _count("user_memories") == 0

    # 数据库连接不在这里关闭：tests/conftest.py 的 session 级 autouse fixture
    # 会在同一事件循环上统一关闭（在这里 close_db 会永久锁死后续测试的懒恢复）。


async def test_unauthorized_groups_have_no_side_effects():
    """未授权群零副作用；授权群正常（全部桩化，不连真实 QQ / AI）。

    跑在 pytest-asyncio 的 session 事件循环上：nonebot.init() 与 SQLite
    连接都与其它异步测试共用同一 loop，session 结束统一关闭数据库。
    """
    await _gate_scenario()
