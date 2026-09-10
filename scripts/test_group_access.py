"""群聊访问白名单验证脚本（v0.3.1，无测试框架，README 测试方法 Case 51~57）。

覆盖场景：
- 纯解析（Case 10 / 54 / 55 / 56 / 57）：parse_allowed_group_ids 的
  空值 / * / 逗号+空格 / 非法项 / 混合项；
- 启动语义（Case 8 / 9 / 54 / 55 / 56）：子进程按不同 ALLOWED_GROUP_IDS
  导入 bot 模块（不启动服务器），检查启动日志与退出码；
- 插件门禁（Case 51 / 52 / 53 / 1 / 3 / 5）：构造伪 OneBot 事件直接调用
  ai_chat / context_recorder / debug 的 handler：
  未授权群不读取消息正文、不回复、不调用 AI（假 Provider 桩计数）、
  不写 messages / users / relationships / user_memories；
  授权群正常进入流程（AI 调用被桩替换，不产生真实 API 费用）。

运行（项目根目录）：
    .venv\\Scripts\\python.exe scripts\\test_group_access.py
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

if sys.platform == "win32":
    # 与 bot.py 一致：Windows 控制台强制 UTF-8，避免中文输出乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from nonebot.exception import FinishedException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PYTHON = sys.executable

# 进程内插件测试使用的配置（必须在导入任何 services/plugins 之前设置）
_TMP_DIR = tempfile.mkdtemp(prefix="group_access_test_")
os.environ["ALLOWED_GROUP_IDS"] = "111, 222"
os.environ["CHAT_HISTORY_DB"] = str(Path(_TMP_DIR) / "test_chat_history.db")
os.environ["SPLIT_REPLY_ENABLED"] = "false"
os.environ.setdefault("AI_PROVIDER", "deepseek")
os.environ.setdefault("AI_FALLBACK", "")

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 1. 纯解析测试
# --------------------------------------------------------------------------
def test_parse() -> None:
    from services.group_access import parse_allowed_group_ids

    check("parse: 空字符串 → 禁止所有群", parse_allowed_group_ids("") == (frozenset(), False))
    check("parse: None → 禁止所有群", parse_allowed_group_ids(None) == (frozenset(), False))
    check("parse: 全空白 → 禁止所有群", parse_allowed_group_ids("   ") == (frozenset(), False))
    check("parse: '*' → 允许所有群", parse_allowed_group_ids("*") == (frozenset(), True))
    check("parse: '  *  ' → 允许所有群", parse_allowed_group_ids("  *  ") == (frozenset(), True))

    ids, allow_all = parse_allowed_group_ids("111, 222 ,333")
    check(
        "parse: 逗号两侧空格正确解析",
        ids == frozenset({111, 222, 333}) and not allow_all,
    )
    ids, allow_all = parse_allowed_group_ids("111,222")
    check("parse: 两个群号", ids == frozenset({111, 222}) and not allow_all)
    ids, allow_all = parse_allowed_group_ids("111,,222")
    check("parse: 空段忽略", ids == frozenset({111, 222}) and not allow_all)

    for bad in ("111,abc", "abc", "111, 22x", "12.5", "-1", "0", "*,111", "111, *"):
        try:
            parse_allowed_group_ids(bad)
            check(f"parse: 非法项 {bad!r} 抛 ValueError", False)
        except ValueError:
            check(f"parse: 非法项 {bad!r} 抛 ValueError", True)


# --------------------------------------------------------------------------
# 2. 子进程：不同 ALLOWED_GROUP_IDS 下的模块状态 / 启动语义
# --------------------------------------------------------------------------
def run_subprocess(code: str, group_ids: str | None, timeout: int = 300):
    """以指定 ALLOWED_GROUP_IDS 运行一段 Python 代码（cwd=项目根目录）。"""
    env = os.environ.copy()
    if group_ids is None:
        env.pop("ALLOWED_GROUP_IDS", None)
    else:
        env["ALLOWED_GROUP_IDS"] = group_ids
    result = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        timeout=timeout,
    )
    out = (result.stdout or b"").decode("utf-8", errors="replace")
    err = (result.stderr or b"").decode("utf-8", errors="replace")
    return result.returncode, out + "\n" + err


_SNIPPET = (
    "from services.group_access import ("
    "ALLOW_ALL_GROUPS, ALLOWED_GROUP_IDS, is_group_allowed);"
    "print(ALLOW_ALL_GROUPS, sorted(ALLOWED_GROUP_IDS), "
    "is_group_allowed(111), is_group_allowed(222), "
    "is_group_allowed(333), is_group_allowed(999999))"
)


def test_module_state() -> None:
    # Case 54 / 6：111、222 可用，333 忽略
    rc, out = run_subprocess(_SNIPPET, "111,222")
    ok = rc == 0 and out.strip().endswith("False [111, 222] True True False False")
    check("env=111,222: 111/222 允许、333/其他 拒绝", ok, out.strip().splitlines()[-1])

    # Case 55 / 7：* 恢复所有群可用
    rc, out = run_subprocess(_SNIPPET, "*")
    ok = rc == 0 and out.strip().endswith("True [] True True True True")
    check("env=*: 所有群允许", ok, out.strip().splitlines()[-1])

    # Case 8：留空
    rc, out = run_subprocess(_SNIPPET, "")
    ok = rc == 0 and out.strip().endswith("False [] False False False False")
    check("env 留空: 所有群拒绝", ok, out.strip().splitlines()[-1])

    # Case 8：变量不存在
    rc, out = run_subprocess(_SNIPPET, None)
    ok = rc == 0 and out.strip().endswith("False [] False False False False")
    check("env 不存在: 所有群拒绝", ok, out.strip().splitlines()[-1])

    # Case 10 / 57：空格
    rc, out = run_subprocess(_SNIPPET, "111, 222 ,333")
    ok = rc == 0 and out.strip().endswith("False [111, 222, 333] True True True False")
    check("env=111, 222 ,333: 三个群都允许", ok, out.strip().splitlines()[-1])

    # Case 9 / 56：非法配置 → 导入即 ValueError
    rc, out = run_subprocess(_SNIPPET, "111,abc")
    check(
        "env=111,abc: 导入报 ValueError 退出",
        rc != 0 and "ALLOWED_GROUP_IDS" in out and "ValueError" in out,
        f"rc={rc}",
    )


def test_startup_logs() -> None:
    # 用「导入 bot 模块」验证启动阶段校验：模块级代码 = bot.py 启动时执行的
    # 完整校验链（load_dotenv / nonebot.init / 校验 / load_plugins），
    # 只差 __main__ 里的 nonebot.run()（不启动服务器）。
    rc, out = run_subprocess("import bot", "111, 222")
    check(
        "启动: 合法配置输出 allowed groups configured: 2",
        rc == 0 and "[GROUP ACCESS] allowed groups configured: 2" in out,
        f"rc={rc}",
    )

    rc, out = run_subprocess("import bot", "*")
    check(
        "启动: * 输出 all groups are allowed",
        rc == 0 and "[GROUP ACCESS] all groups are allowed" in out,
        f"rc={rc}",
    )

    rc, out = run_subprocess("import bot", "")
    check(
        "启动: 空配置输出 WARNING 且不退出",
        rc == 0 and "[GROUP ACCESS] no allowed groups configured" in out,
        f"rc={rc}",
    )

    rc, out = run_subprocess("import bot", "111,abc")
    check(
        "启动: 非法配置 ERROR 并退出",
        rc == 1 and "[GROUP ACCESS]" in out and "ALLOWED_GROUP_IDS" in out,
        f"rc={rc}",
    )


# --------------------------------------------------------------------------
# 3. 插件门禁（构造伪 OneBot 事件直接调用 handler）
# --------------------------------------------------------------------------
class FakeEvent:
    """伪 OneBot GroupMessageEvent。

    get_plaintext 模拟 onebot v11 的真实行为：去掉开头的 @机器人 本体，
    因此 "@bot" → ""、 "@bot hello" → "hello"。
    """

    def __init__(self, group_id: int, user_id: int, self_id: int, text: str):
        self.group_id = group_id
        self.user_id = user_id
        self.self_id = self_id
        self._text = text
        self.plaintext_calls = 0
        # sender_display_name 需要 sender（authorized 路径才会读到）
        self.sender = SimpleNamespace(user_id=user_id, nickname="小明", card="")

    def get_plaintext(self) -> str:
        self.plaintext_calls += 1
        text = self._text
        if text.startswith("@bot"):
            text = text[len("@bot"):].strip()
        return text


class FakeChat:
    """替代 NoneBot Matcher 的 finish/send：记录回复，不真正发消息。

    finish 与真实 Matcher.finish 一致地抛 FinishedException
    （对照 nonebot/internal/matcher/matcher.py 2.5.0），
    保证 handler 内 finish 之后的代码在测试里同样不会执行。
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


async def test_plugin_gates() -> None:
    import nonebot

    nonebot.init()

    import services.database as dbm
    from plugins import ai_chat as ai
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

    # Case 51 / 4：非白名单群 @机器人 hello → 不读正文、不调 AI、不回复
    ev = FakeEvent(333, 1001, 999, "@bot hello")
    await _call_handler(ai.handle, ev)
    check("非白名单 @: 未读取消息正文", ev.plaintext_calls == 0)
    check("非白名单 @: 未调用 AI（无 API 费用）", calls["answer"] == 0)
    check("非白名单 @: 未回复任何内容", not ai_chat.finished and not ai_chat.sent)

    # Case 53 / 5：非白名单群只 @机器人 → 也不回复「有什么想问我的？」
    ev = FakeEvent(333, 1001, 999, "@bot")
    await _call_handler(ai.handle, ev)
    check("非白名单只 @: 未回复「有什么想问我的？」", not ai_chat.finished)
    check("非白名单只 @: 未读取正文、未调 AI", ev.plaintext_calls == 0 and calls["answer"] == 0)

    # Case 52 / 3：非白名单群普通消息 → 不写入任何表
    ev = FakeEvent(333, 1003, 999, "hello from unauthorized group")
    await _call_handler(cr.handle, ev)
    check("非白名单普通消息: messages 无记录", await _count("messages", "group_id = 333") == 0)
    check("非白名单普通消息: users 无记录", await _count("users", "user_id = 1003") == 0)
    check("非白名单普通消息: relationships 无记录", await _count("relationships") == 0)
    check("非白名单普通消息: user_memories 无记录", await _count("user_memories") == 0)

    # 非白名单群 \debug 命令 → 完全沉默（debug 插件共用同一白名单）
    ev = FakeEvent(333, 1001, 999, "\\debug status")
    await _call_handler(dg.handle, ev)
    check("非白名单 \\debug: 未读取正文、未回复", ev.plaintext_calls == 0 and not dg_chat.finished)

    # Case 2：白名单群 @机器人 → 进入 AI 流程并回复（AI 层由桩替代）
    ev = FakeEvent(111, 1001, 999, "@bot hello")
    await _call_handler(ai.handle, ev)
    check("白名单群 @: 读取正文并调用 AI 层", ev.plaintext_calls == 1 and calls["answer"] == 1)
    check("白名单群 @: 回复了 AI 结果", ai_chat.finished == ["stub-answer"])

    # Case 6：第二个白名单群同样可用
    ev = FakeEvent(222, 1002, 999, "@bot hi")
    await _call_handler(ai.handle, ev)
    check("白名单群 222 @: 同样可用", calls["answer"] == 2)

    # 白名单群只 @机器人 → 保持 v0.1 行为（不调 API，提示语）
    ev = FakeEvent(111, 1001, 999, "@bot")
    await _call_handler(ai.handle, ev)
    check(
        "白名单群只 @: 回复「有什么想问我的？」且不调 API",
        ai_chat.finished[-1] == "有什么想问我的？" and calls["answer"] == 2,
    )

    # Case 1：白名单群普通消息 → 正常写入 messages + users
    ev = FakeEvent(111, 1003, 999, "hello from allowed group")
    await _call_handler(cr.handle, ev)
    check(
        "白名单群普通消息: 写入 messages",
        await _count("messages", "group_id = 111 AND content = 'hello from allowed group'") == 1,
    )
    check("白名单群普通消息: upsert users", await _count("users", "user_id = 1003") == 1)
    check(
        "context_recorder 永不写 relationships",
        await _count("relationships") == 0,
    )
    check(
        "context_recorder 永不写 user_memories",
        await _count("user_memories") == 0,
    )

    await dbm.close_db()


async def main() -> None:
    test_parse()
    test_module_state()
    test_startup_logs()
    await test_plugin_gates()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} 项：{FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
