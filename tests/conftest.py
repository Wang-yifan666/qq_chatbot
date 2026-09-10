"""pytest 全局配置（v0.3.1 测试基线）。

- 把项目根目录加入 sys.path（tests/ 与 services/ 平级，pytest 默认只加 tests/）；
- 在 import 任何业务模块之前设置隔离的环境变量：全部 SQLite 库指向临时目录，
  绝不触碰真实的 data/ 与真实 .env；Persona RAG / 拆分回复 / web search 显式关闭；
- 全程不加载 .env、不连接 QQ / 真实 LLM / 真实搜索服务、不下载 embedding 模型。
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 一次性临时目录：所有测试 SQLite 库都写在这里（进程结束由系统清理）
_TMP_DIR = tempfile.mkdtemp(prefix="qq_bot_pytest_")

# 必须在业务模块 import 之前设置（services/* 大多在 import 期读取环境变量）
os.environ["ALLOWED_GROUP_IDS"] = "111, 222"
os.environ["CHAT_HISTORY_DB"] = str(Path(_TMP_DIR) / "chat_history.db")
os.environ["MEMORY_DB_PATH"] = str(Path(_TMP_DIR) / "qq_ai_bot.db")
os.environ["CLOSE_USER_ID"] = ""
os.environ["DEBUG_ADMIN_QQ"] = ""
os.environ["SPLIT_REPLY_ENABLED"] = "false"
os.environ["PERSONA_RAG_ENABLED"] = "false"
os.environ["WEB_SEARCH_ENABLED"] = "false"
os.environ["LOG_MESSAGE_CONTENT"] = "false"
# v0.4 主动行为默认全部关闭：相关测试自行 monkeypatch 模块属性
os.environ["SCHEDULED_TASKS_ENABLED"] = "false"
os.environ["MORNING_GREETING_ENABLED"] = "false"
os.environ["AMBIENT_ENABLED"] = "false"
# v0.5 视觉：与代码默认一致地显式开启（个别用例 monkeypatch 模块属性关闭）
os.environ["VISION_ENABLED"] = "true"
os.environ["VISION_MAX_IMAGES"] = "4"
os.environ["VISION_DETAIL"] = "auto"
os.environ.setdefault("AI_PROVIDER", "deepseek")
os.environ.setdefault("AI_FALLBACK", "")


import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _close_databases_at_session_end():
    """session 结束时在同一事件循环上关闭全部 SQLite 连接。

    测试打开过 aiosqlite 连接（懒初始化）且没有逐个关闭；
    不关的话 aiosqlite 的后台工作会让 pytest 进程在测试全部通过后挂住。
    """
    yield
    from services.database import close_db
    from services.personal_memory_store import close_memory_db

    await close_db()
    await close_memory_db()
