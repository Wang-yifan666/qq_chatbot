"""QQ AI 聊天机器人入口。

职责：
1. 加载 .env 环境变量
2. 初始化 NoneBot2
3. 注册 OneBot V11 适配器
4. 启动前检查关键配置（所选 AI 服务商的 API Key）
5. 加载 plugins/ 下的插件并启动

运行方式（在项目根目录执行）：
    python bot.py
"""

import os
import sys

if sys.platform == "win32":
    # 中文 Windows 的控制台默认按 GBK 解码输出，强制使用 UTF-8，避免中文日志乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import nonebot
from dotenv import load_dotenv
from nonebot import logger
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

from services import redact_secrets
from services.database import DB_PATH
from services.database import close_db
from services.database import init_db

# 1. 把 .env 中的配置加载到环境变量。
#    必须在读取任何配置之前执行，这样 NoneBot2 和 DeepSeek 客户端都能读到配置。
load_dotenv()

# 2. 初始化 NoneBot2。
#    会自动读取 .env 中的 DRIVER / HOST / PORT / ONEBOT_ACCESS_TOKEN 等配置。
nonebot.init()

driver = nonebot.get_driver()

# 3. 注册 OneBot V11 适配器。
#    注册后 NoneBot2 会监听 ws://127.0.0.1:8080/onebot/v11/ws，
#    等待 NapCat 主动连接（反向 WebSocket）。
driver.register_adapter(OneBotV11Adapter)

# 4. 启动前检查关键配置：缺少所选 AI 服务商的 API Key 时给出明确提示并退出，
#    而不是等到群里来消息时才报出难以理解的错误。
#    主服务商由 .env 的 AI_PROVIDER 决定（deepseek | zhipu）；
#    AI_FALLBACK 为可选备用服务商（调用失败时自动降级），留空表示不降级。
VALID_PROVIDERS = ("deepseek", "zhipu")

AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").strip().lower()
if AI_PROVIDER not in VALID_PROVIDERS:
    logger.error(
        "[AI CHAT] 未知的 AI_PROVIDER：{}！可选值只有 deepseek 或 zhipu，请检查 .env 配置。",
        AI_PROVIDER,
    )
    sys.exit(1)

AI_FALLBACK = (os.getenv("AI_FALLBACK", "") or "").strip().lower()
if AI_FALLBACK:
    if AI_FALLBACK not in VALID_PROVIDERS:
        logger.error(
            "[AI CHAT] 未知的 AI_FALLBACK：{}！可选值只有 deepseek 或 zhipu（或留空表示不降级）。",
            AI_FALLBACK,
        )
        sys.exit(1)
    if AI_FALLBACK == AI_PROVIDER:
        logger.error(
            "[AI CHAT] AI_FALLBACK 不能与 AI_PROVIDER 相同（都是 {}），请检查 .env 配置。",
            AI_FALLBACK,
        )
        sys.exit(1)

# 依次检查用到的每个服务商是否配置了 API Key
for provider in [AI_PROVIDER] + ([AI_FALLBACK] if AI_FALLBACK else []):
    key_env = "ZHIPU_API_KEY" if provider == "zhipu" else "DEEPSEEK_API_KEY"
    role = "主" if provider == AI_PROVIDER else "备用"
    if not os.getenv(key_env):
        logger.error(
            "[AI CHAT] 未检测到 {}！{} 作为{}服务商需要该 Key，"
            "请在 .env 中填写后重新启动。",
            key_env,
            provider,
            role,
        )
        sys.exit(1)

# 5. close 用户配置校验（必须在 load_dotenv 之后执行）：
#    CLOSE_USER_ID 是 close 关系的唯一真相来源；空 = 没有 close 用户；
#    非法值（非数字）在启动阶段直接报错退出，而不是运行到聊天时才暴露。
#    注意：日志中绝不输出真实 CLOSE_USER_ID。
try:
    from services.relationship_service import CLOSE_USER_ID
except ValueError as exc:
    logger.error(
        "[RELATIONSHIP] {}，请检查 .env 的 CLOSE_USER_ID 配置。",
        redact_secrets(str(exc)),
    )
    sys.exit(1)

if CLOSE_USER_ID is not None:
    logger.info("[RELATIONSHIP] close target configured")
else:
    logger.info("[RELATIONSHIP] 未配置 close 用户（CLOSE_USER_ID 为空）")

# 6. 加载 plugins/ 目录下的全部插件（ai_chat / context_recorder）。
nonebot.load_plugins("plugins")


# 7. 数据库生命周期钩子（NoneBot2 2.5.0 提供 driver.on_startup / on_shutdown，
#    见 nonebot/internal/driver/_lifespan.py；API 已对照本仓库安装版本确认）。
@driver.on_startup
async def _init_chat_history_db() -> None:
    """启动时初始化 SQLite（自动建 data/ 目录、库文件与全部表：
    messages / users / relationships / user_memories）。

    初始化失败只记录清晰 ERROR 日志，Bot 继续以“无上下文单轮问答”模式运行，
    且运行中每次读写会再尝试懒恢复。
    """
    try:
        await init_db()
        logger.info("[CONTEXT] SQLite 存储已就绪：{}", DB_PATH)
    except Exception as exc:
        logger.error(
            "[CONTEXT] SQLite 初始化失败，群聊上下文 / 用户 / 关系 / 记忆功能"
            "暂时不可用（单轮问答不受影响）：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )


@driver.on_shutdown
async def _close_chat_history_db() -> None:
    """进程退出前关闭数据库连接。"""
    await close_db()


if __name__ == "__main__":
    # 启动 NoneBot2（阻塞运行，Ctrl+C 退出）
    nonebot.run()
