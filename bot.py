"""QQ AI 聊天机器人入口。

职责：
1. 加载 .env 环境变量
2. 初始化 NoneBot2
3. 注册 OneBot V11 适配器
4. 启动前检查关键配置（所选 AI 服务商的 API Key）
   以及群聊访问白名单（ALLOWED_GROUP_IDS，fail-closed）
5. 加载 plugins/ 下的插件并启动

运行方式（在项目根目录执行）：
    python bot.py
"""

import os
import sys
import threading

if sys.platform == "win32":
    # 中文 Windows 的控制台默认按 GBK 解码输出，强制使用 UTF-8，避免中文日志乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 1. 把 .env 中的配置加载到环境变量（bootstrap 顺序约束）。
#    必须放在任何依赖环境变量的模块 import 之前：services.database 的
#    DB_PATH（CHAT_HISTORY_DB）与 services.personal_memory_store 的
#    DB_PATH（MEMORY_DB_PATH）都在模块导入阶段通过 os.getenv() 计算，
#    plugins/ 下各插件与大部分 services 模块同样在导入期读取配置。
#    因此 load_dotenv() 必须先于全部业务模块 import 执行，否则 .env
#    中的覆盖值不会生效；也不要把它散落到各模块里重复调用。
from dotenv import load_dotenv

load_dotenv()

# 2. 以下是业务 import：全部发生在 load_dotenv() 之后。
import nonebot
from nonebot import logger
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

from services import redact_secrets
from services.database import DB_PATH
from services.database import close_db
from services.database import init_db
from services.personal_memory_store import DB_PATH as MEMORY_DB_PATH
from services.personal_memory_store import close_memory_db
from services.personal_memory_store import init_memory_db

# 3. 初始化 NoneBot2。
#    会自动读取 .env 中的 DRIVER / HOST / PORT / ONEBOT_ACCESS_TOKEN 等配置。
nonebot.init()

driver = nonebot.get_driver()

# 4. 注册 OneBot V11 适配器。
#    注册后 NoneBot2 会监听 ws://127.0.0.1:8080/onebot/v11/ws，
#    等待 NapCat 主动连接（反向 WebSocket）。
driver.register_adapter(OneBotV11Adapter)

# 5. 启动前检查关键配置：缺少所选 AI 服务商的 API Key 时给出明确提示并退出，
#    而不是等到群里来消息时才报出难以理解的错误。
#    主服务商由 .env 的 AI_PROVIDER 决定（deepseek | zhipu）；
#    AI_FALLBACK 为可选备用服务商（调用失败时自动降级），留空表示不降级。
#    支持两种降级：跨服务商（如 deepseek→zhipu），
#    以及同服务商双模型（如 deepseek 主模型→deepseek 备用模型，
#    需配置 AI_FALLBACK_MODEL 且与主模型不同）。
VALID_PROVIDERS = ("deepseek", "zhipu")

# 各服务商默认模型（与 services/deepseek.py / zhipu.py 保持一致，仅用于启动校验）
# DeepSeek 当前推荐模型：deepseek-flash（V4.1 Flash，原生支持 text + image）
_PROVIDER_DEFAULT_MODELS = {"deepseek": "deepseek-flash", "zhipu": "glm-4.7-flash"}


def _effective_primary_model(provider: str) -> str:
    """主模型：AI_MODEL 优先，否则服务商默认模型。"""
    generic = (os.getenv("AI_MODEL") or "").strip()
    if generic:
        return generic
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL") or _PROVIDER_DEFAULT_MODELS["deepseek"]
    return os.getenv("ZHIPU_MODEL") or _PROVIDER_DEFAULT_MODELS["zhipu"]


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
        # 同服务商降级：必须配置备用模型，且与主模型不同
        fallback_model = (os.getenv("AI_FALLBACK_MODEL") or "").strip()
        primary_model = _effective_primary_model(AI_PROVIDER)
        if not fallback_model:
            logger.error(
                "[AI CHAT] AI_FALLBACK 与 AI_PROVIDER 相同（都是 {}）时属于同服务商双模型降级，"
                "必须在 .env 中配置 AI_FALLBACK_MODEL（备用模型名）。",
                AI_FALLBACK,
            )
            sys.exit(1)
        if fallback_model == primary_model:
            logger.error(
                "[AI CHAT] AI_FALLBACK_MODEL={} 与主模型 {} 相同，"
                "同服务商降级需要两个不同的模型。",
                fallback_model,
                primary_model,
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

# 6. close 用户配置校验（必须在 load_dotenv 之后执行）：
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

# 6.5 群聊访问白名单校验（fail-closed）：
#     解析与合法性检查集中在 services/group_access.py；非法配置（如 111,abc）
#     在启动阶段报 ERROR 并退出，而不是等第一条群消息才暴露。
#     日志只输出群数量，不打印真实 QQ 群号。
try:
    from services.group_access import ALLOWED_GROUP_IDS
    from services.group_access import ALLOW_ALL_GROUPS
except ValueError as exc:
    logger.error("[GROUP ACCESS] {}", redact_secrets(str(exc)))
    sys.exit(1)

if ALLOW_ALL_GROUPS:
    logger.info("[GROUP ACCESS] all groups are allowed")
elif not ALLOWED_GROUP_IDS:
    # 用户可能故意暂时关闭机器人：只 WARNING，不退出
    logger.warning(
        "[GROUP ACCESS] no allowed groups configured; all group messages will be ignored"
    )
else:
    logger.info("[GROUP ACCESS] allowed groups configured: {}", len(ALLOWED_GROUP_IDS))

# 6.6 定时任务（v0.4）：加载官方生态的 APScheduler 插件，再校验 + 注册 Scheduled Task。
#     MORNING_GREETING_TIME / MORNING_GREETING_GROUP_IDS 等非法配置在启动阶段
#     ValueError → 明确报错退出（与白名单校验同一原则）。
try:
    nonebot.load_plugin("nonebot_plugin_apscheduler")
except Exception as exc:
    logger.error(
        "[SCHEDULED] 加载 nonebot_plugin_apscheduler 失败（请先执行 "
        "pip install -r requirements.txt）：{}: {}",
        type(exc).__name__,
        redact_secrets(str(exc)),
    )
    sys.exit(1)

try:
    from services.scheduled_tasks import setup_scheduled_tasks
except ValueError as exc:
    logger.error("[SCHEDULED] {}", redact_secrets(str(exc)))
    sys.exit(1)

setup_scheduled_tasks(driver)

# 7. 加载 plugins/ 目录下的全部插件（ai_chat / context_recorder / ambient）。
nonebot.load_plugins("plugins")


# 8. 数据库生命周期钩子（NoneBot2 2.5.0 提供 driver.on_startup / on_shutdown，
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
    await close_memory_db()


@driver.on_startup
async def _init_memory_db() -> None:
    """启动时初始化 Personal Memory 库（data/qq_ai_bot.db）。

    失败只记录清晰 ERROR 日志：\\debug 命令与 Mini-RAG 暂不可用，
    普通聊天完全不受影响（ai_chat 的检索步骤会自动降级为无记忆对话）。
    """
    try:
        await init_memory_db()
        logger.info("[MEMORY] Personal Memory 数据库已就绪：{}", MEMORY_DB_PATH)
    except Exception as exc:
        logger.error(
            "[MEMORY] Personal Memory 数据库初始化失败"
            "（调试命令与 Mini-RAG 暂不可用，普通聊天不受影响）：{}: {}",
            type(exc).__name__,
            redact_secrets(str(exc)),
        )


@driver.on_startup
async def _warmup_persona_rag() -> None:
    """启动时后台预热 Persona RAG（加载 embedding 模型 + 本地索引）。

    embedding 模型首次加载需要数秒：提前在后台线程预热，第一条 @ 消息
    就不会被模型加载阻塞。失败只记 ERROR 日志：Persona RAG 自动禁用，
    普通聊天完全不受影响。模型只加载一次，运行中不会重复加载。
    """

    def _warm() -> None:
        from services.persona_rag import warmup

        warmup()

    threading.Thread(target=_warm, daemon=True, name="persona-rag-warmup").start()


if __name__ == "__main__":
    # 启动 NoneBot2（阻塞运行，Ctrl+C 退出）
    nonebot.run()
