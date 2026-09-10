"""主动发送（v0.4）：无消息事件时的 QQ 群消息发送 + 回答入库。

SCHEDULED / AMBIENT 两种主动模式共用：
- 获取当前连接中的 OneBot V11 Bot（没有 GroupMessageEvent 也能发消息）；
- 调 send_group_msg 主动发消息；
- 成功后把完整实际发送内容以 role=assistant 写入 Context（messages 表），
  这样随后的「@夜子 你刚刚说什么」DIRECT 流程能看到刚才的主动发言；
- 重复入库防御：context_recorder 已跳过 self_id 消息，且这里只在发送成功后
  主动写一次，NapCat 的 self-message 回报不会再写第二遍。
"""

from nonebot import get_bots
from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot

from services import redact_secrets
from services.context_store import add_message
from services.prompt_builder import BOT_NAME


def get_onebot_bot() -> Bot | None:
    """返回当前连接中的任意一个 OneBot V11 Bot；没有连接返回 None。"""
    for bot in get_bots().values():
        if isinstance(bot, Bot):
            return bot
    return None


async def send_group_message(bot: Bot, group_id: int, text: str) -> bool:
    """主动发送一条群消息。失败只记日志并返回 False（绝不抛出）。"""
    try:
        await bot.send_group_msg(group_id=group_id, message=text)
        return True
    except Exception as exc:
        logger.error(
            "[PROACTIVE] 发送群消息失败 (group_id={}): {}: {}",
            group_id,
            type(exc).__name__,
            redact_secrets(str(exc)),
        )
        return False


async def save_assistant_message(group_id: int, self_id: int | None, text: str) -> bool:
    """把主动发送的内容写进 Context（role=assistant，与 DIRECT 回答同一张表）。

    self_id 取不到时落 0：assistant 行的 user_id 只用于区分角色，不参与用户身份。
    """
    return await add_message(
        group_id=group_id,
        user_id=self_id or 0,
        nickname=BOT_NAME,
        role="assistant",
        content=text,
    )
