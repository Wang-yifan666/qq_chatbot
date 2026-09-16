"""POKE 插件（v0.6）：QQ 群聊“戳一戳 / 拍一拍”→ services/poke.py。

- 使用 nonebot-adapter-onebot.v11 自带的 PokeNotifyEvent（OneBot v11 notice：
  notice_type=notify, sub_type=poke），不自己解析原始 JSON；
- 只处理：群聊 poke 且 target_id == 当前机器人 self_id（“有人戳机器人”）；
  机器人戳别人（user_id == self_id）/ 群友互戳（target 不是机器人）全部忽略，
  绝不形成 poke 循环；
- 第一版只支持群聊 poke：私聊 poke（group_id 为 None）忽略；
- 群白名单（fail-closed）在处理器最前面：非白名单群不读任何数据、不建立用户、
  不调用模型、不写数据库、不发送文字、不执行 group_poke、不产生任何额外副作用；
- 业务逻辑全部在 services/poke.py（可测试），本插件只做事件门禁与转发。
"""

from nonebot import logger
from nonebot import on_notice
from nonebot.adapters.onebot.v11 import PokeNotifyEvent

from services.group_access import is_group_allowed
from services.poke import POKE_ENABLED
from services.poke import on_group_poke

# notice 事件匹配器：adapter 已把 notice_type=notify + sub_type=poke 解析为
# PokeNotifyEvent（v2.4.6 自带），handler 的类型标注即事件过滤。
poke = on_notice(priority=10, block=False)


@poke.handle()
async def handle(event: PokeNotifyEvent):
    # 0. 群访问白名单（fail-closed）最先判断：日志只输出群号，
    #    绝不输出该群的任何事件细节。
    if not is_group_allowed(event.group_id):
        logger.info("[POKE] 未授权群忽略 group_id={}", event.group_id)
        return

    # 1. 总开关（默认开启；关闭时本插件完全静默）。
    if not POKE_ENABLED:
        return

    # 2. 第一版只处理群聊 poke：私聊 poke 的 group_id 为 None，直接忽略。
    if event.group_id is None:
        return

    # 3. 事件过滤（关键语义）：
    #    - target_id != self_id → 被戳的不是机器人（群友互戳 / 机器人戳别人）→ 忽略；
    #    - user_id == self_id → 机器人自己戳别人产生的通知 → 忽略，
    #      杜绝机器人之间或自身触发的 poke 循环。
    if event.target_id != event.self_id or event.user_id == event.self_id:
        return

    await on_group_poke(event.group_id, event.user_id)
