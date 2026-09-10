"""AMBIENT 插件（v0.4）：没有 @机器人 时，偶尔自然加入正在进行的群聊。

- 触发源是普通 GroupMessageEvent（priority=30，晚于 context_recorder 的 20，
  因此触发消息已先入库，ambient 读到的历史包含它）；
- @ 消息被 ai_chat（priority=10, block=True）拦截，\\debug 被 debug 插件
  （priority=1, block=True）拦截，都到不了这里；
- 群白名单（fail-closed）：未授权群直接丢弃，不读正文、不回复、不调用任何 AI；
- 业务逻辑全部在 services/ambient.py（可测试），本插件只做事件门禁与转发。
"""

from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent

from services.ambient import AMBIENT_ENABLED
from services.ambient import on_group_message
from services.group_access import is_group_allowed

# priority=30：晚于 context_recorder（20）；block=False：不阻断后续处理器。
ambient = on_message(priority=30, block=False)


@ambient.handle()
async def handle(event: GroupMessageEvent):
    # 0. 群访问白名单（fail-closed）：未授权群不读正文、不回复、不调用 AI。
    if not is_group_allowed(event.group_id):
        return
    # 防御 reportSelfMessage：机器人自己的消息绝不触发插话。
    if event.user_id == event.self_id:
        return
    # 总开关（默认关闭）：未开启时本插件完全静默。
    if not AMBIENT_ENABLED:
        return
    await on_group_message(event)
