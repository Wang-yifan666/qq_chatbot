"""群聊上下文记录插件（v0.2 → v0.3.1 群白名单）：把白名单群里所有纯文本消息写入 SQLite。

职责边界：
- 只负责“记录”，永不回复、永不调用 AI、不产生任何模型费用；
- 记录对象是白名单群里所有纯文本消息，即使没有 @机器人
  （这是“真群聊上下文”的前提：A/B 的讨论也要入库，机器人才能理解“那”指什么）；
- 群访问白名单（fail-closed）：处理器第一行就执行 is_group_allowed(event.group_id)，
  未授权群的任何消息直接丢弃——不写 messages / users / relationships /
  user_memories，也不读取消息正文。这是隐私边界：即使机器人被意外拉进陌生群，
  也不会在后台收集该群的聊天内容；
- 不负责 @机器人 的消息：ai_chat 的匹配器 priority=10、block=True，
  本匹配器 priority=20、block=False。@ 消息被 ai_chat 拦截并由它自己保存
  （未授权群时由 ai_chat 的白名单检查拦截丢弃），保证每条消息最多入库一次；
- 防御性跳过机器人自身消息（event.user_id == event.self_id）：即使 NapCat
  开启 reportSelfMessage，机器人回复也不会被重复保存
  （机器人回答由 ai_chat 以 role=assistant 主动保存）。

priority / block 语义已对照本仓库安装的 NoneBot2 2.5.0 源码验证
（nonebot/message.py）：事件按 priority 从小到大依次检查；
matcher.block=True 且已运行时抛 StopPropagation，更低优先级不再执行。
"""

from nonebot import logger
from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent

from services.context_store import add_message
from services.group_access import is_group_allowed
from services.prompt_builder import sender_display_name
from services.user_store import upsert_user

# 优先级 20 > ai_chat 的 10：普通非 @ 群消息轮到本插件入库；
# @机器人 的消息已被 ai_chat 拦截（block=True），不会走到这里重复保存。
recorder = on_message(priority=20, block=False)


@recorder.handle()
async def handle(event: GroupMessageEvent):
    # handler 参数声明为 GroupMessageEvent：私聊等非群消息事件不匹配本 handler
    # （NoneBot2 事件参数按类型过滤，见 nonebot/internal/params.py 的 EventParam）

    # 0. 群访问白名单（fail-closed）：未授权群的任何消息直接丢弃，
    #    不读取消息内容、不写任何表。权限判断必须在一切 DB 副作用之前。
    if not is_group_allowed(event.group_id):
        return

    # 防御 reportSelfMessage：机器人自己的消息不记录（回答由 ai_chat 保存）
    if event.user_id == event.self_id:
        return

    # 只记录纯文本；纯图片 / 表情 / 文件等没有文本的消息跳过
    content = event.get_plaintext().strip()
    if not content:
        return

    logger.debug(
        "[CONTEXT] 记录群消息 group_id={} user_id={} content={}",
        event.group_id,
        event.user_id,
        content,
    )

    # 顺带记录用户身份（user_id 稳定身份 + 最近显示名）；
    # 失败只记 ERROR（user_store 内部处理），不影响消息入库
    await upsert_user(event.user_id, sender_display_name(event))

    # 保存失败只记 ERROR（context_store 内部处理），不影响任何消息流转
    await add_message(
        group_id=event.group_id,
        user_id=event.user_id,
        nickname=sender_display_name(event),
        role="user",
        content=content,
    )
