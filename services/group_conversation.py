"""群会话共享状态（v0.4）：DIRECT / AMBIENT / SCHEDULED 三种模式共用。

三种模式都可能往同一个群发消息，并且都会读写同群的 Context：
- DIRECT（ai_chat，@消息）处理时持锁；
- AMBIENT（群聊事件主动插话）与 SCHEDULED（定时任务）也持同一把锁，
  避免 08:00 定时消息与用户 @ 的回复并发写 Context / 并发发消息。

锁按需创建：dict 读写没有 await，在事件循环内原子，不需要额外的保护锁。
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field


@dataclass
class GroupConversationState:
    """一个群的共享会话状态。

    lock：三种模式共用的串行锁（谁先拿到谁先执行）。
    ambient_*：AMBIENT 模式的防抖 / 频率状态，由 services/ambient.py 维护，
    其他模式不读写。
    """

    group_id: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # AMBIENT：正在等待安静期的决策任务（新消息到达时取消并重排）
    ambient_pending_task: "asyncio.Task | None" = None
    # AMBIENT：本群最近一次插话的 monotonic 时间戳（用于每小时上限），只保留最近 cap 个
    ambient_sent_at: list[float] = field(default_factory=list)

    def note_ambient_sent(self, monotonic_now: float, max_per_hour: int) -> None:
        """记录一次 AMBIENT 插话：清掉 1 小时前的旧记录，只保留最近 max_per_hour 个。"""
        self.ambient_sent_at = [t for t in self.ambient_sent_at if monotonic_now - t < 3600.0]
        self.ambient_sent_at.append(monotonic_now)
        if max_per_hour > 0 and len(self.ambient_sent_at) > max_per_hour:
            self.ambient_sent_at = self.ambient_sent_at[-max_per_hour:]

    def ambient_hourly_count(self, monotonic_now: float) -> int:
        """最近 1 小时内的 AMBIENT 插话次数。"""
        return sum(1 for t in self.ambient_sent_at if monotonic_now - t < 3600.0)


_states: dict[int, GroupConversationState] = {}


def get_group_conversation_state(group_id: int) -> GroupConversationState:
    """获取某个群的共享会话状态；首次访问时创建。"""
    state = _states.get(group_id)
    if state is None:
        state = _states[group_id] = GroupConversationState(group_id=group_id)
    return state
