"""services/group_conversation.py：DIRECT / AMBIENT / SCHEDULED 共享状态测试（v0.4）。"""

import asyncio

from services.group_conversation import get_group_conversation_state


def test_same_group_returns_same_state_and_lock():
    state_a = get_group_conversation_state(111)
    state_b = get_group_conversation_state(111)
    assert state_a is state_b
    assert state_a.lock is state_b.lock


def test_different_groups_are_independent():
    state_a = get_group_conversation_state(111)
    state_b = get_group_conversation_state(222)
    assert state_a is not state_b
    assert state_a.lock is not state_b.lock


def test_ambient_hourly_count():
    state = get_group_conversation_state(333)
    now = 1_000_000.0
    state.note_ambient_sent(now, max_per_hour=3)
    state.note_ambient_sent(now + 1.0, max_per_hour=3)
    assert state.ambient_hourly_count(now + 2.0) == 2
    # 1 小时前的记录过期
    assert state.ambient_hourly_count(now + 3601.0) == 0


def test_ambient_sent_list_is_capped():
    state = get_group_conversation_state(444)
    now = 2_000_000.0
    for i in range(5):
        state.note_ambient_sent(now + i, max_per_hour=3)
    assert len(state.ambient_sent_at) == 3


async def test_shared_lock_serializes_across_modes():
    """拿到同一把锁的第二个协程必须等第一个释放（DIRECT/SCHEDULED 不并发的机制）。"""
    state = get_group_conversation_state(555)
    order: list[str] = []

    async def holder():
        async with state.lock:
            order.append("first")
            await asyncio.sleep(0.1)

    async def waiter():
        async with state.lock:
            order.append("second")

    t1 = asyncio.create_task(holder())
    await asyncio.sleep(0.02)
    t2 = asyncio.create_task(waiter())
    await asyncio.gather(t1, t2)
    assert order == ["first", "second"]
