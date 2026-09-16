"""poke 发送服务（v0.6.1）：主动戳人 + PacketBackend 能力熔断。

v0.6.1 能力模型（把「收到戳」与「主动戳回」正式拆开）：
    POKE_ENABLED
      ├── inbound poke      收到别人戳机器人（核心能力，plugins/poke.py 门禁）
      ├── text response     LLM 短文本回复（核心能力，services/poke.py）
      └── poke_back         主动戳回（可选能力，本模块）
                            └─ 依赖 NapCat PacketBackend 发包能力；其可用性取决于
                               QQ build × CPU 架构 × NapCat 版本组合，因此必须按
                               「可能动态不可用」处理——PacketBackend 失败绝不能
                               被当作整个 POKE 功能失败。

PacketBackend 熔断（三态；几十行实现，不引入第三方库）：
    UNKNOWN ──首次真实戳回──► 正常调用 group_poke
        │ 成功 → AVAILABLE（此后正常并行调用，不再走探测锁）
        │ 明确 PB 不支持（retcode=1400 且 wording/message 含 packetbackend
        │   语义）→ OPEN（禁止调用，retry TTL 默认 30 分钟）
        │ 其它失败（普通 ActionFailed / timeout / WS 瞬断）→ 不开熔断
        ▼
    OPEN ──TTL 到期后的下一次合法戳回──► HALF_OPEN（单次探测）
        │ 成功 → AVAILABLE（记 recovered）
        │ 仍 PB 不支持 → OPEN（TTL 重置）
        │ 其它失败 → 回到 OPEN（未证明恢复；不重置 TTL，按原 retry_at 再探测）
        ▼
    HALF_OPEN：同一时刻最多一个探测（asyncio.Lock 保护），并发请求直接跳过。

设计红线：
- 绝不主动 group_poke 任何人做「健康检查」（lazy detection：首次真实
  戳回需求到来时才探测能力）；
- 绝不对 PacketBackend 失败做 send_poke 别名 fallback：换 action 名解决不了
  PacketBackend 本身不可用，只会产生双重错误日志与双倍 API 调用；
- 日志结构化且无敏感信息：不打印完整 exception / NapCat wording /
  stacktrace / 事件原文；首次 backend unavailable 记 WARNING，开熔断记
  WARNING，熔断期间的跳过记 DEBUG，恢复记 INFO。
"""

import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.exception import ActionFailed

# NapCat 支持的群聊戳一戳 API 名（OneBot 11 扩展；NapCat 也兼容 send_poke 别名，
# 这里只使用 group_poke 一个入口。注意：send_poke 同样依赖 PacketBackend，
# 对 PacketBackend 不可用做别名 fallback 是无意义且有害的，因此永不 fallback）。
_GROUP_POKE_API = "group_poke"

# PacketBackend 熔断的重试 TTL（秒）。与 POKE_*_COOLDOWN 完全独立：
# 前者是「技术故障恢复探测节奏」，后者是「社交防刷节奏」，绝不混用。
# 暂不放进 .env（避免配置膨胀）；如需调整改这里的常量。
_PACKET_BACKEND_RETRY_SECONDS = 1800


class PokeSendStatus(str, Enum):
    """一次主动戳回的发送结果分类（业务层只需要知道类别，不需要 NapCat 原文）。"""

    SUCCESS = "success"                          # 已成功戳回
    DISABLED = "disabled"                        # 配置关闭（POKE_POKE_BACK_ENABLED=false）
    BACKEND_UNAVAILABLE = "backend_unavailable"  # PacketBackend 明确不支持 / 熔断跳过
    TEMPORARY_FAILURE = "temporary_failure"      # timeout / 网络 / WS 瞬断类
    API_FAILURE = "api_failure"                  # 其它 ActionFailed（不触发熔断）


@dataclass(frozen=True)
class PokeSendResult:
    """结构化发送结果：ok 为最终判定，retcode 仅在 API 失败时携带。"""

    status: PokeSendStatus
    ok: bool
    retcode: int | None = None


def _poke_back_enabled() -> bool:
    """读取 POKE_POKE_BACK_ENABLED（默认 true；与 services/poke.py 语义一致）。

    这里独立解析一份，保证本模块（能力层）不依赖 poke.py（业务层），
    避免循环 import。
    """
    raw = (os.getenv("POKE_POKE_BACK_ENABLED") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _classify_action_failed(exc: ActionFailed) -> tuple[PokeSendStatus, int | None]:
    """保守地把 ActionFailed 分成两类：PB 明确不支持 / 其它 API 失败。

    只有「retcode == 1400 且 wording/message 明确包含 packetbackend 语义」
    才判定为 BACKEND_UNAVAILABLE（触发熔断）。1400 单独出现、或
    packetbackend 字样配上其它 retcode，都不触发熔断。
    """
    info = getattr(exc, "info", None) or {}
    retcode = info.get("retcode")
    wording = str(info.get("wording") or "").lower()
    message = str(info.get("message") or "").lower()
    if retcode == 1400 and "packetbackend" in (wording + message):
        return PokeSendStatus.BACKEND_UNAVAILABLE, int(retcode)
    return PokeSendStatus.API_FAILURE, int(retcode) if isinstance(retcode, int) else None


# ==========================================================================
# PacketBackend 三态熔断
# ==========================================================================

_STATE_UNKNOWN = "unknown"
_STATE_AVAILABLE = "available"
_STATE_OPEN = "open"
_STATE_HALF_OPEN = "half_open"


class PokeBackendBreaker:
    """PacketBackend 能力熔断（三态）。

    - AVAILABLE：正常并行调用（不走锁）；
    - OPEN：跳过调用；retry_at 到期后的下一次合法戳回转 HALF_OPEN 探测；
    - UNKNOWN / HALF_OPEN：调用前拿探测锁，同一时刻最多一个探测，
      并发请求返回 skip（绝不排队阻塞 poke 处理）；
    - 只有 SUCCESS 与 BACKEND_UNAVAILABLE 会改变长期状态；
      TEMPORARY_FAILURE / API_FAILURE 在 AVAILABLE 时不影响状态，
      在 HALF_OPEN 探测时视为「未证明恢复」回到 OPEN（不重置 TTL）。
    """

    def __init__(self, retry_after_seconds: float = _PACKET_BACKEND_RETRY_SECONDS) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.state = _STATE_UNKNOWN
        self._retry_at: float | None = None
        self._probe_lock = asyncio.Lock()

    async def acquire_attempt(self) -> tuple[bool, bool]:
        """返回 (allowed, hold_lock)。

        - allowed=False：本次必须跳过（circuit open 未到期 / 探测在途）；
        - allowed=True：允许调用；hold_lock=True 表示调用期间持有探测锁，
          调用方在 finally 中必须 release_attempt()；AVAILABLE 状态下
          hold_lock=False（正常并行，不碰锁）。
        """
        if self.state == _STATE_AVAILABLE:
            return True, False
        now = time.monotonic()
        if self.state == _STATE_OPEN and self._retry_at is not None and now < self._retry_at:
            return False, False
        # UNKNOWN / HALF_OPEN / OPEN 已到期：探测路径（同一时刻仅一个）。
        if self._probe_lock.locked():
            return False, False
        await self._probe_lock.acquire()
        # 拿到锁后再确认一次（并发窗口内的二次检查）。
        if self.state == _STATE_OPEN and self._retry_at is not None and time.monotonic() < self._retry_at:
            self._probe_lock.release()
            return False, False
        if self.state == _STATE_OPEN:
            self.state = _STATE_HALF_OPEN
            logger.info("[POKE] poke_back backend probe (half-open)")
        return True, True

    def release_attempt(self) -> None:
        self._probe_lock.release()

    def record_result(self, status: PokeSendStatus) -> None:
        previous = self.state
        if status is PokeSendStatus.SUCCESS:
            self.state = _STATE_AVAILABLE
            self._retry_at = None
            if previous in (_STATE_OPEN, _STATE_HALF_OPEN):
                logger.info("[POKE] poke_back backend recovered")
            elif previous == _STATE_UNKNOWN:
                logger.info("[POKE] poke_back backend available")
            return
        if status is PokeSendStatus.BACKEND_UNAVAILABLE:
            self._retry_at = time.monotonic() + self.retry_after_seconds
            self.state = _STATE_OPEN
            if previous == _STATE_OPEN:
                logger.warning(
                    "[POKE] poke_back circuit re-opened after failed probe retry_after_seconds={}",
                    int(self.retry_after_seconds),
                )
            else:
                logger.warning(
                    "[POKE] poke_back circuit opened retry_after_seconds={}",
                    int(self.retry_after_seconds),
                )
            return
        # TEMPORARY_FAILURE / API_FAILURE：不开熔断。
        if previous == _STATE_HALF_OPEN:
            # 探测没有成功证明恢复 → 保守回到 OPEN，不重置 TTL。
            self.state = _STATE_OPEN
            logger.warning(
                "[POKE] poke_back probe failed reason={}, backend stays open",
                status.value,
            )
        # UNKNOWN / AVAILABLE 状态下：保持原状态，等待下一次真实需求再试。

    def reset(self) -> None:
        """重置熔断状态（测试隔离 / 排查用）。"""
        self.state = _STATE_UNKNOWN
        self._retry_at = None


_backend = PokeBackendBreaker()


def reset_backend_state() -> None:
    """重置全局 PacketBackend 熔断（测试隔离 / 排查用）。"""
    _backend.reset()


# ==========================================================================
# 唯一戳回入口
# ==========================================================================


async def send_group_poke(bot: Bot, group_id: int, user_id: int) -> PokeSendResult:
    """在群里戳指定用户一次（绝不抛出）。

    - 配置关闭 → DISABLED；
    - 熔断 OPEN / 探测在途 → 跳过（BACKEND_UNAVAILABLE，DEBUG 日志）；
    - 调用成功 → SUCCESS；
    - 明确 PB 不支持 → BACKEND_UNAVAILABLE + 开熔断；
    - 其它 ActionFailed → API_FAILURE（不开熔断）；
    - timeout / 网络类 → TEMPORARY_FAILURE（不开熔断）。

    日志只记 group_id / user_id / 状态类别 / retcode，绝不输出
    完整 exception / NapCat wording / stacktrace。
    """
    if not _poke_back_enabled():
        return PokeSendResult(PokeSendStatus.DISABLED, False)

    allowed, hold = await _backend.acquire_attempt()
    if not allowed:
        logger.debug("[POKE] poke_back skipped group_id={} user_id={} reason=circuit", group_id, user_id)
        return PokeSendResult(PokeSendStatus.BACKEND_UNAVAILABLE, False)

    try:
        try:
            await bot.call_api(_GROUP_POKE_API, group_id=group_id, user_id=user_id)
        except ActionFailed as exc:
            status, retcode = _classify_action_failed(exc)
            _backend.record_result(status)
            if status is PokeSendStatus.BACKEND_UNAVAILABLE:
                logger.warning(
                    "[POKE] poke_back backend unavailable group_id={} user_id={} "
                    "retcode={} reason=packet_backend_unsupported",
                    group_id,
                    user_id,
                    retcode,
                )
            else:
                logger.warning(
                    "[POKE] poke_back failed group_id={} user_id={} reason=api_failure retcode={}",
                    group_id,
                    user_id,
                    retcode,
                )
            return PokeSendResult(status, False, retcode)
        except Exception:
            _backend.record_result(PokeSendStatus.TEMPORARY_FAILURE)
            logger.warning(
                "[POKE] poke_back failed group_id={} user_id={} reason=temporary_failure",
                group_id,
                user_id,
            )
            return PokeSendResult(PokeSendStatus.TEMPORARY_FAILURE, False)
        else:
            _backend.record_result(PokeSendStatus.SUCCESS)
            return PokeSendResult(PokeSendStatus.SUCCESS, True)
    finally:
        if hold:
            _backend.release_attempt()
