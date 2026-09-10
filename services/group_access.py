"""群聊访问白名单（v0.3.1）：fail-closed 静态群白名单。

核心原则：只有配置在 .env 的 ALLOWED_GROUP_IDS 里的 QQ 群，机器人才会工作；
没有进入白名单的群必须被完全忽略（不回复、不调用 AI、不写入任何数据库）。

环境变量解析、合法性检查、白名单判断全部集中在本模块：

    ALLOWED_GROUP_IDS=123,456     只有群 123 与 456 可用
    ALLOWED_GROUP_IDS=123, 456    逗号两侧空格允许（每项会 strip）
    ALLOWED_GROUP_IDS=*           允许所有群（显式恢复旧版本行为）
    ALLOWED_GROUP_IDS= 或未配置   不允许任何群（fail-closed，默认沉默）
    含非法项（如 123,abc）        抛 ValueError → bot.py 启动报错退出

白名单修改后重启 Bot 生效。本模块不依赖数据库、不做动态管理。
"""

import os

# 允许所有群（ALLOWED_GROUP_IDS 为 "*" 时置 True）
ALLOW_ALL_GROUPS: bool

# 允许的群号集合（ALLOW_ALL_GROUPS 为 True 时为空集，不再逐个比对）
ALLOWED_GROUP_IDS: frozenset[int]


def parse_allowed_group_ids(raw: str | None) -> tuple[frozenset[int], bool]:
    """解析 ALLOWED_GROUP_IDS 原始字符串，返回 (允许的群号集合, 是否允许所有群)。

    - None / 空 / 全空白 → (空集合, False)：不允许任何群；
    - 去掉首尾空白后为 "*" → (空集合, True)：允许所有群；
    - 其余按英文逗号切分：空项忽略，非空项必须是纯数字的正整数群号，
      否则抛 ValueError（错误信息不含原始配置值）。
    """
    text = (raw or "").strip()
    if not text:
        return frozenset(), False
    if text == "*":
        return frozenset(), True

    allowed: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        # isdigit() 拒绝负数 / 小数 / 混杂字符；<=0 拒绝 "0" 这类无效群号
        if not part.isdigit() or int(part) <= 0:
            raise ValueError(
                "ALLOWED_GROUP_IDS 配置非法：只能包含英文逗号分隔的正整数 QQ 群号，"
                "或单独一个 *（表示允许所有群），请检查 .env"
            )
        allowed.add(int(part))
    return frozenset(allowed), False


def _load_config() -> tuple[frozenset[int], bool]:
    """模块导入时读取一次环境变量（bot.py 在 load_dotenv 之后导入本模块）。"""
    return parse_allowed_group_ids(os.getenv("ALLOWED_GROUP_IDS"))


ALLOWED_GROUP_IDS, ALLOW_ALL_GROUPS = _load_config()


def is_group_allowed(group_id: int) -> bool:
    """判断某个 QQ 群是否在白名单内（fail-closed：默认拒绝一切）。"""
    return ALLOW_ALL_GROUPS or group_id in ALLOWED_GROUP_IDS
