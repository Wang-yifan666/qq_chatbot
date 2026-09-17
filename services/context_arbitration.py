"""Context Arbitration（v0.8）：决定“这一轮到底该让哪些历史/记忆进入 Prompt”。

要解决的问题
------------
改造前，只要库里存在个人资料，几乎每一轮都会被塞进 Prompt：

- Mini-RAG 的唯一过滤是 `score > 0`，而“当前说话者自己的资料”被无条件 +10
  （services/memory_retriever.SCORE_CURRENT_USER）→ 当前用户的每条资料恒入选；
- LLM 自动记忆走 `ORDER BY importance DESC LIMIT 10`，没有任何相关性排序；
- 结果是模型每轮都能看到“喜欢 Ubuntu Mono / 正在学 Lean4 / 在学习电路”，
  并倾向于把这些事实塞回一段与它们无关的对话里——也就是用户抱怨的
  “记得 ≠ 必须提”被实现成了“记得 = 必须提”。

设计原则（优先级从高到低）
--------------------------
    current_message > active_topic > relevant_memory > unrelated_history

- 当前消息永远优先，历史只用来理解指代与延续话题；
- 只有当**这一轮确实有可判断的内容**时，才让个人记忆参与；
  纯语气词 / 单字寒暄 / 只有标点的消息，恰恰是最不该被塞记忆的场合；
- 本模块不做语义理解（不写 NLP 分类器）：它只负责“明显不需要记忆”的
  确定性早退，真正的相关性判断交给 LLM（Prompt 里有明确纪律）。

为什么不是阈值打分
------------------
给记忆打分再设阈值，会在“资料只有几条”的真实场景里产生难解释的行为抖动。
这里采用更保守、可解释的规则：**先从消息里取内容词，取不到内容词就不注入**。
误判的代价是“少注入一次背景”，而不是“把不相干的事硬塞进对话”。
"""

from __future__ import annotations

import re

# 语气词 / 寒暄 / 应答：单独出现时不构成“有内容的一轮对话”
FILLER_TOKENS: frozenset[str] = frozenset(
    {
        "在",
        "在吗",
        "在么",
        "在不在",
        "嗯",
        "恩",
        "哦",
        "噢",
        "喔",
        "啊",
        "呀",
        "哈",
        "哈哈",
        "哈哈哈",
        "嘿嘿",
        "呵",
        "额",
        "呃",
        "唉",
        "哎",
        "喂",
        "嗨",
        "hi",
        "hello",
        "hey",
        "ok",
        "okk",
        "okay",
        "好的",
        "好",
        "好吧",
        "行",
        "行吧",
        "可以",
        "收到",
        "懂了",
        "知道了",
        "了解",
        "明白",
        "谢谢",
        "谢了",
        "多谢",
        "感谢",
        "晚安",
        "早安",
        "早上好",
        "晚上好",
        "中午好",
        "你好",
        "大家好",
        "再见",
        "拜拜",
        "无聊",
        "草",
        "？",
        "?",
        "。",
        ".",
        "…",
        "...",
        "!",
        "！",
        "~",
        "～",
    }
)

# 寒暄 / 道别 / 状态通报类整句：整条消息等于其中之一时，同样不该动用个人背景。
# （中文没有空格分词，"我去睡了" 这类整句必须按整条匹配，不能靠切词识别）
FILLER_PHRASES: frozenset[str] = frozenset(
    {
        "在吗",
        "在不在",
        "在么",
        "在嘛",
        "有人在吗",
        "你在吗",
        "在干什么",
        "在干嘛",
        "干嘛呢",
        "你干嘛呢",
        "干啥呢",
        "干嘛",
        "干什么",
        "干啥",
        "怎么了",
        "咋了",
        "什么事",
        "有事吗",
        "在忙吗",
        "忙吗",
        "早",
        "早上好",
        "早安",
        "早啊",
        "晚上好",
        "晚安",
        "午安",
        "中午好",
        "下午好",
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "大家好",
        "来了",
        "我来了",
        "走了",
        "我先走了",
        "我走了",
        "走了啊",
        "我去了",
        "我去睡了",
        "我睡了",
        "睡了",
        "睡觉了",
        "睡觉去了",
        "去睡了",
        "该睡了",
        "我睡了晚安",
        "我去吃饭",
        "我去吃饭了",
        "吃饭去了",
        "去吃饭了",
        "我吃饭去了",
        "我下班了",
        "下班了",
        "上课去了",
        "我上课去了",
        "我出门了",
        "出门了",
        "我回来了",
        "回来了",
        "好累",
        "累了",
        "我好累",
        "困了",
        "我好困",
        "好困",
        "无聊",
        "好无聊",
        "哈哈哈哈",
        "哈哈哈哈哈",
        "笑死",
        "哈哈哈笑死",
        "好的",
        "好的好的",
        "好",
        "行",
        "行吧",
        "可以",
        "收到",
        "明白了",
        "知道了",
        "懂了",
        "谢谢",
        "谢谢你",
        "谢了",
        "多谢",
        "感谢",
        "辛苦了",
        "厉害",
        "牛逼",
        "牛",
        "强",
        "草",
        "卧槽",
        "我错了",
        "对不起",
        "抱歉",
    }
)

# 内容指示符：出现这些字符说明这条消息大概率带着真实信息（英文 / 数字 / 代码 / 专名）
_CONTENT_HINT_RE = re.compile(r"[A-Za-z0-9_+#=<>/\\{}()\[\]]")

# 只由拉丁字母与空白组成的短消息视为寒暄（hi / hey / ok / fine / thanks…）
_LATIN_ONLY_RE = re.compile(r"[A-Za-z\s]+")

# 短消息阈值：**短 ≠ 没内容**（"表还没签字" 只有 5 个字。
# 因此长度只用于最后兜底：没有任何内容指示符、也不属于已知寒暄的超短句，
# 才按“这是语气/寒暄”处理。
SHORT_MESSAGE_CHARS = 8

# 从消息里切分“候选内容词”：按空白与常见标点切开
_TOKEN_SPLIT_RE = re.compile(r"[\s、，,。.；;：:！!？?…~～\-—+*/\\|\[\]【】()（）<>《》\"'“”‘’]+")

# 长度 ≥ 2 的片段才可能是内容词（单字几乎都是语气词，中文里极少有独立信息量）
MIN_CONTENT_TOKEN_CHARS = 2


def content_tokens(text: str) -> list[str]:
    """取出消息中的候选内容词（保序、去重、最小长度过滤）。"""
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_SPLIT_RE.split(text or ""):
        token = raw.strip()
        if len(token) < MIN_CONTENT_TOKEN_CHARS:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tokens.append(token)
    return tokens


def normalize_message(text: str) -> str:
    """去掉标点与空白，得到用于整句匹配的形式。"""
    return _TOKEN_SPLIT_RE.sub("", text or "").strip().lower()


def has_substantive_content(text: str) -> bool:
    """这条消息是否包含“值得动用个人背景”的内容。

    判定顺序（全部确定性、无 NLP 模型）：
    1. 整句命中已知寒暄 / 道别 / 状态通报 → 无内容；
    2. 没有任何内容词（只有标点 / 单字语气词）→ 无内容；
    3. 出现英文 / 数字 / 代码符号 → 有内容（技术问题经常很短）；
    4. 除掉纯语气词后还剩内容词 → 有内容；
    5. 否则只有在“短于 SHORT_MESSAGE_CHARS”时按寒暄处理。
    """
    raw = (text or "").strip()
    if not raw:
        return False

    normalized = normalize_message(raw)
    if not normalized:
        return False
    if normalized in FILLER_PHRASES:
        return False

    tokens = content_tokens(raw)
    if not tokens:
        return False

    # 短英文寒暄（hi / hey / ok / fine / thanks…）单独处理：
    # 这类消息只由拉丁字母组成，不该因为“含英文字符”就被当成技术内容。
    if len(_LATIN_ONLY_RE.sub("", raw)) == 0 and len(normalized) < SHORT_MESSAGE_CHARS:
        return False

    # 内容指示符：长度 ≥2 的片段里出现英文 / 数字 / 代码符号才算数，
    # 避免把 “hi” 里的 i 误判成技术内容。
    if _CONTENT_HINT_RE.search(raw) and any(len(token) >= 2 for token in tokens):
        return True

    meaningful = [token for token in tokens if token.lower() not in FILLER_TOKENS]
    if meaningful:
        # 全都是语气词时 meaningful 为空；只要还剩内容词，就说明这一轮有实质信息
        if len(meaningful) > 1 or len(meaningful[0]) >= MIN_CONTENT_TOKEN_CHARS:
            return True

    return len(normalized) >= SHORT_MESSAGE_CHARS


def should_inject_personal_memory(question: str) -> bool:
    """这一轮是否应该注入个人资料（Mini-RAG / 长期记忆）。

    返回 False 的两种情况：
    - 消息里没有任何内容词（只有标点 / 表情 / 单字）；
    - 消息只是寒暄、应答或道别（“在吗”“嗯”“谢谢”“我去睡了”）。

    注意：这不是“相关性判断”，而是“相关性判断的前提”。
    真正的相关性仍然由 LLM 依据 Prompt 里的纪律完成——本函数只保证
    不会在一句“在吗”后面塞进一份个人档案。
    """
    return has_substantive_content(question)


# 进入 SYSTEM 的纪律条文（与 Prompt 层配合，程序层负责前置过滤）
CONTEXT_ARBITRATION_RULES = """【上下文仲裁纪律（优先级从高到低）】
current_message > active_topic > relevant_memory > unrelated_history

- **当前消息拥有最高优先级**：先回答/回应此刻这句话，再考虑要不要用到历史；
- active_topic 是“这几轮正在聊的事”；历史只有与当前话题有明显语义关系时才可以引用；
- relevant_memory 是背景资料，只在当前话题真的相关时使用：
  资料存在 **不代表** 你应该提起它；
- unrelated_history 一律忽略：不要把已经换掉的旧话题拉回来，
  不要在一个新问题后面追问上一个问题的进度；
- **记得 ≠ 必须提**：你知道某人的偏好、项目、技能，只是让你在相关时刻更自然，
  不是每轮都要汇报“我还记得你在做 X”；
- 一轮对话只服务于此刻的社交目的：答完就停，不要为了显得记得很多而补充旧信息。"""
