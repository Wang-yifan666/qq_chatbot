"""模型名归一化与能力表（v0.7）：整个项目唯一的模型 alias / capability 来源。

动机（v0.7 Message Understanding Layer）：
- 历史配置里同一个 API 模型出现过多个名字（`deepseek-v4-flash`、
  `deepseek-v4-flash-vision-exp`、`deepseek-flash`），它们指向的是**同一个**
  支持 text + image 的模型；
- 如果各处各自维护一份“这个模型名支不支持视觉”的判断，就会出现
  “API 实际已经在用支持 Vision 的模型，但本地 VISION_CAPABLE_MODELS 判断失败，
  图片在发给模型之前被过滤掉”的隐蔽 bug；
- 因此把所有 alias 归一化与能力判断集中到本模块，其他模块只能调用
  `normalize_model_name()` / `is_vision_capable()`，不得再复制一份集合。

设计要点：
- **归一化只做 alias → canonical 映射**，不改写未知模型名（未知模型原样返回
  小写形式），因此 `deepseek-chat` 仍然是 `deepseek-chat`（text-only）；
- **能力表存 canonical 名**，判断时先归一化再查表；
- 纯函数 + 模块级 frozenset，没有副作用，可被任意模块在 import 期安全使用。
"""

# ===== alias → canonical 映射 =====
# key 必须是 normalize_model_name() 的输出形式（小写、去空白）。
MODEL_ALIASES: dict[str, str] = {
    # DeepSeek：V4.1 Flash 家族的不同历史写法都指向同一个多模态模型
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
    "deepseek-v4-flash-vision": "deepseek-flash",
    "deepseek-v4.1-flash": "deepseek-flash",
    "deepseek-flash": "deepseek-flash",
}

# ===== 视觉能力表 =====
# 只包含“官方验证支持 image input”的 canonical 模型名。
# 未列出的模型（deepseek-chat / 全部 glm-* / 未知名字）一律视为 text-only。
VISION_CAPABLE_MODELS: frozenset[str] = frozenset({"deepseek-flash"})


def normalize_model_name(model: str | None) -> str:
    """把模型名归一化成 canonical 形式（小写；alias 展开；去空白）。

    - None / 空字符串 → 空字符串（调用方决定回落默认模型）；
    - 已知 alias → canonical（`deepseek-v4-flash-vision-exp` → `deepseek-flash`）；
    - 未知模型名 → 原样小写返回（绝不猜测能力，也绝不改写用户配置）。
    """
    if model is None:
        return ""
    key = str(model).strip().lower()
    if not key:
        return ""
    return MODEL_ALIASES.get(key, key)


def model_aliases_of(model: str | None) -> tuple[str, ...]:
    """返回指向同一个 canonical 模型的全部已知名字（debug / 测试用）。"""
    canonical = normalize_model_name(model)
    if not canonical:
        return ()
    names = [name for name, target in MODEL_ALIASES.items() if target == canonical]
    if canonical not in names:
        names.append(canonical)
    return tuple(sorted(names))


def is_vision_capable(model: str | None) -> bool:
    """该模型名（任意 alias 写法）是否被验证支持图片输入。"""
    canonical = normalize_model_name(model)
    return bool(canonical) and canonical in VISION_CAPABLE_MODELS
