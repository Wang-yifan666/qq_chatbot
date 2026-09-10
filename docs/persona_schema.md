# DialogueUnit 标注 Schema（v3）

本文件描述 Persona RAG 语料文件（`data/persona_processed/` 下你自己的标注语料，
已 gitignore，版权数据禁止提交）中每行 DialogueUnit 的字段含义。仓库内只保留本
schema 文档与自造示例（`docs/persona_examples.jsonl`），不含任何原作台词。

## 字段说明

### 基础字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 语料单元稳定 ID（如 `example_acd82f5ef46f`） |
| `speaker` | string | 说话人；构建脚本只索引 `speaker == 角色名` 且 text 非空的行（speaker 过滤值在 `scripts/build_persona_rag.py` / `services/persona_rag.py` 中写死，换用你自己的语料时改成你语料中的 speaker 值） |
| `text` | string | 角色台词原文（版权内容，禁止提交） |
| `context` | list[dict] | 台词前文，每项 `{speaker, text}`；构建索引时只取最后 1~3 条并截断 |
| `source` | dict | `{file, line, encoding}` 来源信息；**属于 metadata，绝不进 embedding** |
| `scene` | string | 场景标签（当前语料多为 `unknown`） |

### 人际状态

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `relation_stage` | string | 语料中的心理距离：`hostile / guarded / neutral / accustomed / trusting / vulnerable`（与 QQ 侧的 `stranger / acquaintance / familiar / close` 是两套体系，运行时用权重映射） |
| `intimacy_level` | int | 0~4 亲密等级；普通模式（close≠恋爱）下 `>=3` 一律排除（双保险） |
| `primary_addressee` | string | 主要对话对象 |

### 语义标签

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `emotion` | list[str] | 情绪标签 |
| `topics` | list[str] | 话题标签（`book / magic_book / grief / daily_chat / other` 等） |
| `response_intent` | string | 回应意图（如 `state / avoid / refuse / comfort`） |
| `speech_style` | list[str] | 语言风格标签 |

### 检索质量字段（上一阶段标注脚本的派生结果）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `rag_quality` | float | 0~1 检索质量评分；运行时 `>= PERSONA_RAG_MIN_QUALITY(0.65)` 才可用 |
| `spoiler_level` | int | 0~3 剧透等级；默认 `PERSONA_RAG_MAX_SPOILER_LEVEL=0` 硬过滤 |
| `plot_specific` | bool | 剧情专属标记；**不硬过滤**，rerank 小幅降权（×0.90） |
| `romance_specific` | bool | 恋爱专属标记；普通模式硬过滤（`true` 一律排除） |
| `persona_note` | string | 人格反应摘要（“角色在类似情况下会如何反应”）——retrieval_text 的第一个字段，权重最高 |
| `needs_review` | bool | 是否待人工复核（仅标注流程用，运行时不过滤） |
| `rag_candidate` | bool | **上一阶段标注脚本的派生结果，不是永久真值**：运行时绝不作为过滤条件，只用于 debug 日志对比 |
| `source_route` | string | 来源路线（如 `main`）；**不参与硬过滤** |
| `label_meta` | dict | 标注元信息（模型、schema 版本） |

## 运行时如何决定“能否检索”

程序运行时根据以下字段动态决定，与 `rag_candidate` 无关：

```text
rag_quality       >= PERSONA_RAG_MIN_QUALITY (0.65)
spoiler_level     <= PERSONA_RAG_MAX_SPOILER_LEVEL (0)
romance_specific  == false
intimacy_level    <= MAX_INTIMACY[relationship]  (stranger:0 acquaintance:1 familiar:2 close:2)
speaker           == 角色名 且 text 非空（过滤值写死在构建脚本与检索模块中，换语料时同步修改）
```
