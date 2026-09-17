"""感知层（Perception Layer，v0.7）：把 QQ 输入解释成结构化多模态上下文。

本包只负责“发生了什么”，绝不负责“夜子会怎么说”：

    QQ Event
      ↓
    Message Resolver（message_resolver.py）        ← 唯一事件解释入口
      ↓
    NormalizedMessage / ReplyContext / ForwardNode  ← content.py 数据模型
      ↓
    File Reader（file_reader.py）                   ← 只负责“文件里写了什么”
      ↓
    Multimodal Content Builder（multimodal_builder.py）← 结构化 → LLM blocks
      ↓
    Conversation Context（用户/历史/记忆/人格仍由原 pipeline 负责）

边界（v0.7 开发原则）：
- 本包不导入 prompt_builder / persona_rag / memory_* / relationship_*，
  不写人格文案、不做模板化回复、不调用 LLM；
- 图片、文件、合并转发、被回复消息一律属于 **UNTRUSTED USER DATA**，
  最终只能以“数据”身份进入 Prompt；
- 所有外部内容（文件正文、转发正文、图片描述）都必须先过
  services/perception/limits.py 的统一预算，绝不无上限塞进 Prompt。
"""
