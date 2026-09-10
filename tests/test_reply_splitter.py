"""services/reply_splitter.py：回复拆分规则测试（v0.3.1）。"""

from services.reply_splitter import split_reply


class TestSplitReply:
    def test_short_text_not_split(self):
        assert split_reply("你好") == ["你好"]

    def test_multi_paragraph_split(self):
        parts = split_reply("第一段。\n\n第二段。\n\n第三段。")
        assert parts == ["第一段。", "第二段。", "第三段。"]

    def test_consecutive_blank_lines_are_one_separator(self):
        assert split_reply("a\n\n\n\nb") == ["a", "b"]

    def test_code_block_never_split(self):
        text = "说明文字\n\n```python\nline1\n\nline2\n```\n\n结尾"
        parts = split_reply(text)
        assert parts == ["说明文字", "```python\nline1\n\nline2\n```", "结尾"]

    def test_max_chars_chunking(self):
        text = "第一句很长。" * 20
        parts = split_reply(text, max_parts=10, max_chars=50)
        assert len(parts) > 1
        assert all(len(p) <= 50 for p in parts)

    def test_max_parts_merges_rest_into_last(self):
        text = "\n\n".join(f"第{i}段" for i in range(10))
        parts = split_reply(text, max_parts=3, max_chars=1000)
        assert len(parts) == 3
        assert parts[0] == "第0段"
        assert parts[1] == "第1段"
        assert parts[2] == "\n\n".join(f"第{i}段" for i in range(2, 10))

    def test_max_parts_one_keeps_whole_text(self):
        parts = split_reply("一段。", max_parts=1, max_chars=1000)
        assert len(parts) == 1

    def test_empty_input_returns_one_empty_part(self):
        assert split_reply("") == [""]
        assert split_reply("   \n\n  ") == [""]

    def test_hard_cut_for_single_long_sentence(self):
        text = "x" * 120
        parts = split_reply(text, max_parts=5, max_chars=50)
        assert all(0 < len(p) <= 50 for p in parts)
        assert "".join(parts) == text

    def test_leading_trailing_blank_lines_ignored(self):
        assert split_reply("\n\n你好\n\n") == ["你好"]

    def test_extremely_large_input_still_bounded(self):
        # 100 段 × 50 字：max_parts=20 时必须只返回 20 条（防刷屏）
        text = "\n\n".join("段" * 50 for _ in range(100))
        parts = split_reply(text, max_parts=20, max_chars=4000)
        assert len(parts) == 20
