"""services/vision.py：图片提取 / 限制 / multimodal attach / 占位符测试（v0.5）。"""

from types import SimpleNamespace

import pytest

import services.vision as vision
from services.vision import ImageExtraction
from services.vision import VisionImage
from services.vision import attach_images_to_last_user_message
from services.vision import build_context_text
from services.vision import build_image_blocks
from services.vision import extract_images
from services.vision import messages_have_images


def _seg(seg_type: str, data: dict | None = None):
    return SimpleNamespace(type=seg_type, data=data or {})


def _event(segments: list):
    return SimpleNamespace(get_message=lambda: segments)


def _img(url: str, detail: str = "auto") -> VisionImage:
    return VisionImage(url=url, detail=detail)


class TestExtractImages:
    def test_single_image_with_url(self):
        result = extract_images(_event([_seg("text", {"text": "hi"}), _seg("image", {"url": "http://x/img.jpg"})]))
        assert result.total == 1
        assert result.accepted == 1
        assert result.rejected == 0
        assert result.images == [_img("http://x/img.jpg")]

    def test_segment_without_url_rejected(self):
        result = extract_images(_event([_seg("image", {"file": "abc.img"})]))
        assert result.total == 1
        assert result.accepted == 0
        assert result.rejected == 1
        assert result.images == []

    def test_multi_image_order_preserved(self):
        result = extract_images(
            _event(
                [
                    _seg("image", {"url": "http://x/1.jpg"}),
                    _seg("text", {"text": "看看"}),
                    _seg("image", {"url": "http://x/2.jpg"}),
                ]
            )
        )
        assert [img.url for img in result.images] == ["http://x/1.jpg", "http://x/2.jpg"]

    def test_over_limit_truncated(self, monkeypatch):
        monkeypatch.setattr(vision, "VISION_MAX_IMAGES", 2)
        result = extract_images(
            _event([_seg("image", {"url": f"http://x/{i}.jpg"}) for i in range(5)])
        )
        assert result.total == 5
        assert result.accepted == 2
        assert result.rejected == 3
        assert [img.url for img in result.images] == ["http://x/0.jpg", "http://x/1.jpg"]

    def test_file_size_too_big_rejected(self, monkeypatch):
        monkeypatch.setattr(vision, "VISION_MAX_IMAGE_BYTES", 1024)
        result = extract_images(
            _event(
                [
                    _seg("image", {"url": "http://x/big.jpg", "file_size": "4096"}),
                    _seg("image", {"url": "http://x/ok.jpg", "file_size": "512"}),
                    _seg("image", {"url": "http://x/nosize.jpg"}),
                ]
            )
        )
        assert result.total == 3
        assert result.accepted == 2
        assert result.rejected == 1
        assert [img.url for img in result.images] == ["http://x/ok.jpg", "http://x/nosize.jpg"]

    def test_invalid_file_size_ignored(self):
        result = extract_images(_event([_seg("image", {"url": "http://x/a.jpg", "file_size": "abc"})]))
        assert result.accepted == 1

    def test_detail_config_applied(self, monkeypatch):
        monkeypatch.setattr(vision, "VISION_DETAIL", "high")
        result = extract_images(_event([_seg("image", {"url": "http://x/a.jpg"})]))
        assert result.images[0].detail == "high"


class TestAttachImages:
    def _messages(self):
        return [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "上下文 DATA"},
            {"role": "user", "content": "当前提问者 user_id=1\n当前消息：\n你好"},
        ]

    def test_last_user_message_becomes_multimodal(self):
        messages = self._messages()
        attached = attach_images_to_last_user_message(
            messages, [_img("http://x/1.jpg"), _img("http://x/2.jpg")]
        )
        last = attached[-1]
        assert isinstance(last["content"], list)
        assert last["content"][0] == {"type": "text", "text": "当前提问者 user_id=1\n当前消息：\n你好"}
        assert last["content"][1]["type"] == "image_url"
        assert last["content"][1]["image_url"] == {"url": "http://x/1.jpg", "detail": "auto"}
        assert last["content"][2]["image_url"] == {"url": "http://x/2.jpg", "detail": "auto"}

    def test_images_only_on_user_role(self):
        messages = self._messages()
        attached = attach_images_to_last_user_message(messages, [_img("http://x/1.jpg")])
        for message in attached[:-1]:
            assert isinstance(message["content"], str), "system / 其它 user 消息保持纯文本"
        # system 绝不含图片
        assert not messages_have_images([attached[0]])

    def test_original_messages_not_mutated(self):
        messages = self._messages()
        attach_images_to_last_user_message(messages, [_img("http://x/1.jpg")])
        assert isinstance(messages[-1]["content"], str)
        assert messages[0]["content"] == "SYSTEM"

    def test_empty_images_returns_unchanged(self):
        messages = self._messages()
        attached = attach_images_to_last_user_message(messages, [])
        assert attached == messages

    def test_messages_have_images_detection(self):
        plain = self._messages()
        assert not messages_have_images(plain)
        attached = attach_images_to_last_user_message(plain, [_img("http://x/1.jpg")])
        assert messages_have_images(attached)

    def test_build_image_blocks_shape(self):
        blocks = build_image_blocks([_img("http://x/1.jpg", "low")])
        assert blocks == [{"type": "image_url", "image_url": {"url": "http://x/1.jpg", "detail": "low"}}]


class TestContextPlaceholder:
    def test_text_only_unchanged(self):
        assert build_context_text("你好", 0) == "你好"

    def test_text_plus_images(self):
        assert build_context_text("这个报错怎么看", 2) == "这个报错怎么看\n[附带 2 张图片]"

    def test_pure_image(self):
        assert build_context_text("", 1) == "[发送了 1 张图片]"

    def test_no_url_leak(self):
        text = build_context_text("", 3)
        assert "http" not in text
        assert "base64" not in text


class TestConfigDefaults:
    def test_defaults(self):
        assert vision.VISION_ENABLED is True
        assert vision.VISION_MAX_IMAGES == 4
        assert vision.VISION_DETAIL == "auto"
        assert vision.VISION_MAX_IMAGE_BYTES == 10 * 1024 * 1024

    def test_invalid_detail_env_falls_back(self, monkeypatch):
        import importlib

        monkeypatch.setenv("VISION_DETAIL", "bogus")
        importlib.reload(vision)
        try:
            assert vision.VISION_DETAIL == "auto"
        finally:
            monkeypatch.undo()
            importlib.reload(vision)

    def test_invalid_max_images_env_falls_back(self, monkeypatch):
        import importlib

        monkeypatch.setenv("VISION_MAX_IMAGES", "999")
        importlib.reload(vision)
        try:
            assert vision.VISION_MAX_IMAGES == 4
        finally:
            monkeypatch.undo()
            importlib.reload(vision)
