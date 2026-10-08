"""生图命令测试（v0.10）：命令解析 / 权限白名单 / 冷却与限额 / 存档 / 响应兼容。

全部是纯函数与 mock：
- 不连 QQ、不调真实生图 API、不调真实 LLM；
- 覆盖两条容易出错的边界：`/painter` 不能被当成 `/paint`、
  以及生图接口「返回 b64」与「返回 url」两种形态都要能处理。
"""

import base64
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import plugins.paint as paint_mod
import services.image_gen as image_gen

# 一个最小 PNG 的文件头（只需前 8 字节就能判断类型）
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


# ======================================================================
# 命令解析
# ======================================================================


class TestExtractPrompt:
    def test_basic(self):
        assert paint_mod.extract_prompt("/paint 一只戴着毛线帽的橘猫") == "一只戴着毛线帽的橘猫"

    def test_case_insensitive(self):
        assert paint_mod.extract_prompt("/PAINT 猫") == "猫"
        assert paint_mod.extract_prompt("/Paint 猫") == "猫"

    def test_fullwidth_slash(self):
        """中文输入法很容易打出全角斜杠，必须一起支持。"""
        assert paint_mod.extract_prompt("／paint 猫") == "猫"

    def test_extra_spaces_stripped(self):
        assert paint_mod.extract_prompt("   /paint    猫   ") == "猫"

    def test_newline_after_prefix(self):
        """真实踩过的坑：先打 /paint 再回车写描述，换行必须算空白。

        群里第一条 /paint 就是这么发的，因为换行不在白名单里，
        规则判定"不是命令"，消息被当成普通聊天丢给了 AI。
        """
        assert paint_mod.extract_prompt("/paint\n画一张猫的图片") == "画一张猫的图片"
        assert paint_mod.extract_prompt("/paint\n\n猫") == "猫"
        assert paint_mod.extract_prompt("/paint\r\n猫") == "猫"

    @pytest.mark.parametrize("sep", [" ", "\t", "\n", "\r\n", "\u3000", "\v", "\f"])
    def test_any_whitespace_separator(self, sep):
        assert paint_mod.extract_prompt(f"/paint{sep}猫") == "猫"

    def test_at_prefix_already_stripped(self):
        """get_plaintext() 会去掉 @机器人，所以传进来就是纯命令文本。"""
        assert paint_mod.extract_prompt("/paint 猫") == "猫"

    def test_missing_prompt_returns_empty(self):
        assert paint_mod.extract_prompt("/paint") == ""
        assert paint_mod.extract_prompt("/paint   ") == ""

    def test_not_a_command(self):
        assert paint_mod.extract_prompt("普通消息") is None
        assert paint_mod.extract_prompt("") is None

    def test_similar_prefix_is_not_a_command(self):
        """`/painter`、`/paints` 不能被误判成 /paint。"""
        assert paint_mod.extract_prompt("/painter 猫") is None
        assert paint_mod.extract_prompt("/paints 猫") is None
        assert paint_mod.extract_prompt("/paint2 猫") is None

    def test_too_long_prompt_is_truncated(self):
        long_prompt = "猫" * (paint_mod.MAX_PROMPT_CHARS + 500)
        result = paint_mod.extract_prompt("/paint " + long_prompt)
        assert result is not None
        assert len(result) == paint_mod.MAX_PROMPT_CHARS


# ======================================================================
# 管理员白名单
# ======================================================================


class TestAdminWhitelist:
    def test_parse_skips_invalid(self):
        assert paint_mod._parse_qq_set("123, ,abc,456") == {123, 456}

    def test_empty(self):
        assert paint_mod._parse_qq_set("") == set()

    def test_paint_admins_takes_priority(self):
        env = {"PAINT_ADMIN_QQ": "111", "DEBUG_ADMIN_QQ": "222"}
        assert paint_mod._load_admin_qq(env) == {111}

    def test_falls_back_to_debug_admins(self):
        """没配 PAINT_ADMIN_QQ 时沿用 \\debug 的白名单。"""
        env = {"DEBUG_ADMIN_QQ": "222,333"}
        assert paint_mod._load_admin_qq(env) == {222, 333}

    def test_both_empty_means_disabled(self):
        assert paint_mod._load_admin_qq({}) == set()


# ======================================================================
# 冷却与每日上限
# ======================================================================


class TestQuota:
    def _state(self):
        return paint_mod.QuotaState(last_at={})

    def test_first_use_allowed(self):
        allowed, reason, _ = paint_mod.check_quota(self._state(), 1, now=1000.0, cooldown=60, daily_limit=5)
        assert (allowed, reason) == (True, "ok")

    def test_cooldown_blocks_and_reports_remaining(self):
        state = self._state()
        paint_mod.record_use(state, 1, now=1000.0)
        allowed, reason, remain = paint_mod.check_quota(state, 1, now=1030.0, cooldown=60, daily_limit=5)
        assert (allowed, reason) == (False, "cooldown")
        assert remain == 30

    def test_cooldown_expires(self):
        state = self._state()
        paint_mod.record_use(state, 1, now=1000.0)
        allowed, reason, _ = paint_mod.check_quota(state, 1, now=1061.0, cooldown=60, daily_limit=5)
        assert (allowed, reason) == (True, "ok")

    def test_cooldown_is_per_user(self):
        state = self._state()
        paint_mod.record_use(state, 1, now=1000.0)
        allowed, reason, _ = paint_mod.check_quota(state, 2, now=1005.0, cooldown=60, daily_limit=5)
        assert (allowed, reason) == (True, "ok")

    def test_daily_limit(self):
        state = self._state()
        for i in range(3):
            paint_mod.record_use(state, 1, now=1000.0 + i)
        allowed, reason, limit = paint_mod.check_quota(state, 1, now=2000.0, cooldown=0, daily_limit=3)
        assert (allowed, reason, limit) == (False, "daily_limit", 3)

    def test_daily_limit_resets_next_day(self):
        state = self._state()
        day1 = time.mktime(time.strptime("2026-10-08 12:00:00", "%Y-%m-%d %H:%M:%S"))
        day2 = day1 + 86400
        for i in range(3):
            paint_mod.record_use(state, 1, now=day1 + i)
        allowed, reason, _ = paint_mod.check_quota(state, 1, now=day2, cooldown=0, daily_limit=3)
        assert (allowed, reason) == (True, "ok")

    def test_zero_limits_mean_unlimited(self):
        state = self._state()
        paint_mod.record_use(state, 1, now=1000.0)
        allowed, reason, _ = paint_mod.check_quota(state, 1, now=1000.0, cooldown=0, daily_limit=0)
        assert (allowed, reason) == (True, "ok")


# ======================================================================
# 存档
# ======================================================================


class TestArchive:
    def test_filename_suffix_by_mime(self):
        now = time.mktime(time.strptime("2026-10-08 13:14:15", "%Y-%m-%d %H:%M:%S"))
        assert paint_mod.build_image_filename(42, now, "image/png") == "20261008-131415-42.png"
        assert paint_mod.build_image_filename(42, now, "image/jpeg") == "20261008-131415-42.jpg"
        assert paint_mod.build_image_filename(42, now, "image/webp") == "20261008-131415-42.webp"
        assert paint_mod.build_image_filename(42, now, "image/unknown") == "20261008-131415-42.png"

    def test_save_writes_image_and_meta(self, tmp_path):
        image = SimpleNamespace(data=PNG_BYTES, mime="image/png")
        path = paint_mod.save_archive(image, "prompt: 猫\n", directory=tmp_path, now=1000.0, user_id=7)
        assert path is not None
        assert path.read_bytes() == PNG_BYTES
        assert path.with_suffix(".txt").read_text(encoding="utf-8") == "prompt: 猫\n"

    def test_save_failure_is_not_fatal(self, tmp_path):
        """存档失败不能让整条命令崩掉 —— 图还是要发出去。"""
        blocker = tmp_path / "blocked"
        blocker.write_text("我是一个文件，不是目录", encoding="utf-8")
        image = SimpleNamespace(data=PNG_BYTES, mime="image/png")
        assert paint_mod.save_archive(image, "x", directory=blocker, now=1000.0, user_id=1) is None


# ======================================================================
# 生图接口配置
# ======================================================================


class TestConfig:
    def test_defaults(self):
        cfg = image_gen.get_config({})
        assert cfg.model == image_gen.DEFAULT_MODEL
        assert cfg.base_url == image_gen.DEFAULT_BASE_URL
        assert cfg.size == image_gen.DEFAULT_SIZE
        assert cfg.enabled is True
        assert cfg.configured is False, "没配 key 就不该认为可用"

    def test_configured_requires_key(self):
        assert image_gen.get_config({"PAINT_API_KEY": "sk-x"}).configured is True
        assert image_gen.get_config({"PAINT_API_KEY": "  "}).configured is False

    def test_disabled_switch(self):
        assert image_gen.get_config({"PAINT_ENABLED": "false", "PAINT_API_KEY": "k"}).configured is False
        assert image_gen.get_config({"PAINT_ENABLED": "0", "PAINT_API_KEY": "k"}).configured is False

    def test_invalid_enum_values_fall_back(self):
        cfg = image_gen.get_config(
            {"PAINT_QUALITY": "ultra", "PAINT_OUTPUT_FORMAT": "webp", "PAINT_RESPONSE_FORMAT": "xml"}
        )
        assert cfg.quality == "auto"
        assert cfg.output_format == "png"
        assert cfg.response_format == "b64_json"

    def test_invalid_numbers_fall_back(self):
        cfg = image_gen.get_config({"PAINT_TIMEOUT_SECONDS": "abc", "PAINT_MAX_BYTES": "-5"})
        assert cfg.timeout == image_gen.DEFAULT_TIMEOUT
        assert cfg.max_bytes == image_gen.DEFAULT_MAX_BYTES

    def test_custom_gateway(self):
        """其它中转：只改 base_url / model 就能用。"""
        cfg = image_gen.get_config(
            {"PAINT_BASE_URL": "https://relay.example.com/v1", "PAINT_MODEL": "gpt-image-1"}
        )
        assert cfg.base_url == "https://relay.example.com/v1"
        assert cfg.model == "gpt-image-1"


# ======================================================================
# 尺寸校验（换API 文档给的硬性规则）
# ======================================================================


class TestValidateSize:
    @pytest.mark.parametrize(
        "size",
        ["auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048", "3840x2160", "1024X1024"],
    )
    def test_accepted(self, size):
        value, reason = image_gen.validate_size(size)
        assert reason is None, f"{size} 应该被接受，却报 {reason}"
        assert value == size.lower()

    @pytest.mark.parametrize(
        "size,keyword",
        [
            ("9999x9999", "最大边长"),      # 超过 3840
            ("1000x1000", "16"),            # 不是 16 的倍数
            ("3840x1024", "长边/短边"),      # 3.75:1，超过 3:1（其它规则都满足）
            ("256x256", "总像素数"),         # 像素太少
            ("abc", "无法解析"),
            ("", "空的尺寸"),
        ],
    )
    def test_rejected_with_reason(self, size, keyword):
        value, reason = image_gen.validate_size(size)
        assert reason is not None and keyword in reason
        assert value == image_gen.DEFAULT_SIZE, "被拒绝时必须回落到默认尺寸"

    def test_config_uses_validated_size(self):
        assert image_gen.get_config({"PAINT_SIZE": "3840x2160"}).size == "3840x2160"
        assert image_gen.get_config({"PAINT_SIZE": "9999x9999"}).size == image_gen.DEFAULT_SIZE


class TestRequestBody:
    def _cfg(self, **overrides):
        return image_gen.get_config({"PAINT_API_KEY": "k", **overrides})

    def test_matches_documented_shape(self):
        """请求体要和文档示例一致，且绝不带上「不支持 / 不建议」的参数。"""
        body = image_gen.build_request_body("一只橘猫", self._cfg())
        assert body["model"] == "gpt-image-2"
        assert body["prompt"] == "一只橘猫"
        assert body["n"] == 1, "文档：n 仅支持 1"
        assert body["size"] == "1024x1024"
        assert body["quality"] == "auto"
        assert body["output_format"] == "png"
        assert body["response_format"] == "b64_json"
        for forbidden in ("stream", "partial_images", "style"):
            assert forbidden not in body, f"{forbidden} 不能传（文档明确不支持/不建议）"

    def test_moderation_only_when_configured(self):
        assert "moderation" not in image_gen.build_request_body("x", self._cfg())
        body = image_gen.build_request_body("x", self._cfg(PAINT_MODERATION="low"))
        assert body["moderation"] == "low"


class TestSdkCall:
    @pytest.mark.asyncio
    async def test_vendor_params_go_through_extra_body(self, monkeypatch):
        """SDK 不认识的字段必须走 extra_body，否则会直接 TypeError。"""
        captured = {}

        class FakeImages:
            async def generate(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    data=[SimpleNamespace(b64_json=PNG_B64, url=None, revised_prompt=None)]
                )

        monkeypatch.setattr(
            image_gen, "_get_client", lambda cfg: SimpleNamespace(images=FakeImages())
        )
        cfg = image_gen.get_config({"PAINT_API_KEY": "k"})
        image = await image_gen.generate_image("一只猫", cfg)

        assert image is not None and image.data == PNG_BYTES
        assert captured["model"] == "gpt-image-2"
        assert captured["prompt"] == "一只猫"
        assert captured["n"] == 1
        extra = captured["extra_body"]
        assert extra["size"] == "1024x1024"
        assert extra["output_format"] == "png"
        assert extra["response_format"] == "b64_json"
        # 这几个字段绝不能作为标准参数出现（SDK 没有 output_format 这个参数）
        for key in ("output_format", "quality", "size", "response_format", "moderation"):
            assert key not in captured, f"{key} 不该作为标准参数传给 SDK"

    @pytest.mark.asyncio
    async def test_request_failure_returns_none(self, monkeypatch):
        class BoomImages:
            async def generate(self, **kwargs):
                raise RuntimeError("network down")

        monkeypatch.setattr(
            image_gen, "_get_client", lambda cfg: SimpleNamespace(images=BoomImages())
        )
        cfg = image_gen.get_config({"PAINT_API_KEY": "k"})
        assert await image_gen.generate_image("x", cfg) is None

    @pytest.mark.asyncio
    async def test_not_configured_does_not_call_api(self, monkeypatch):
        def explode(cfg):  # pragma: no cover - 不该被调用
            raise AssertionError("没配 key 时不应该调用 SDK")

        monkeypatch.setattr(image_gen, "_get_client", explode)
        assert await image_gen.generate_image("x", image_gen.get_config({})) is None


# ======================================================================
# 返回形态兼容（b64_json / url）
# ======================================================================


class TestExtractImageBytes:
    def test_b64_json(self):
        item = SimpleNamespace(b64_json=PNG_B64)
        data, mime = image_gen.extract_image_bytes(item)
        assert data == PNG_BYTES
        assert mime == "image/png"

    def test_url_form_returns_marker(self):
        """url 形态先返回空字节 + 标记，由上层去下载。"""
        item = SimpleNamespace(b64_json=None, url="https://example.com/a.png")
        assert image_gen.extract_image_bytes(item) == (b"", "url")

    def test_neither_present(self):
        assert image_gen.extract_image_bytes(SimpleNamespace(b64_json=None, url=None)) is None

    def test_garbage_b64(self):
        assert image_gen.extract_image_bytes(SimpleNamespace(b64_json="!!!not-base64!!!")) is None

    def test_oversize_rejected(self):
        big = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 5000).decode()
        assert image_gen.extract_image_bytes(SimpleNamespace(b64_json=big), max_bytes=100) is None

    def test_mime_detection(self):
        assert image_gen._guess_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 8) == "image/jpeg"
        assert image_gen._guess_mime(b"GIF89a" + b"\x00" * 8) == "image/gif"
        assert image_gen._guess_mime(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"
        assert image_gen._guess_mime(b"anything else") == "image/png"


# ======================================================================
# 端到端（全 mock）：不触网、不调 LLM
# ======================================================================


class FakeMatcher:
    """记录 finish/send 收到的消息，避免真正发送 QQ 消息。"""

    def __init__(self):
        self.sent: list = []
        self.finished: list = []

    async def send(self, msg):
        self.sent.append(msg)

    async def finish(self, msg):
        self.finished.append(msg)
        raise _Finished()


class _Finished(Exception):
    pass


def _flatten(msg) -> str:
    if isinstance(msg, str):
        return msg
    parts = []
    for seg in msg:
        if seg.type == "text":
            parts.append(seg.data.get("text", ""))
        elif seg.type == "image":
            parts.append(f"[image:{len(seg.data.get('file', b''))}]")
    return "".join(parts)


def _fake_event(group_id: int = 111, user_id: int = 1, text: str = "/paint 一只猫"):
    """构造一个够用的假事件（sender / self_id / message 都是 handler 会用到的）。"""
    return SimpleNamespace(
        group_id=group_id,
        user_id=user_id,
        self_id=1000000001,
        message=[],
        reply=None,
        get_plaintext=lambda: text,
        sender=SimpleNamespace(card="", nickname="测试用户", user_id=user_id),
    )


def _stub_rewrite(monkeypatch):
    """把提示词改写换成恒等函数。

    e2e 测试关心的是管线（谁被调用、存了什么、发了什么），不是改写本身；
    而改写会真的调 LLM —— 全量跑时别的测试可能已经装好了可用的 LLM 桩，
    结果改写生效、断言里的原话被替换掉，测试就随执行顺序时好时坏。
    改写本身由 TestRewritePrompt / TestCleanImagePrompt 单独覆盖。
    """

    async def identity(prompt: str, editing: bool = False) -> str:
        return prompt

    monkeypatch.setattr(paint_mod, "rewrite_prompt", identity)


def _record_context(monkeypatch) -> list:
    """把 add_message 换成记录器，并返回记录列表。"""
    calls: list = []

    async def fake_add_message(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(paint_mod, "add_message", fake_add_message)
    return calls


# ======================================================================
# 图生图：参考图的选择与上传
# ======================================================================


def _img(url: str = "https://cdn.example.com/a.jpg"):
    """构造一个图片段（只要 type/data.url 够用，不必是真的 MessageSegment）。"""
    return SimpleNamespace(type="image", data={"url": url, "file": "abc.jpg"})


def _txt(text: str = "x"):
    return SimpleNamespace(type="text", data={"text": text})


class TestCollectImageUrl:
    def test_first_http_image(self):
        msg = [_txt(), _img("https://cdn/a.jpg"), _img("https://cdn/b.jpg")]
        assert paint_mod.collect_image_url(msg) == "https://cdn/a.jpg"

    def test_non_http_url_ignored(self):
        """NapCat 有时只给本机文件名（不是 URL），绝不能当地址用。"""
        msg = [_img("983973C6.jpg"), _img("file:///tmp/a.jpg")]
        assert paint_mod.collect_image_url(msg) == ""

    def test_empty(self):
        assert paint_mod.collect_image_url([]) == ""
        assert paint_mod.collect_image_url(None) == ""
        assert paint_mod.collect_image_url([_txt("只有文字")]) == ""

    def test_has_image_segment(self):
        assert paint_mod.has_image_segment([_img("")]) is True
        assert paint_mod.has_image_segment([_txt()]) is False
        assert paint_mod.has_image_segment(None) is False


class TestChooseReference:
    def _event(self, current=None, replied=None):
        reply = SimpleNamespace(message=replied) if replied is not None else None
        return SimpleNamespace(message=current or [], reply=reply)

    def test_current_message_wins(self):
        url, missing = paint_mod.choose_reference(
            self._event(current=[_img("https://cdn/now.jpg")], replied=[_img("https://cdn/old.jpg")])
        )
        assert url == "https://cdn/now.jpg" and missing is False

    def test_falls_back_to_replied_message(self):
        """引用一张图再 /paint —— 这是最常见的使用方式。"""
        url, missing = paint_mod.choose_reference(
            self._event(current=[_txt(" /paint 改成抱着手机")], replied=[_img("https://cdn/old.jpg")])
        )
        assert url == "https://cdn/old.jpg" and missing is False

    def test_image_without_url_is_reported(self):
        url, missing = paint_mod.choose_reference(self._event(replied=[_img("not-a-url")]))
        assert url == "" and missing is True

    def test_no_image_at_all(self):
        url, missing = paint_mod.choose_reference(self._event(current=[_txt("/paint 一只猫")]))
        assert url == "" and missing is False

    def test_no_reply_attribute(self):
        assert paint_mod.choose_reference(SimpleNamespace(message=[])) == ("", False)


class TestEditBody:
    def test_includes_input_fidelity(self):
        cfg = image_gen.get_config({"PAINT_API_KEY": "k"})
        body = image_gen.build_edit_body("把它改成抱着手机", cfg)
        assert body["input_fidelity"] == "high", "文档：编辑时用 high 尽量保留原主体"
        assert body["model"] == "gpt-image-2"
        for forbidden in ("stream", "partial_images", "style"):
            assert forbidden not in body

    def test_input_fidelity_can_be_disabled(self):
        cfg = image_gen.get_config({"PAINT_API_KEY": "k", "PAINT_INPUT_FIDELITY": ""})
        assert "input_fidelity" not in image_gen.build_edit_body("x", cfg)


class TestEditCall:
    @pytest.mark.asyncio
    async def test_edit_uploads_image_and_uses_extra_body(self, monkeypatch):
        captured = {}

        class FakeImages:
            async def edit(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    data=[SimpleNamespace(b64_json=PNG_B64, url=None, revised_prompt=None)]
                )

        monkeypatch.setattr(
            image_gen, "_get_client", lambda cfg: SimpleNamespace(images=FakeImages())
        )
        cfg = image_gen.get_config({"PAINT_API_KEY": "k"})
        image = await image_gen.edit_image("改成抱着手机", PNG_BYTES, "image/png", cfg)

        assert image is not None and image.data == PNG_BYTES
        assert captured["model"] == "gpt-image-2"
        assert captured["prompt"] == "改成抱着手机"
        # image 必须是 (文件名, 内容, mime) 三元组
        name, content, mime = captured["image"]
        assert name.endswith(".png") and content == PNG_BYTES and mime == "image/png"
        assert captured["extra_body"]["input_fidelity"] == "high"
        for key in ("output_format", "quality", "size", "response_format", "input_fidelity"):
            assert key not in captured, f"{key} 不该作为标准参数传给 SDK"

    @pytest.mark.asyncio
    async def test_edit_failure_returns_none(self, monkeypatch):
        class BoomImages:
            async def edit(self, **kwargs):
                raise RuntimeError("nope")

        monkeypatch.setattr(
            image_gen, "_get_client", lambda cfg: SimpleNamespace(images=BoomImages())
        )
        cfg = image_gen.get_config({"PAINT_API_KEY": "k"})
        assert await image_gen.edit_image("x", PNG_BYTES, "image/png", cfg) is None

    @pytest.mark.asyncio
    async def test_empty_reference_is_rejected(self):
        cfg = image_gen.get_config({"PAINT_API_KEY": "k"})
        assert await image_gen.edit_image("x", b"", "image/png", cfg) is None


@pytest.mark.asyncio
async def test_end_to_end_img2img(monkeypatch, tmp_path):
    """引用一张图 + /paint 描述 → 走图生图，并把参考图也存档。"""
    matcher = FakeMatcher()
    monkeypatch.setattr(paint_mod, "paint", matcher)
    monkeypatch.setattr(paint_mod, "PAINT_ADMINS", {1})
    monkeypatch.setattr(paint_mod, "PAINT_DIR", tmp_path)
    monkeypatch.setattr(paint_mod, "_state", paint_mod.QuotaState(last_at={}))
    monkeypatch.setattr(paint_mod, "is_group_allowed", lambda gid: True)
    monkeypatch.setattr(
        paint_mod, "PAINT_CONFIG", image_gen.get_config({"PAINT_API_KEY": "k"})
    )

    ref_bytes = b"\xff\xd8\xff" + b"ref-image"

    async def fake_fetch(url):
        assert url == "https://cdn/old.jpg"
        return ref_bytes

    edited_with = {}

    async def fake_edit(prompt, image, mime):
        edited_with.update(prompt=prompt, image=image, mime=mime)
        return image_gen.GeneratedImage(data=PNG_BYTES, mime="image/png", source="b64_json")

    async def fake_generate(prompt):  # pragma: no cover - 不该被调用
        raise AssertionError("有参考图时不应该走文生图")

    monkeypatch.setattr(paint_mod, "fetch_image_bytes", fake_fetch)
    monkeypatch.setattr(paint_mod, "edit_image", fake_edit)
    _stub_rewrite(monkeypatch)
    monkeypatch.setattr(paint_mod, "generate_image", fake_generate)
    context_calls = _record_context(monkeypatch)

    async def fake_line(prompt):
        return "改好啦～"

    monkeypatch.setattr(paint_mod, "persona_line", fake_line)

    event = _fake_event(text="/paint 把它改成抱着手机")
    event.message = [_txt(" /paint 把它改成抱着手机")]
    event.reply = SimpleNamespace(message=[_img("https://cdn/old.jpg")])

    with pytest.raises(_Finished):
        await paint_mod.handle_paint(event)

    # 改图时发给绘图模型的不是原话，而是"锚点 + 指令"：
    # 锚点由程序拼，模型改不掉 —— 这是防止照片里的人被换掉的最后一道保险
    assert edited_with["prompt"].startswith("在原图基础上修改")
    assert "保持原图里主体的长相与身份不变" in edited_with["prompt"]
    assert edited_with["prompt"].endswith("把它改成抱着手机")
    assert edited_with["image"] == ref_bytes
    assert edited_with["mime"] == "image/jpeg", "参考图是 jpg，上传时 mime 要跟着变"
    assert "[image:" in _flatten(matcher.finished[0])
    assert len(list(tmp_path.glob("*-ref.jpg"))) == 1, "参考图也要留档"
    meta = next(tmp_path.glob("*.txt")).read_text(encoding="utf-8")
    assert "mode: img2img" in meta
    # 上下文记录写的是"画了什么"（scene），不带程序拼的锚点
    assert context_calls[0]["content"] == f"[{paint_mod.BOT_NAME}画了一张图（在原图基础上修改）：把它改成抱着手机]"


@pytest.mark.asyncio
async def test_end_to_end_image_without_url_degrades(monkeypatch):
    """有图但拿不到地址 → 明确告诉用户，而不是悄悄退化成文生图。"""
    matcher = FakeMatcher()
    monkeypatch.setattr(paint_mod, "paint", matcher)
    monkeypatch.setattr(paint_mod, "PAINT_ADMINS", {1})
    monkeypatch.setattr(paint_mod, "_state", paint_mod.QuotaState(last_at={}))
    monkeypatch.setattr(paint_mod, "is_group_allowed", lambda gid: True)
    monkeypatch.setattr(
        paint_mod, "PAINT_CONFIG", image_gen.get_config({"PAINT_API_KEY": "k"})
    )

    async def boom(*a, **kw):  # pragma: no cover
        raise AssertionError("不该调用生图")

    _stub_rewrite(monkeypatch)
    monkeypatch.setattr(paint_mod, "generate_image", boom)
    monkeypatch.setattr(paint_mod, "edit_image", boom)

    event = _fake_event(text="/paint 改成抱着手机")
    event.message = [_txt(" /paint 改成抱着手机")]
    event.reply = SimpleNamespace(message=[_img("983973C6.jpg")])  # 只有文件名，没有 URL

    with pytest.raises(_Finished):
        await paint_mod.handle_paint(event)

    assert "取不到可用的地址" in _flatten(matcher.finished[0])


@pytest.mark.asyncio
async def test_end_to_end_success(monkeypatch, tmp_path):
    """管理员 + 白名单群 + 生图成功 → 发图 + 夜子一句话，并落盘存档。"""
    matcher = FakeMatcher()
    monkeypatch.setattr(paint_mod, "paint", matcher)
    monkeypatch.setattr(paint_mod, "PAINT_ADMINS", {1})
    monkeypatch.setattr(paint_mod, "PAINT_DIR", tmp_path)
    monkeypatch.setattr(paint_mod, "_state", paint_mod.QuotaState(last_at={}))
    monkeypatch.setattr(paint_mod, "is_group_allowed", lambda gid: True)
    monkeypatch.setattr(
        paint_mod,
        "PAINT_CONFIG",
        image_gen.get_config({"PAINT_API_KEY": "k", "PAINT_ENABLED": "true"}),
    )

    async def fake_generate(prompt):
        assert prompt == "一只猫"
        return image_gen.GeneratedImage(data=PNG_BYTES, mime="image/png", source="b64_json")

    async def fake_line(prompt):
        return "画好啦～"

    _stub_rewrite(monkeypatch)
    monkeypatch.setattr(paint_mod, "generate_image", fake_generate)
    monkeypatch.setattr(paint_mod, "persona_line", fake_line)

    event = _fake_event(text="/paint 一只猫")

    with pytest.raises(_Finished):
        await paint_mod.handle_paint(event)

    assert "夜子正在画" in _flatten(matcher.sent[0])
    assert "[image:" in _flatten(matcher.finished[0])
    assert "画好啦～" in _flatten(matcher.finished[0])
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 1


@pytest.mark.asyncio
async def test_end_to_end_non_admin_refused(monkeypatch):
    matcher = FakeMatcher()
    monkeypatch.setattr(paint_mod, "paint", matcher)
    monkeypatch.setattr(paint_mod, "PAINT_ADMINS", {1})
    monkeypatch.setattr(paint_mod, "is_group_allowed", lambda gid: True)
    event = _fake_event(user_id=999, text="/paint 猫")

    with pytest.raises(_Finished):
        await paint_mod.handle_paint(event)

    assert "管理员" in _flatten(matcher.finished[0])


@pytest.mark.asyncio
async def test_end_to_end_generation_failure_degrades(monkeypatch):
    """生图失败 → 只回一句降级文案，不抛异常、不占额度。"""
    matcher = FakeMatcher()
    state = paint_mod.QuotaState(last_at={})
    monkeypatch.setattr(paint_mod, "paint", matcher)
    monkeypatch.setattr(paint_mod, "PAINT_ADMINS", {1})
    monkeypatch.setattr(paint_mod, "_state", state)
    monkeypatch.setattr(paint_mod, "is_group_allowed", lambda gid: True)
    monkeypatch.setattr(
        paint_mod, "PAINT_CONFIG", image_gen.get_config({"PAINT_API_KEY": "k"})
    )

    async def boom(prompt):
        return None

    _stub_rewrite(monkeypatch)
    monkeypatch.setattr(paint_mod, "generate_image", boom)
    event = _fake_event(text="/paint 猫")

    with pytest.raises(_Finished):
        await paint_mod.handle_paint(event)

    assert "失败" in _flatten(matcher.finished[0])
    assert state.count == 0, "失败的请求不该占用每日额度"


# ======================================================================
# 能力声明：让聊天模型知道"自己会画画"（v0.10 修的真实问题）
# ======================================================================


class TestCapabilityState:
    def test_paint_capability_is_declared(self):
        from services.prompt_builder import _build_capability_state

        text = _build_capability_state(web_search_allowed=False, paint_allowed=True)
        assert "paint: true" in text
        assert "/paint" in text, "必须告诉模型命令怎么写"
        assert "画了一张图" in text, "必须告诉模型怎么认群聊里的出图记录"
        assert "不要否认" in text

    def test_paint_disabled_is_honest(self):
        from services.prompt_builder import _build_capability_state

        text = _build_capability_state(web_search_allowed=False, paint_allowed=False)
        assert "paint: false" in text
        assert "没有绘图能力" in text

    def test_web_search_line_still_there(self):
        from services.prompt_builder import _build_capability_state

        text = _build_capability_state(web_search_allowed=True, paint_allowed=False)
        assert "web_search: true" in text

    def test_context_marker_matches_capability_text(self):
        """上下文写入格式与能力声明里引用的格式必须是同一个（否则模型认不出来）。"""
        from services.prompt_builder import _build_capability_state, paint_context_text

        marker = paint_context_text("一只猫")
        text = _build_capability_state(web_search_allowed=False, paint_allowed=True)
        # 能力声明里引用的是省略号版本，取前缀比较
        prefix = paint_context_text("…").split("…")[0]
        assert prefix in text
        assert marker.startswith(prefix)


# ======================================================================
# 提示词改写：先让夜子把要求落成具体画面，再交给绘图模型（v0.10.1）
# ======================================================================


class TestCleanImagePrompt:
    """清理模型的输出：去引号 / 去包裹 / 去开场白 / 压成一行。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("一只趴在窗台上打盹的橘猫，暖色插画风格", "一只趴在窗台上打盹的橘猫，暖色插画风格"),
            ("「一只橘猫」", "一只橘猫"),
            ('"一只橘猫"', "一只橘猫"),
            ("《一只橘猫》", "一只橘猫"),
            ("`一只橘猫`", "一只橘猫"),
            ("好的，一只橘猫", "一只橘猫"),
            ("这是一幅一只橘猫", "一只橘猫"),
            ("一只橘猫\n坐在窗台上", "一只橘猫 坐在窗台上"),
            ("  一只橘猫  ", "一只橘猫"),
        ],
    )
    def test_cleaning(self, raw, expected):
        from services.prompt_builder import clean_image_prompt

        assert clean_image_prompt(raw) == expected

    def test_empty_and_truncate(self):
        from services.prompt_builder import clean_image_prompt

        assert clean_image_prompt("") == ""
        assert clean_image_prompt("   ") == ""
        assert len(clean_image_prompt("猫" * 500, max_chars=20)) == 20

    def test_prompt_asks_for_concreteness_and_her_own_taste(self):
        """改写提示必须要求"写清画面要素"且"体现你自己的偏好" —— 这是这次改动的目的。"""
        from services.prompt_builder import build_paint_prompt_messages

        msgs = build_paint_prompt_messages("画一个你最喜欢的东西", max_chars=100)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        text = msgs[1]["content"]
        assert "画一个你最喜欢的东西" in text
        assert "100" in text
        for keyword in ("主体", "场景", "风格", "你自己的喜好", "不要出现文字"):
            assert keyword in text, f"改写提示里缺少「{keyword}」的要求"


class TestRewritePrompt:
    @pytest.mark.asyncio
    async def test_uses_rewritten_prompt(self, monkeypatch):
        async def fake_ask(messages, model=None):
            assert messages[1]["role"] == "user"
            return "「一只蜷在旧书堆上打盹的橘猫，暖黄台灯光，厚涂插画风格」"

        monkeypatch.setattr(paint_mod, "ask_deepseek", fake_ask)
        monkeypatch.setattr(paint_mod, "PAINT_REWRITE_PROMPT", True)
        result = await paint_mod.rewrite_prompt("画一个你最喜欢的东西")
        assert result == "一只蜷在旧书堆上打盹的橘猫，暖黄台灯光，厚涂插画风格"

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(self, monkeypatch):
        async def boom(messages, model=None):
            raise RuntimeError("llm down")

        monkeypatch.setattr(paint_mod, "ask_deepseek", boom)
        monkeypatch.setattr(paint_mod, "PAINT_REWRITE_PROMPT", True)
        assert await paint_mod.rewrite_prompt("画一只猫") == "画一只猫"

    @pytest.mark.asyncio
    async def test_falls_back_on_empty_rewrite(self, monkeypatch):
        async def empty(messages, model=None):
            return "   "

        monkeypatch.setattr(paint_mod, "ask_deepseek", empty)
        monkeypatch.setattr(paint_mod, "PAINT_REWRITE_PROMPT", True)
        assert await paint_mod.rewrite_prompt("画一只猫") == "画一只猫"

    @pytest.mark.asyncio
    async def test_disabled_skips_llm_entirely(self, monkeypatch):
        async def boom(messages, model=None):  # pragma: no cover
            raise AssertionError("关掉改写时不该调用 LLM")

        monkeypatch.setattr(paint_mod, "ask_deepseek", boom)
        monkeypatch.setattr(paint_mod, "PAINT_REWRITE_PROMPT", False)
        assert await paint_mod.rewrite_prompt("画一只猫") == "画一只猫"


class TestEditModeRewrite:
    """改图模式：绝不能让改写把原图主体换掉（v0.10.2 的真实事故）。"""

    def test_edit_prompt_forbids_describing_the_source_image(self):
        from services.prompt_builder import build_paint_edit_messages

        msgs = build_paint_edit_messages("画一张上面这个人抱着手机的图片", max_chars=80)
        text = msgs[1]["content"]
        assert "画一张上面这个人抱着手机的图片" in text
        assert "80" in text
        for keyword in ("看不到那张图", "不要描述原图", "修改指令"):
            assert keyword in text, f"改图改写提示里缺少「{keyword}」"
        # 关键：不许提人物长相/性别，正是这一条被违反才画出了"黑发少女"
        assert "性别" in text and "长相" in text

    def test_edit_mode_does_not_use_the_persona(self):
        """人格正是"她把自己画进去"的来源，改图必须用中立的 system。"""
        from services.prompt_builder import (
            STATIC_SYSTEM_PROMPT,
            build_paint_edit_messages,
        )

        msgs = build_paint_edit_messages("改成抱着手机")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] != STATIC_SYSTEM_PROMPT
        assert len(msgs[0]["content"]) < 60, "改图模式不该带整份人格"

    def test_edit_mode_keeps_persona_for_text2img(self):
        """文生图仍然走人格版 —— 那是"画她喜欢的东西"这个功能的立身之本。"""
        from services.prompt_builder import (
            STATIC_SYSTEM_PROMPT,
            build_paint_prompt_messages,
        )

        assert build_paint_prompt_messages("画一只猫")[0]["content"] == STATIC_SYSTEM_PROMPT

    def test_anchor_is_deterministic(self):
        from services.prompt_builder import build_edit_prompt

        out = build_edit_prompt("把 T 恤换成红色")
        assert out.startswith("在原图基础上修改")
        assert "保持原图里主体的长相与身份不变" in out
        assert out.endswith("把 T 恤换成红色")

    def test_anchor_falls_back_to_original(self):
        from services.prompt_builder import build_edit_prompt

        out = build_edit_prompt("", "把帽子改成蓝色")
        assert out.endswith("把帽子改成蓝色")
        assert build_edit_prompt("", "") == ""

    @pytest.mark.asyncio
    async def test_editing_flag_selects_the_edit_instruction(self, monkeypatch):
        seen = {}

        async def fake_ask(messages, model=None):
            seen["system"] = messages[0]["content"]
            seen["user"] = messages[1]["content"]
            return "在原图里给这个人手里加一部手机"

        monkeypatch.setattr(paint_mod, "ask_deepseek", fake_ask)
        monkeypatch.setattr(paint_mod, "PAINT_REWRITE_PROMPT", True)

        result = await paint_mod.rewrite_prompt("画一张上面这个人抱着手机的图片", editing=True)
        assert result == "在原图里给这个人手里加一部手机"
        assert "不要描述原图" in seen["user"]
        assert "你自己的喜好" not in seen["user"], "改图模式不该要求她发挥个人审美"

    @pytest.mark.asyncio
    async def test_text2img_flag_keeps_the_old_instruction(self, monkeypatch):
        seen = {}

        async def fake_ask(messages, model=None):
            seen["user"] = messages[1]["content"]
            return "一只橘猫"

        monkeypatch.setattr(paint_mod, "ask_deepseek", fake_ask)
        monkeypatch.setattr(paint_mod, "PAINT_REWRITE_PROMPT", True)

        await paint_mod.rewrite_prompt("画一只猫")
        assert "你自己的喜好" in seen["user"]
        assert "不要描述原图" not in seen["user"]


@pytest.mark.asyncio
async def test_end_to_end_rewritten_prompt_is_what_gets_drawn(monkeypatch, tmp_path):
    """改写后的画面必须贯穿全链路：画它、存档记它、上下文写它。"""
    matcher = FakeMatcher()
    monkeypatch.setattr(paint_mod, "paint", matcher)
    monkeypatch.setattr(paint_mod, "PAINT_ADMINS", {1})
    monkeypatch.setattr(paint_mod, "PAINT_DIR", tmp_path)
    monkeypatch.setattr(paint_mod, "_state", paint_mod.QuotaState(last_at={}))
    monkeypatch.setattr(paint_mod, "is_group_allowed", lambda gid: True)
    monkeypatch.setattr(
        paint_mod, "PAINT_CONFIG", image_gen.get_config({"PAINT_API_KEY": "k"})
    )

    rewritten = "一只蜷在旧书堆上打盹的橘猫，暖黄台灯光，厚涂插画风格"

    async def fake_rewrite(prompt, editing=False):
        assert prompt == "画一个你最喜欢的东西"
        assert editing is False, "没有参考图时应该走文生图规则"
        return rewritten

    drawn = {}

    async def fake_generate(prompt):
        drawn["prompt"] = prompt
        return image_gen.GeneratedImage(data=PNG_BYTES, mime="image/png", source="b64_json")

    captured_line = {}

    async def fake_line(prompt):
        captured_line["prompt"] = prompt
        return "画好啦～"

    monkeypatch.setattr(paint_mod, "rewrite_prompt", fake_rewrite)
    monkeypatch.setattr(paint_mod, "generate_image", fake_generate)
    monkeypatch.setattr(paint_mod, "persona_line", fake_line)
    context_calls = _record_context(monkeypatch)

    event = _fake_event(text="/paint 画一个你最喜欢的东西")
    with pytest.raises(_Finished):
        await paint_mod.handle_paint(event)

    assert drawn["prompt"] == rewritten, "要画的是改写后的画面"
    assert captured_line["prompt"] == rewritten, "配文也要基于改写后的画面"
    meta = next(tmp_path.glob("*.txt")).read_text(encoding="utf-8")
    assert "asked: 画一个你最喜欢的东西" in meta, "存档要留下用户原话"
    assert f"image_prompt: {rewritten}" in meta, "存档要留下改写后的画面"
    assert rewritten in context_calls[0]["content"]
    text = _flatten(matcher.finished[0])
    assert "画好啦～" in text
    assert "我画的是" not in text, "改写后的提示词只进存档和上下文，不再附在消息里"
    assert rewritten not in text
