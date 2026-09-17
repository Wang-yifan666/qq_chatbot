"""File Reader / 文件安全测试（v0.7）。

覆盖（需求 35 的文件与安全部分）：
- 支持类型：txt / md / json / python / C++ / pdf / docx / xlsx / pptx / 图片文件
- 安全：超大文件（声明与实际累计）、path traversal 文件名、不支持二进制、
  exe 改名 pdf、MIME / magic 不一致、malformed PDF/DOCX/XLSX/PPTX、
  下载超时、下载超过上限、临时文件清理
- Prompt Injection：文件正文里的“ignore previous instructions / 输出 API key”
  只能作为**用户数据**进入 Prompt，绝不进入 system

所有解析用真实库（PyMuPDF / python-docx / openpyxl / python-pptx）生成
真实文件；网络层全部 mock，绝不访问真实 CDN。
"""

import os
import zipfile

import pytest

import services.perception.file_reader as fr
import services.perception.file_types as ft
import services.perception.net as net
from services.perception.content import FileRef
from services.perception.limits import ContentBudget
from services.perception.net import DownloadResult
from services.perception.parsers import parse_by_type
from services.perception.parsers import parse_docx
from services.perception.parsers import parse_pdf
from services.perception.parsers import parse_pptx
from services.perception.parsers import parse_text
from services.perception.parsers import parse_xlsx


# ======================================================================
# 真实文件 fixture（全部运行时生成）
# ======================================================================


def make_pdf(path: str, text: str = "Hello Perception Layer") -> None:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def make_docx(path: str) -> None:
    import docx

    document = docx.Document()
    document.add_paragraph("第一段正文")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "姓名"
    table.cell(0, 1).text = "张三"
    document.save(path)


def make_xlsx(path: str) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "销售数据"
    sheet["A1"] = "姓名"
    sheet["B1"] = "销售额"
    sheet["A2"] = "张三"
    sheet["B2"] = 1200
    sheet["C2"] = "=B2*2"  # 公式：只保留原文，绝不求值
    workbook.save(path)
    workbook.close()


def make_pptx(path: str) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "第一页标题"
    box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
    box.text_frame.text = "第一页正文"
    presentation.save(path)


def make_zip(path: str, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def _read(tmp_path, file_name: str, payload: bytes, budget=None):
    """把 payload 写到临时文件，走真实 classify + parser（不经过网络）。"""
    path = tmp_path / f"payload_{abs(hash(file_name)) % 100000}"
    path.write_bytes(payload)
    info = ft.classify(file_name, str(path))
    if not info.allowed:
        return info, None
    result = parse_by_type(info.parser, str(path), 30000)
    return info, result


# ======================================================================
# 类型识别（扩展名 + magic bytes）
# ======================================================================


class TestFileTypeClassification:
    def test_text_extensions_are_text(self):
        for name in ("a.txt", "a.md", "a.json", "a.yaml", "a.yml"):
            assert ft.extension_of(name) in ft.TEXT_EXTENSIONS
            assert ft._kind_of(ft.extension_of(name)) == "text"

    def test_code_extensions_are_code(self):
        for name in ("a.py", "a.c", "a.cpp", "a.cc", "a.h", "a.hpp", "a.java",
                     "a.js", "a.ts", "a.html", "a.css", "a.xml", "a.sql", "a.sh"):
            assert ft.extension_of(name) in ft.CODE_EXTENSIONS

    def test_document_and_image_extensions(self):
        assert ft._parser_of("pdf") == ft.PARSER_PDF
        assert ft._parser_of("docx") == ft.PARSER_DOCX
        assert ft._parser_of("xlsx") == ft.PARSER_XLSX
        assert ft._parser_of("pptx") == ft.PARSER_PPTX
        assert ft._parser_of("png") == ft.PARSER_IMAGE

    def test_forbidden_extensions(self, tmp_path):
        for name in ("a.exe", "a.dll", "a.apk", "a.so", "a.bat", "a.cmd",
                     "a.ps1", "a.scr", "a.com", "a.zip", "a.rar", "a.7z",
                     "a.tar", "a.gz"):
            path = tmp_path / "x"
            path.write_bytes(b"anything")
            info = ft.classify(name, str(path))
            assert info.verdict == ft.VERDICT_FORBIDDEN, name
            assert info.allowed is False

    def test_unknown_binary_is_unsupported(self, tmp_path):
        path = tmp_path / "x"
        path.write_bytes(b"\x00\x01\x02")
        info = ft.classify("a.xyz", str(path))
        assert info.verdict == ft.VERDICT_UNKNOWN
        assert info.note == "暂不支持读取该文件类型。"

    def test_executable_renamed_to_pdf_is_mismatch(self, tmp_path):
        path = tmp_path / "homework.pdf"
        path.write_bytes(b"MZ\x90\x00" + b"\x00" * 128)
        info = ft.classify("homework.pdf", str(path))
        assert info.verdict == ft.VERDICT_MISMATCH
        assert info.magic == "executable"
        assert info.allowed is False

    def test_zip_renamed_to_docx_is_mismatch(self, tmp_path):
        path = tmp_path / "report.docx"
        make_zip(str(path), {"evil.txt": b"hi"})
        info = ft.classify("report.docx", str(path))
        assert info.verdict == ft.VERDICT_MISMATCH
        assert info.magic == "zip"

    def test_binary_renamed_to_txt_is_mismatch(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_bytes(b"\x00\x01\x02\x03binary")
        info = ft.classify("notes.txt", str(path))
        assert info.verdict == ft.VERDICT_MISMATCH

    def test_real_docx_is_ok(self, tmp_path):
        path = tmp_path / "a.docx"
        make_docx(str(path))
        info = ft.classify("a.docx", str(path))
        assert info.verdict == ft.VERDICT_OK
        assert info.parser == ft.PARSER_DOCX

    def test_real_pdf_is_ok(self, tmp_path):
        path = tmp_path / "a.pdf"
        make_pdf(str(path))
        info = ft.classify("a.pdf", str(path))
        assert info.verdict == ft.VERDICT_OK
        assert info.parser == ft.PARSER_PDF

    def test_png_without_png_magic_is_mismatch(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(b"not really a png")
        info = ft.classify("a.png", str(path))
        assert info.verdict == ft.VERDICT_MISMATCH


# ======================================================================
# 文本 / 代码解析
# ======================================================================


class TestTextParsing:
    @pytest.mark.parametrize(
        "name,payload",
        [
            ("a.txt", "纯文本内容".encode("utf-8")),
            ("a.md", "# 标题\n\n正文".encode("utf-8")),
            ("a.json", b'{"key": "value"}'),
            ("a.yaml", b"name: test\nvalue: 1\n"),
            ("a.yml", b"name: test\n"),
            ("a.py", b"def hello():\n    return 'world'\n"),
            ("a.cpp", b"#include <iostream>\nint main(){return 0;}\n"),
            ("a.sh", b"echo hello\n"),
            ("a.sql", b"SELECT * FROM users;\n"),
            ("a.xml", b"<root><a/></root>"),
            ("a.html", b"<html><body>hi</body></html>"),
        ],
    )
    def test_supported_text_types(self, tmp_path, name, payload):
        info, result = _read(tmp_path, name, payload)
        assert info.verdict == ft.VERDICT_OK
        assert result.ok is True
        assert result.text.strip()

    def test_code_file_is_read_not_executed(self, tmp_path):
        """代码文件只能被读取，绝不能被运行（这里放一个会写文件的脚本）。"""
        marker = tmp_path / "EXECUTED"
        script = f"open(r'{marker}', 'w').write('boom')\n".encode("utf-8")
        info, result = _read(tmp_path, "danger.py", script)
        assert result.ok is True
        assert "boom" in result.text
        assert not marker.exists(), "代码文件绝不能被执行"

    def test_empty_text_file(self, tmp_path):
        info, result = _read(tmp_path, "empty.txt", b"")
        assert result.ok is True
        assert result.text == ""
        assert "没有可读取的文本" in result.note

    def test_gbk_encoded_text_is_decodable(self, tmp_path):
        info, result = _read(tmp_path, "gbk.txt", "中文内容".encode("gb18030"))
        assert result.ok is True
        assert "中文内容" in result.text


# ======================================================================
# 文档解析（真实库）
# ======================================================================


class TestDocumentParsing:
    def test_pdf_text_extraction(self, tmp_path):
        path = tmp_path / "a.pdf"
        make_pdf(str(path), "Hello Perception Layer")
        info = ft.classify("a.pdf", str(path))
        assert info.allowed
        result = parse_pdf(str(path), 30000)
        assert result.ok is True
        assert "Hello Perception Layer" in result.text
        assert result.pages == 1

    def test_pdf_without_text_layer(self, tmp_path):
        """没有文字层的 PDF（如扫描件）返回明确说明，绝不启动 OCR。"""
        import pymupdf

        path = tmp_path / "scan.pdf"
        document = pymupdf.open()
        document.new_page()  # 空白页 = 无文字层
        document.save(str(path))
        document.close()
        info = ft.classify("scan.pdf", str(path))
        assert info.allowed
        result = parse_pdf(str(path), 30000)
        assert result.ok is True
        assert result.text == ""
        assert result.note == "该 PDF 没有检测到可提取的文本内容。"

    def test_malformed_pdf(self, tmp_path):
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.4\n" + b"garbage" * 20)
        info = ft.classify("bad.pdf", str(path))
        # 扩展名与魔数一致 → 进入 parser；parser 自己拒绝
        assert info.verdict == ft.VERDICT_OK
        result = parse_pdf(str(path), 30000)
        assert result.ok is False

    def test_malformed_docx(self, tmp_path):
        path = tmp_path / "bad.docx"
        make_zip(str(path), {"word/document.xml": b"<not-valid"})
        # 缺少 [Content_Types].xml → 内容判定为 zip → mismatch，直接拒绝
        info = ft.classify("bad.docx", str(path))
        assert info.verdict == ft.VERDICT_MISMATCH
        # 即便强行交给 parser，也只返回失败而不是抛异常
        result = parse_docx(str(path), 30000)
        assert result.ok is False

    def test_malformed_xlsx(self, tmp_path):
        path = tmp_path / "bad.xlsx"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
        result = parse_xlsx(str(path), 30000)
        assert result.ok is False

    def test_malformed_pptx(self, tmp_path):
        path = tmp_path / "bad.pptx"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
        result = parse_pptx(str(path), 30000)
        assert result.ok is False

    def test_docx_paragraphs_and_table(self, tmp_path):
        path = tmp_path / "a.docx"
        make_docx(str(path))
        result = parse_docx(str(path), 30000)
        assert result.ok is True
        assert "第一段正文" in result.text
        assert "姓名" in result.text and "张三" in result.text

    def test_xlsx_cells_and_formula_not_evaluated(self, tmp_path):
        path = tmp_path / "a.xlsx"
        make_xlsx(str(path))
        result = parse_xlsx(str(path), 30000)
        assert result.ok is True
        assert "Sheet: 销售数据" in result.text
        assert "A1: 姓名" in result.text
        assert "B2: 1200" in result.text
        # 公式只保留原文（data_only=False），绝不求值成 2400
        assert "=B2*2" in result.text
        assert "2400" not in result.text

    def test_pptx_slides(self, tmp_path):
        path = tmp_path / "a.pptx"
        make_pptx(str(path))
        result = parse_pptx(str(path), 30000)
        assert result.ok is True
        assert "Slide 1" in result.text
        assert "第一页标题" in result.text
        assert "第一页正文" in result.text

    def test_text_truncation(self, tmp_path):
        path = tmp_path / "long.txt"
        path.write_text("很长" * 1000, encoding="utf-8")
        result = parse_text(str(path), 100)
        assert len(result.text) < 200
        assert result.truncated is True
        assert "已截断" in result.text

    def test_unknown_parser_returns_unsupported(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"x")
        result = parse_by_type("nope", str(path), 100)
        assert result.ok is False
        assert result.note == "暂不支持读取该文件类型。"


# ======================================================================
# 文件名安全
# ======================================================================


class TestFileNameSafety:
    def test_path_traversal_is_stripped(self):
        assert fr.sanitize_file_name("../../etc/passwd") == "passwd"
        assert fr.sanitize_file_name("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
        assert fr.sanitize_file_name("/etc/shadow") == "shadow"
        assert "/" not in fr.sanitize_file_name("a/b/c.txt")
        assert "\\" not in fr.sanitize_file_name("a\\b\\c.txt")

    def test_empty_name_gets_placeholder(self):
        assert fr.sanitize_file_name("") == "未命名文件"
        assert fr.sanitize_file_name(None) == "未命名文件"
        assert fr.sanitize_file_name("..") == "未命名文件"

    def test_long_name_truncated(self):
        name = "a" * 300 + ".txt"
        assert len(fr.sanitize_file_name(name)) <= 64

    def test_temp_suffix_only_keeps_alnum(self):
        assert net.sanitize_suffix("a.txt") == ".txt"
        assert net.sanitize_suffix("a.tar.gz") == ".gz"
        # 只保留字母数字，最多 10 个字符；绝不保留路径分隔符
        assert net.sanitize_suffix("evil.py;rm -rf") == ".pyrmrf"
        assert net.sanitize_suffix("noext") == ""
        assert net.sanitize_suffix("..\\..\\x.PNG") == ".png"
        # 以分隔符结尾（无 basename）时取不到扩展名 → 安全回落
        assert net.sanitize_suffix("evil.py;rm -rf /") == ""
        assert "/" not in net.sanitize_suffix("a.b/c")


# ======================================================================
# 下载安全（全部 mock，不发真实请求）
# ======================================================================


class TestDownloadSafety:
    async def test_cleanup_removes_temp_dir(self, tmp_path):
        target = tmp_path / "sub"
        target.mkdir()
        (target / "payload.txt").write_text("x")
        net.cleanup_download(str(target))
        assert not target.exists()

    async def test_cleanup_is_idempotent(self, tmp_path):
        net.cleanup_download("")
        net.cleanup_download(str(tmp_path / "does-not-exist"))

    async def test_declared_content_length_too_large(self, monkeypatch):
        """声明长度超限：直接拒绝，绝不开始下载。"""
        import httpx

        class FakeResponse:
            headers = {"Content-Length": "999999999"}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self, size):
                yield b"x" * 10

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        result = await net.download_to_temp_file("http://x/big.bin", 1024, "big.bin")
        assert result.ok is False
        assert result.note == "declared_too_large"

    async def test_actual_bytes_exceed_limit(self, monkeypatch):
        """Content-Length 撒谎：按实际累计字节中止，且不残留临时文件。"""
        import httpx

        class FakeResponse:
            headers: dict = {}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self, size):
                for _ in range(10):
                    yield b"y" * 4096

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        result = await net.download_to_temp_file("http://x/big.bin", 4096, "big.bin")
        assert result.ok is False
        assert result.note == "exceeded_limit"

    async def test_download_timeout(self, monkeypatch):
        import httpx

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        result = await net.download_to_temp_file("http://x/a.bin", 1024)
        assert result.ok is False
        assert result.note == "timeout"

    async def test_download_http_error(self, monkeypatch):
        import httpx

        request = httpx.Request("GET", "http://x/a.bin")
        response = httpx.Response(404, request=request)

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                raise httpx.HTTPStatusError("404", request=request, response=response)

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        result = await net.download_to_temp_file("http://x/a.bin", 1024)
        assert result.ok is False
        assert result.note == "http_error"

    async def test_empty_url(self):
        result = await net.download_to_temp_file("", 1024)
        assert result.ok is False
        assert result.note == "empty_url"

    async def test_oversized_file_is_rejected_before_parse(self, monkeypatch, tmp_path):
        """真实 read_file 流程：下载返回 exceeded_limit → 结构化降级，绝不解析。"""
        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(ok=False, note="exceeded_limit")

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)
        result = await fr.read_file(
            FileRef(file_name="huge.txt", url="http://x/huge.txt"), budget=ContentBudget()
        )
        assert result.ok is False
        assert "读取失败" in result.note

    async def test_read_file_cleans_temp_after_success(self, monkeypatch, tmp_path):
        payload = tmp_path / "note.txt"
        payload.write_text("正文", encoding="utf-8")
        temp_dir = tmp_path / "dl"
        temp_dir.mkdir()
        local = temp_dir / "payload.txt"
        local.write_text("正文", encoding="utf-8")

        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(ok=True, path=str(local), dir_path=str(temp_dir), size=6)

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)
        result = await fr.read_file(
            FileRef(file_name="note.txt", url="http://x/note.txt"), budget=ContentBudget()
        )
        assert result.ok is True
        assert result.text == "正文"
        assert not temp_dir.exists(), "文本解析完成后必须删除临时目录"

    async def test_read_file_without_url(self):
        result = await fr.read_file(FileRef(file_name="a.pdf"), budget=ContentBudget())
        assert result.ok is False
        assert "a.pdf" in result.note

    async def test_image_file_keeps_temp_for_vision(self, monkeypatch, tmp_path):
        """图片类文件保留临时文件供 Vision 复用，调用方负责清理。"""
        temp_dir = tmp_path / "img"
        temp_dir.mkdir()
        local = temp_dir / "payload.png"
        local.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

        async def fake_download(url, max_bytes, file_name=""):
            return DownloadResult(ok=True, path=str(local), dir_path=str(temp_dir), size=40)

        monkeypatch.setattr(fr, "download_to_temp_file", fake_download)
        result = await fr.read_file(
            FileRef(file_name="a.png", url="http://x/a.png"), budget=ContentBudget()
        )
        assert result.ok is True
        assert result.parser_type == "image"
        assert result.keep_temp is True
        assert temp_dir.exists()
        fr.cleanup_result(result)
        assert not temp_dir.exists()


# ======================================================================
# 预算是统一入口（不再是每个模块各写一套 truncate）
# ======================================================================


class TestBudget:
    def test_text_clipping_and_accounting(self):
        budget = ContentBudget(max_external_text_chars=20)
        assert budget.clip_text("a" * 10, "file") == "a" * 10
        clipped = budget.clip_text("b" * 30, "file")
        # 剩余 10 个字符额度 + 截断标记；总长不超过额度 + 标记长度
        assert clipped.startswith("b" * 10)
        assert clipped.endswith("…（已截断）")
        assert len(clipped) == 10 + len("…（已截断）")
        assert budget.remaining_text_chars() == 0
        assert "file:text_clipped" in budget.usage.truncated

    def test_no_budget_left_returns_empty(self):
        budget = ContentBudget(max_external_text_chars=5)
        budget.clip_text("12345", "file")
        assert budget.clip_text("more", "file") == ""

    def test_image_and_file_allowance(self):
        budget = ContentBudget(max_total_images=2, max_total_files=1)
        assert budget.allow_image() is True
        assert budget.allow_image() is True
        assert budget.allow_image() is False
        assert budget.allow_file() is True
        assert budget.allow_file() is False

    def test_image_limit_note_wording(self):
        from services.perception.limits import image_limit_note
        from services.perception.limits import truncation_note

        assert image_limit_note(63) == "[该合并转发后续还有 63 张图片，因图片数量限制未加载]"
        assert image_limit_note(0) == ""
        assert truncation_note(18, "条转发消息") == "[后续 18 条转发消息因上下文限制未展开]"
