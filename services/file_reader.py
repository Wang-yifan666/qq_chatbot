"""services.file_reader：File Reader 的公共入口（v0.7）。

实现集中在 `services/perception/file_reader.py`（感知层），本模块只是把
公开 API 稳定地重导出，避免出现“两套文件读取实现”：

    from services.file_reader import read_file, FileReadResult, sanitize_file_name

File Reader 的职责**只有**：QQ file → 安全获取 → 类型判断 → 内容提取 →
结构化结果。它不负责 Persona、不负责回答用户、不知道 Prompt 的存在。
"""

from services.perception.file_reader import FileReadResult
from services.perception.file_reader import cleanup_result
from services.perception.file_reader import is_supported_extension
from services.perception.file_reader import read_file
from services.perception.file_reader import sanitize_file_name
from services.perception.parsers import parse_by_type
from services.perception.parsers import parse_docx
from services.perception.parsers import parse_pdf
from services.perception.parsers import parse_pptx
from services.perception.parsers import parse_text
from services.perception.parsers import parse_xlsx

__all__ = [
    "FileReadResult",
    "read_file",
    "sanitize_file_name",
    "cleanup_result",
    "is_supported_extension",
    "parse_by_type",
    "parse_text",
    "parse_pdf",
    "parse_docx",
    "parse_xlsx",
    "parse_pptx",
]
