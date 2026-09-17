"""文档一致性检查：README 里声明的文件是否与真实仓库一致。

用法（在项目根目录执行）：
    python scripts/check_readme_consistency.py

检查三件事：
1. **漏记**：`services/` `plugins/` `scripts/` `tests/` `docs/` 下真实存在、
   但 README 目录树里一个字都没提的文件；
2. **幽灵**：README 目录树里列了、磁盘上却不存在的文件（会误导克隆者）；
3. **顶层关键文件**是否存在。

已被 `.gitignore` 忽略的文件（本地一次性脚本、评测产物、真实数据）会被单独列出并
**不算漏记** —— 它们本来就不该出现在公开 README 里。

返回码：有漏记或幽灵 → 1（可直接用于 CI / 手动自检）。
"""

import re
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# 只检查「应该被记录在目录树里」的源码 / 脚本 / 测试 / 文档目录
CHECK_DIRS = ["services", "plugins", "scripts", "tests", "docs"]
IGNORE_NAMES = {"__init__.py"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}

# 顶层文件：要么存在（运行时生成 / 已忽略），要么在 README 里说明
TOP_LEVEL = [
    "bot.py",
    "requirements.txt",
    "requirements-dev.txt",
    "pytest.ini",
    ".gitignore",
    ".env.example",
    "README.md",
]

# 目录树里以「顶层」身份出现、不在 CHECK_DIRS 里的条目（它们有单独的说明段落）
TOP_LEVEL_TREE_ONLY = {
    ".env",
    "persona.txt",
    ".gitkeep",
    "chat_history.db",
    "qq_ai_bot.db",
}


def is_ignored(path: Path) -> bool:
    """该文件是否被 .gitignore 忽略（忽略失败时当作未忽略，宁可多报）。"""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=ROOT,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def main() -> int:
    readme_path = ROOT / "README.md"
    if not readme_path.is_file():
        print("[ERROR] 找不到 README.md", file=sys.stderr)
        return 1
    readme = readme_path.read_text(encoding="utf-8", errors="replace")

    missing: dict[str, list[str]] = {}
    ignored_files: list[str] = []
    for dirname in CHECK_DIRS:
        base = ROOT / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name in IGNORE_NAMES:
                continue
            if "__pycache__" in path.parts or path.suffix in IGNORE_SUFFIXES:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if path.name in readme:
                continue
            if is_ignored(path):
                ignored_files.append(rel)
                continue
            missing.setdefault(dirname, []).append(rel)

    print("=== 1. README 未提及（应记录却漏记）===")
    if not missing:
        print("  ✅ 全部已记录")
    else:
        for dirname, names in missing.items():
            print(f"  [{dirname}]")
            for name in names:
                print(f"    ✗ {name}")

    if ignored_files:
        print("\n=== 已 gitignore（本就不该出现在公开 README，不算漏记）===")
        for name in ignored_files:
            print(f"    · {name}")

    # 反向：目录树里写了、磁盘上不存在
    print("\n=== 2. README 目录树里的幽灵条目 ===")
    tree = re.search(r"## 目录结构\s*```(.*?)```", readme, re.S)
    ghosts: list[str] = []
    if not tree:
        print("  [warn] 没找到「## 目录结构」代码块，跳过")
    else:
        for line in tree.group(1).splitlines():
            m = re.match(
                r"^[\s│├└─]*([\w./\-]+\.(?:py|md|json|jsonl|txt|ini|example|db|sh))\s", line
            )
            if not m:
                continue
            name = m.group(1)
            if name in TOP_LEVEL or name in TOP_LEVEL_TREE_ONLY:
                continue  # 顶层 / 运行时生成，由第 3 节或单独段落负责
            if not any(
                (ROOT / d / name).is_file() or (ROOT / d / "perception" / name).is_file()
                for d in CHECK_DIRS
            ):
                ghosts.append(name)
        if ghosts:
            for name in sorted(set(ghosts)):
                print(f"    ✗ {name}")
        else:
            print("  ✅ 目录树里的条目都存在")

    print("\n=== 3. 顶层关键文件 ===")
    top_missing = []
    for name in TOP_LEVEL:
        exists = (ROOT / name).is_file()
        print(f"  {'✓' if exists else '✗'} {name}")
        if not exists:
            top_missing.append(name)

    ok = not missing and not ghosts and not top_missing
    print("\n" + ("✅ README 与仓库一致" if ok else "❌ 存在不一致，请按上面提示修正"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
