"""推送前隐私守门人：检查「即将提交到 GitHub」的文件里是否残留真实群聊数据。

用法（在项目根目录执行）：
    python scripts/scan_privacy.py

## 设计要点（重要）

本文件会被提交进仓库，因此**绝不硬编码任何真实 QQ 号 / 群号 / 昵称**——
否则守门人自己就成了泄露源。敏感值全部在运行时从**已被 .gitignore 忽略**的
数据源推导：

| 来源 | 取什么 |
| --- | --- |
| `.env`（已忽略） | 所有 `*_ID` / `*_IDS` / `*_QQ` 键里的纯数字（群号、close 用户、管理员） |
| `data/chat_history.db`（已忽略，只读） | `messages` 表里的全部 `user_id`（含机器人自己）与 `nickname` |
| 内置常量 | 只有**结构性特征**：腾讯 CDN 域名 —— 这些不是个人信息 |

检查范围 = git 认为会提交的文件（已跟踪的修改 + 未跟踪且未被忽略的），
以 `git status --porcelain -uall` 为准；正确忽略的数据文件不会误报。

昵称只在**非代码文件**（md/json/txt…）里匹配，避免 ASCII 短昵称在源码里误伤。

发现命中时返回码为 1 —— 已配置为 git pre-push hook（`.git/hooks/pre-push`），
临时跳过：`git push --no-verify`。
"""

import re
import sqlite3
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# 结构性特征（非个人信息）：腾讯文件 / 图片 CDN。命中说明把真实聊天资源地址写进了仓库。
STRUCTURAL_PATTERNS = {
    # 只保留真实 CDN 域名：命中说明把聊天资源地址写进了仓库。
    # 刻意不匹配 /ftn_handler/ 这类通用 URL 路径段 —— 太泛，会把脱敏示例也误伤。
    "腾讯CDN直链": r"(gzc-download\.ftn\.qq\.com|multimedia\.nt\.qq\.com\.cn)",
}

TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".txt", ".ini", ".cfg",
    ".yml", ".yaml", ".example", ".toml", ".sh",
}
CODE_SUFFIXES = {".py", ".sh", ".ini", ".cfg", ".toml", ".yml", ".yaml", ".example"}

# QQ 号 / 群号长度区间（5~12 位）
ID_RE = re.compile(r"^\d{5,12}$")


def load_env_tokens() -> set[str]:
    """从 .env（已忽略）里取所有 ID 类键的纯数字值。"""
    tokens: set[str] = set()
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return tokens
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if not re.fullmatch(r"[A-Z0-9_]*(ID|IDS|QQ)", key.strip().upper()):
            continue
        for part in re.split(r"[,\s]+", value.strip()):
            if ID_RE.match(part):
                tokens.add(part)
    return tokens


def load_db_tokens() -> tuple[set[str], set[str], set[str]]:
    """从本地聊天库（已忽略，只读）取真实 user_id 与昵称。

    返回 (真实ID, 群成员昵称, 机器人自己的昵称)。
    机器人昵称（assistant 行的 nickname，如「夜子」）**不算敏感**——
    它是项目主角的名字，README / 文档里到处都是，必须排除，否则全是误报。
    """
    ids: set[str] = set()
    nicks: set[str] = set()
    bot_nicks: set[str] = set()
    db = ROOT / "data" / "chat_history.db"
    if not db.is_file():
        return ids, nicks, bot_nicks
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for (uid,) in con.execute("SELECT DISTINCT user_id FROM messages"):
                text = str(uid)
                if ID_RE.match(text):
                    ids.add(text)
            for (nick,) in con.execute(
                "SELECT DISTINCT nickname FROM messages WHERE nickname IS NOT NULL"
            ):
                _add_nick(nicks, nick)
            for (nick,) in con.execute(
                "SELECT DISTINCT nickname FROM messages "
                "WHERE role = 'assistant' AND nickname IS NOT NULL"
            ):
                _add_nick(bot_nicks, nick)
            for (nick,) in con.execute(
                "SELECT DISTINCT latest_nickname FROM users WHERE latest_nickname IS NOT NULL"
            ):
                _add_nick(nicks, nick)
        finally:
            con.close()
    except Exception as exc:
        print(f"  [warn] 读取本地聊天库失败，昵称检测降级：{type(exc).__name__}")
    # 机器人自己的昵称从敏感集合里剔除
    nicks -= bot_nicks
    return ids, nicks, bot_nicks


def _add_nick(bucket: set[str], value) -> None:
    text = str(value or "").strip()
    # 2~20 字符：太短容易误伤（单字、符号），太长不像昵称
    if 2 <= len(text) <= 20 and not text.isdigit():
        bucket.add(text)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def candidates() -> list[Path]:
    """git 会提交的文件：已修改的跟踪文件 + 未被忽略的未跟踪文件。"""
    out = git("status", "--porcelain", "-uall")
    files: list[Path] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:].strip()
        if status.strip() == "D":  # 删除的文件无需扫描
            continue
        rel = Path(path.strip('"'))
        full = ROOT / rel
        if not full.is_file() or full.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if rel.as_posix() == "scripts/scan_privacy.py":  # 守门人自己（已无敏感值）
            continue
        files.append(rel)
    return sorted(set(files))


def main() -> int:
    env_tokens = load_env_tokens()
    db_ids, db_nicks, bot_nicks = load_db_tokens()
    id_tokens = env_tokens | db_ids

    print("=== 数据源 ===")
    print(f"  .env 里的 ID 类值 : {len(env_tokens)} 个")
    print(f"  聊天库里的 ID     : {len(db_ids)} 个（含机器人自己）")
    print(f"  群成员昵称        : {len(db_nicks)} 个（已排除机器人昵称 {len(bot_nicks)} 个）")
    if not id_tokens:
        print("  [warn] 没有取到任何真实 ID —— 检查将大幅降级（只有结构性特征生效）")

    files = candidates()
    print(f"\n=== 扫描 {len(files)} 个「将被提交」的文本文件 ===\n")

    findings = 0
    for rel in files:
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        hits: list[tuple[str, int, str]] = []

        for token in id_tokens:
            for m in re.finditer(rf"(?<!\d){re.escape(token)}(?!\d)", text):
                hits.append(("真实ID", text[: m.start()].count("\n") + 1, token))

        # 昵称只在非代码文件里查，避免短 ASCII 昵称在源码中误伤
        if rel.suffix.lower() not in CODE_SUFFIXES:
            for nick in db_nicks:
                for m in re.finditer(re.escape(nick), text):
                    hits.append(("群友昵称", text[: m.start()].count("\n") + 1, nick))

        for label, pattern in STRUCTURAL_PATTERNS.items():
            for m in re.finditer(pattern, text):
                hits.append((label, text[: m.start()].count("\n") + 1, m.group(0)))

        if not hits:
            continue
        findings += len(hits)
        print(f"⚠ {rel}")
        seen = set()
        for label, line_no, value in sorted(hits, key=lambda h: h[1]):
            key = (label, line_no, value)
            if key in seen:
                continue
            seen.add(key)
            print(f"    L{line_no:<5} [{label}] {value}")
        print()

    if findings == 0:
        print("✅ 未发现真实群聊数据，可以安全提交")
        return 0
    print(f"❌ 共 {findings} 处命中，请先脱敏或加入 .gitignore")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
