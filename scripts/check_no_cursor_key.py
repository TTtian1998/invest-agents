#!/usr/bin/env python3
"""扫描仓库文本，禁止出现 Cursor 用户密钥前缀。"""

from __future__ import annotations

import sys
from pathlib import Path

NEEDLE = "crs" + "r_"
ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
    "dist",
    "build",
    "mlruns",
    "qlib_data",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".json",
    ".j2",
    ".sh",
    ".example",
    ".gitignore",
    ".cfg",
    ".ini",
}


def iter_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix and path.suffix not in TEXT_SUFFIXES and path.name not in {".env.example", ".gitignore"}:
            if path.name.startswith("."):
                pass
            else:
                continue
        out.append(path)
    return out


def main() -> int:
    hits: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if NEEDLE in text:
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        print("仓库里出现了 Cursor 密钥前缀：", file=sys.stderr)
        for item in hits:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
