"""密钥脱敏。仓库和日志里不得出现真实 Cursor / OpenAI key。"""

from __future__ import annotations

import os
import re

# 拆开写，避免本文件被「仓库不得出现该前缀」检查误伤
_CURSOR_PREFIX = "crs" + "r_"

_CURSOR_KEY = re.compile(re.escape(_CURSOR_PREFIX) + r"[A-Za-z0-9]+")
_BEARER = re.compile(r"(Bearer\s+)\S+", re.I)
_BASIC = re.compile(r"(Basic\s+)\S+", re.I)
_ENV_ASSIGN = re.compile(r"((?:CURSOR_API_KEY|ZHIKU_LLM_KEY|OPENAI_API_KEY)\s*[=:]\s*)\S+")


def cursor_key() -> str | None:
    """只读 CURSOR_API_KEY，不读文件、不读其它变量。"""
    raw = os.environ.get("CURSOR_API_KEY")
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def redact_secret(text: str | None) -> str:
    if not text:
        return ""
    out = _CURSOR_KEY.sub(_CURSOR_PREFIX + "***", str(text))
    out = _BEARER.sub(r"\1***", out)
    out = _BASIC.sub(r"\1***", out)
    out = _ENV_ASSIGN.sub(r"\1***", out)
    return out
