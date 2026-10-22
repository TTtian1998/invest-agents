"""仓库不得出现 Cursor 密钥前缀。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_repo_has_no_cursor_key_prefix() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_no_cursor_key.py"
    ran = subprocess.run([sys.executable, str(script)], check=False)
    assert ran.returncode == 0
