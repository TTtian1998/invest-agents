"""按 provider+symbol+date+kind 做简单 SQLite 缓存。"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any


class SQLiteCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    provider TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    d TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (provider, symbol, d, kind)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, provider: str, symbol: str, d: date, kind: str) -> Any | None:
        key = d.isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM cache WHERE provider=? AND symbol=? AND d=? AND kind=?",
                (provider, symbol, key, kind),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, provider: str, symbol: str, d: date, kind: str, payload: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache(provider, symbol, d, kind, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (provider, symbol, d.isoformat(), kind, json.dumps(payload, ensure_ascii=False, default=str)),
            )
