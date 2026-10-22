"""决策日志：带时间戳、只追加不可改的 JSONL。复盘只读。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from zhiku.calendar import now_shanghai

_PRIVACY_KEYS = frozenset(
    {
        "qty",
        "quantity",
        "cost",
        "cost_basis",
        "position_qty",
        "sell_qty",
        "merged_qty",
        "broker",
        "account",
        "name_personal",
    }
)


def _scrub(value: Any) -> Any:
    """日志不写数量/成本/账户。代码与公开价位可保留。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k).lower() in _PRIVACY_KEYS:
                continue
            out[k] = _scrub(v)
        return out
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


class DecisionLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, kind: str, payload: dict[str, Any], *, ts: datetime | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = (ts or now_shanghai()).isoformat()
        row = {"ts": stamp, "kind": kind, "payload": _scrub(payload)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def append_many(
        self, kind: str, rows: list[dict[str, Any]], *, ts: datetime | None = None
    ) -> int:
        n = 0
        for row in rows:
            self.append(kind, row, ts=ts)
            n += 1
        return n

    def read(self) -> list[dict[str, Any]]:
        """只读。不提供 update/delete。"""
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
