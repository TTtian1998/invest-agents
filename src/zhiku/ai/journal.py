"""AI 结论决策日志。每条带当时可见信息和时间戳。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from zhiku.calendar import now_shanghai


class DecisionJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(entry)
        row.setdefault("ts", now_shanghai().isoformat(timespec="seconds"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_result(self, result, *, available: dict[str, Any] | None = None) -> int:
        n = 0
        asof = getattr(result, "asof", None)
        backend = getattr(result, "backend", "")
        for ext in getattr(result, "extractions", []) or []:
            self.append(
                {
                    "kind": "extraction",
                    "asof": asof,
                    "backend": backend,
                    "symbol": ext.symbol,
                    "payload": ext.model_dump(mode="json"),
                    "available": available or {},
                }
            )
            n += 1
        for j in getattr(result, "judgments", []) or []:
            self.append(
                {
                    "kind": "judgment",
                    "asof": asof,
                    "backend": backend,
                    "symbol": j.symbol,
                    "stance": j.stance,
                    "confidence": j.confidence,
                    "invalidation": j.invalidation,
                    "time_stop_days": j.time_stop_days,
                    "formula_score": j.formula_score,
                    "formula_rank": j.formula_rank,
                    "cites": [c.cite for c in j.long_case + j.short_case],
                    "payload": j.model_dump(mode="json"),
                    "available": available or {},
                }
            )
            n += 1
        if getattr(result, "regime", None) is not None:
            self.append(
                {
                    "kind": "regime",
                    "asof": asof,
                    "backend": backend,
                    "payload": result.regime.model_dump(mode="json"),
                    "available": available or {},
                }
            )
            n += 1
        return n


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
