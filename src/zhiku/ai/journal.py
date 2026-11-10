"""Append-only review records plus backward-compatible scorecard events."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from zhiku.ai.pack import review_result_from_obj, validate_review_pack, validate_review_result
from zhiku.ai.schema import ReviewResult
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
        self._append_many([entry])

    def _append_many(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = now_shanghai().isoformat(timespec="seconds")
        rows = [{"ts": timestamp, **entry} for entry in entries]
        # A complete review and its scorecard projection are written together.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def latest_result_for(self, pack: dict[str, Any]) -> ReviewResult | None:
        """Legacy same-day symbol events are intentionally never replayed as a review."""
        validate_review_pack(pack)
        for entry in reversed(self.read()):
            if entry.get("kind") != "review_result" or entry.get("snapshot_id") != pack["snapshot_id"]:
                continue
            if entry.get("asof") != pack["asof"]:
                continue
            try:
                result = review_result_from_obj(entry.get("payload"))
                return validate_review_result(result, pack)
            except (ValueError, TypeError, KeyError):
                # Old/corrupt/incompatible records remain auditable but cannot be consumed.
                continue
        return None

    def record_pack_export(self, pack: dict[str, Any], path: Path) -> bool:
        validate_review_pack(pack)
        if any(
            entry.get("kind") == "pack_exported" and entry.get("snapshot_id") == pack["snapshot_id"]
            for entry in self.read()
        ):
            return False
        self.append({
            "kind": "pack_exported",
            "snapshot_id": pack["snapshot_id"],
            "asof": pack["asof"],
            "backend": "external",
            "path": str(path),
            "n_candidates": len(pack["candidates"]),
        })
        return True

    def append_result(
        self,
        result: ReviewResult,
        *,
        pack: dict[str, Any] | None = None,
        available: dict[str, Any] | None = None,
    ) -> int:
        """Validate and persist one immutable review; repeated imports are idempotent.

        Unbound legacy results retain their scorecard events only. A result with a
        snapshot_id requires its actual evidence pack, so it cannot bypass validation.
        """
        if pack is not None:
            result = validate_review_result(result, pack)
            previous = self.latest_result_for(pack)
            if previous is not None:
                if previous.to_dict() != result.to_dict():
                    raise ValueError("此 snapshot_id 已有不同研判，拒绝覆盖原决策记录")
                return 0
        elif result.snapshot_id:
            raise ValueError("带 snapshot_id 的研判必须提供证据 pack 后才能写入日志")
        common = {
            "asof": result.asof,
            "backend": result.backend,
            "snapshot_id": result.snapshot_id,
            "available": available or {},
        }
        entries: list[dict[str, Any]] = []
        for extraction in result.extractions:
            entries.append({
                **common,
                "kind": "extraction",
                "symbol": extraction.symbol,
                "payload": extraction.model_dump(mode="json"),
            })
        for judgment in result.judgments:
            entries.append({
                **common,
                "kind": "judgment",
                "symbol": judgment.symbol,
                "stance": judgment.stance,
                "confidence": judgment.confidence,
                "invalidation": judgment.invalidation,
                "time_stop_days": judgment.time_stop_days,
                "formula_score": judgment.formula_score,
                "formula_rank": judgment.formula_rank,
                "cites": [claim.cite for claim in judgment.long_case + judgment.short_case],
                "payload": judgment.model_dump(mode="json"),
            })
        if result.regime is not None:
            entries.append({
                **common,
                "kind": "regime",
                "payload": result.regime.model_dump(mode="json"),
            })
        if pack is not None:
            entries.append({**common, "kind": "review_result", "payload": result.to_dict()})
        if entries:
            self._append_many(entries)
        return len(entries)


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
