"""Review one saved evidence snapshot once; replay only its matching decision."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from zhiku.ai.backend import ExternalBackend, resolve_backend
from zhiku.ai.journal import DecisionJournal
from zhiku.ai.judge import JUDGE_SYS
from zhiku.ai.pack import (
    build_review_pack,
    result_from_parts,
    review_result_from_obj,
    save_review_snapshot,
    validate_review_result,
)
from zhiku.ai.schema import Judgment, ReviewResult
from zhiku.ai.secrets import redact_secret
from zhiku.pipeline.settings import PipelineSettings
from zhiku.screens.fundamentals import Fundamentals


def _review_candidates(engine, pack: dict[str, Any], notes: list[str]) -> ReviewResult:
    """Compatibility fallback: one call per candidate, only from frozen pack evidence."""
    judgments: list[Judgment] = []
    for item in pack["candidates"]:
        candidate = item["candidate"]
        symbol = candidate["symbol"]
        evidence = {
            "snapshot_id": pack["snapshot_id"],
            "asof": pack["asof"],
            "constraints": pack["constraints"],
            **item,
            "news_excerpts": [
                excerpt for excerpt in pack["news_excerpts"]
                if not excerpt.get("symbol") or excerpt.get("symbol") == symbol
            ],
            "macro": pack["macro"],
            "trailing": pack["trailing"],
        }
        try:
            raw = engine.complete_json(JUDGE_SYS, json.dumps(evidence, ensure_ascii=False))
            # Identity and formula metadata come from the snapshot, not the model.
            judgment = Judgment.model_validate({
                **raw,
                "symbol": symbol,
                "name": candidate.get("name") or "",
                "formula_rank": item["rank"],
                "formula_score": candidate.get("formula_score"),
                "backend": engine.name,
                "long_case": raw.get("long_case") or raw.get("long") or [],
                "short_case": raw.get("short_case") or raw.get("short") or [],
            })
            checked = validate_review_result(
                result_from_parts(
                    asof=pack["asof"], snapshot_id=pack["snapshot_id"], backend=engine.name,
                    extractions=[], judgments=[judgment], regime=None,
                ),
                pack,
            )
            judgments.extend(checked.judgments)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{symbol} 研判失败：{redact_secret(str(exc))}")
    return result_from_parts(
        asof=pack["asof"], snapshot_id=pack["snapshot_id"], backend=engine.name,
        extractions=[], judgments=judgments, regime=None, notes=notes,
    )


def run_ai_layer(
    *,
    asof: date,
    paths,
    settings: PipelineSettings,
    candidates: list[dict],
    rows: list[Fundamentals],
    news: dict[str, Any] | None,
    macro: dict[str, Any] | None,
    trailing: dict[str, Any] | None = None,
    backend=None,
) -> tuple[ReviewResult | None, Path | None, list[str]]:
    notes: list[str] = []
    pack = build_review_pack(
        asof=asof, candidates=candidates, rows=rows, masters=settings.ai.masters,
        news=news, macro=macro, trailing=trailing,
        judge_n=int(getattr(settings.ai, "judge_n", 10)),
    )
    pack_path = save_review_snapshot(Path(paths.reports), pack)
    journal = DecisionJournal(Path(paths.journal) / "ai_decisions.jsonl")
    previous = journal.latest_result_for(pack)
    if previous is not None:
        notes.append(f"复用证据快照 {pack['snapshot_id'][:12]} 的已验证研判；未重复调用 AI。")
        notes.extend(note for note in previous.notes if note not in notes)
        return previous, pack_path, notes
    engine = backend or resolve_backend(getattr(settings.ai, "backend", None))
    if isinstance(engine, ExternalBackend) or engine.name == "external":
        notes.append(f"后端=external：已保存研判快照 {pack_path.name}，用 zk export-review 导出、ingest-review 写回。")
        journal.record_pack_export(pack, pack_path)
        return None, pack_path, notes
    if not pack["candidates"]:
        notes.append("快照没有可研判候选，未调用 AI。")
        return None, pack_path, notes
    notes.append(f"AI 研判后端={engine.name}；只使用已保存的同一证据快照。")
    try:
        complete_review_pack = getattr(engine, "complete_review_pack", None)
        if callable(complete_review_pack):
            # Never let a backend mutate the authoritative saved evidence in memory.
            raw = complete_review_pack(deepcopy(pack))
            result = review_result_from_obj(raw) if isinstance(raw, dict) else raw
        else:
            result = _review_candidates(engine, pack, notes)
        result = validate_review_result(result, pack)
        if not result.judgments:
            notes.append("没有有效候选研判，不装成已判断；保留快照供重试。")
            return None, pack_path, notes
        journal.append_result(
            result, pack=pack,
            available={"n_candidates": len(pack["candidates"]), "n_news": len(pack["news_excerpts"])},
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"AI 打包研判失败，不装成已判断：{redact_secret(str(exc))}")
        return None, pack_path, notes
    notes.extend(note for note in result.notes if note not in notes)
    return result, pack_path, notes
