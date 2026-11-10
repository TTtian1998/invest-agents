"""日报入口：有后端则判断，否则导出任务包。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from zhiku.ai.backend import ExternalBackend, resolve_backend
from zhiku.ai.extract import extract_news_bundle
from zhiku.ai.journal import DecisionJournal
from zhiku.ai.judge import judge_symbol, news_excerpts_for
from zhiku.ai.pack import build_review_pack, result_from_parts, write_review_pack
from zhiku.ai.regime import advise_regime
from zhiku.ai.schema import Judgment, ReviewResult
from zhiku.ai.secrets import redact_secret
from zhiku.pipeline.settings import PipelineSettings
from zhiku.screens.fundamentals import Fundamentals


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
        asof=asof,
        candidates=candidates,
        rows=rows,
        masters=settings.ai.masters,
        news=news,
        macro=macro,
        trailing=trailing,
        judge_n=int(getattr(settings.ai, "judge_n", 10) or 10),
    )
    pack_path = Path(paths.reports) / f"{asof.isoformat()}-review-pack.json"
    write_review_pack(pack_path, pack)
    engine = backend or resolve_backend(getattr(settings.ai, "backend", None))
    if isinstance(engine, ExternalBackend) or engine.name == "external":
        notes.append(f"无 LLM 或后端=external：已导出研判任务包 {pack_path.name}，用 zk ingest-review 写回。")
        journal = DecisionJournal(Path(paths.journal) / "ai_decisions.jsonl")
        journal.append(
            {
                "kind": "pack_exported",
                "asof": asof.isoformat(),
                "backend": "external",
                "path": str(pack_path),
                "n_candidates": len(pack.get("candidates") or []),
                "available": {"macro": bool(macro), "news": bool(news)},
            }
        )
        return None, pack_path, notes

    notes.append(f"AI 研判后端={engine.name}")
    complete_review_pack = getattr(engine, "complete_review_pack", None)
    if callable(complete_review_pack):
        try:
            result = complete_review_pack(pack)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Cursor/打包研判失败，不装成已判断：{redact_secret(str(exc))}")
            return None, pack_path, notes
        journal = DecisionJournal(Path(paths.journal) / "ai_decisions.jsonl")
        journal.append_result(
            result,
            available={"n_candidates": len(candidates), "n_news": len((news or {}).get("clusters") or [])},
        )
        return result, pack_path, notes
    extractions = []
    try:
        extractions = extract_news_bundle(engine, news or {})
    except Exception as exc:  # noqa: BLE001
        notes.append(f"文本抽取跳过：{redact_secret(str(exc))}")
    by = {r.symbol: r for r in rows}
    judgments: list[Judgment] = []
    judge_n = int(getattr(settings.ai, "judge_n", 10) or 10)
    for i, cand in enumerate(candidates[:judge_n], start=1):
        symbol = cand.get("symbol")
        row = by.get(symbol) if symbol is not None else None
        if row is None:
            continue
        try:
            judgments.append(
                judge_symbol(
                    engine,
                    row,
                    masters=settings.ai.masters,
                    news_excerpts=news_excerpts_for(row.symbol, news or {}),
                    formula_score=cand.get("formula_score"),
                    formula_rank=i,
                )
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{row.symbol} 研判失败：{redact_secret(str(exc))}")
    regime = None
    try:
        regime = advise_regime(engine, macro=macro, trailing=trailing)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"市场状态跳过：{redact_secret(str(exc))}")
    result = result_from_parts(
        asof=asof.isoformat(),
        backend=engine.name,
        extractions=extractions,
        judgments=judgments,
        regime=regime,
        notes=notes,
    )
    journal = DecisionJournal(Path(paths.journal) / "ai_decisions.jsonl")
    journal.append_result(
        result,
        available={"n_candidates": len(candidates), "n_news": len((news or {}).get("clusters") or [])},
    )
    return result, pack_path, notes
