"""外部研判任务包的导出与写回。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from zhiku.ai.schema import (
    PACK_SCHEMA,
    RESULT_SCHEMA,
    Extraction,
    Judgment,
    RegimeAdvice,
    ReviewResult,
)
from zhiku.review.checklists import run_checklists
from zhiku.screens.fundamentals import Fundamentals

PACK_CONSTRAINTS = (
    "只使用本包里的候选、截面、清单和新闻摘录。不要编造包外数字。"
    "输出 zhiku.review_result.v1。stance 只能是 研究跟进/观察/回避。"
    "每条主张必须有 cite。详见 docs/ai_layer.md。"
)


def build_review_pack(
    *,
    asof: date,
    candidates: list[dict],
    rows: list[Fundamentals],
    masters: list[str],
    news: dict[str, Any] | None = None,
    macro: dict[str, Any] | None = None,
    trailing: dict[str, Any] | None = None,
    judge_n: int = 10,
) -> dict[str, Any]:
    by = {r.symbol: r for r in rows}
    top = candidates[:judge_n]
    packed_rows = []
    for i, cand in enumerate(top, start=1):
        row = by.get(cand.get("symbol"))
        if row is None:
            continue
        cards, conflicts = run_checklists(row, masters)
        snap = row.model_dump(mode="json")
        # 不输出任何可能被误当成持仓的键
        snap.pop("fx_to_cny", None)
        packed_rows.append(
            {
                "rank": i,
                "candidate": {
                    "symbol": cand.get("symbol"),
                    "name": cand.get("name"),
                    "formula_score": cand.get("formula_score"),
                    "qlib_score": cand.get("qlib_score"),
                    "screens": cand.get("screens") or cand.get("screens_all") or {},
                },
                "snapshot": {
                    k: snap.get(k)
                    for k in (
                        "symbol",
                        "name",
                        "pe",
                        "pb",
                        "roe",
                        "ocf",
                        "debt_asset",
                        "momentum_12_1",
                        "dist_to_high",
                        "eps_growth",
                        "gross_margin",
                        "ncav",
                    )
                },
                "checklists": [
                    {
                        "master": c.master,
                        "stance": c.stance,
                        "school": c.school,
                        "items": [it.to_dict() for it in c.items],
                    }
                    for c in cards
                ],
                "disagreements": conflicts,
            }
        )
    excerpts = []
    for cluster in (news or {}).get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        excerpts.append(
            {
                "symbol": cluster.get("symbol"),
                "headline": cluster.get("headline"),
                "summary": (cluster.get("summary") or "")[:500],
                "source": cluster.get("source") or cluster.get("source_kind"),
                "published": cluster.get("published"),
            }
        )
        if len(excerpts) >= 20:
            break
    return {
        "schema": PACK_SCHEMA,
        "asof": asof.isoformat() if hasattr(asof, "isoformat") else str(asof),
        "constraints": PACK_CONSTRAINTS,
        "candidates": packed_rows,
        "news_excerpts": excerpts,
        "macro": {
            "geo": (macro or {}).get("geo"),
            "rate_expectation": (macro or {}).get("rate_expectation"),
            "window_banners": (macro or {}).get("window_banners"),
        },
        "trailing": trailing or {},
    }


def write_review_pack(path: Path, pack: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_review_result(path: Path) -> ReviewResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("研判结果必须是 JSON 对象")
    schema = raw.get("schema") or raw.get("schema_name")
    if schema != RESULT_SCHEMA:
        raise ValueError(f"schema 应为 {RESULT_SCHEMA}，实际 {schema}")
    raw = dict(raw)
    for item in raw.get("judgments") or []:
        if isinstance(item, dict):
            if "long" in item and "long_case" not in item:
                item["long_case"] = item["long"]
            if "short" in item and "short_case" not in item:
                item["short_case"] = item["short"]
    result = ReviewResult.model_validate(raw)
    for j in result.judgments:
        if not j.long_case and not j.short_case:
            raise ValueError(f"{j.symbol} 缺少带 cite 的多空主张")
    return result


def result_from_parts(
    *,
    asof: str,
    backend: str,
    extractions: list[Extraction],
    judgments: list[Judgment],
    regime: RegimeAdvice | None,
    notes: list[str] | None = None,
) -> ReviewResult:
    return ReviewResult(
        schema_name=RESULT_SCHEMA,
        asof=asof,
        backend=backend,
        extractions=extractions,
        judgments=judgments,
        regime=regime,
        notes=notes or [],
    )
