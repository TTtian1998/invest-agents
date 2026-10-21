"""前 10 名：规则清单之上做多空辩论，输出研究跟进/观察/回避。"""

from __future__ import annotations

from typing import Any

from zhiku.ai.backend import LLMBackend
from zhiku.ai.schema import CitedClaim, Judgment
from zhiku.review.checklists import run_checklists
from zhiku.screens.fundamentals import Fundamentals

JUDGE_SYS = (
    "任务=研判辩论。公式只做初筛。你根据已给出的规则清单和原文摘录做多空辩论，"
    "只输出 JSON：stance (研究跟进|观察|回避), confidence (0-1), invalidation, time_stop_days, "
    "long ([{claim,cite}]), short ([{claim,cite}])。"
    "每条 claim 必须 cite 所给字段名或原文短句。不要改数字，不要说买入。"
)


def _pack_context(row: Fundamentals, checklists: list[dict], news_excerpts: list[dict]) -> str:
    snap = {
        "symbol": row.symbol,
        "name": row.name,
        "pe": row.pe,
        "pb": row.pb,
        "roe": row.roe,
        "ocf": row.ocf,
        "debt_asset": row.debt_asset,
        "momentum_12_1": row.momentum_12_1,
        "dist_to_high": row.dist_to_high,
        "eps_growth": row.eps_growth,
    }
    return (
        f"截面={snap}\n清单={checklists}\n新闻摘录={news_excerpts[:4]}"
    )


def judge_symbol(
    backend: LLMBackend,
    row: Fundamentals,
    *,
    masters: list[str],
    news_excerpts: list[dict] | None = None,
    formula_score: float | None = None,
    formula_rank: int | None = None,
) -> Judgment:
    cards, conflicts = run_checklists(row, masters)
    checks = [
        {
            "master": c.master,
            "stance": c.stance,
            "school": c.school,
            "items": [i.to_dict() for i in c.items],
        }
        for c in cards
    ]
    user = _pack_context(row, checks, news_excerpts or [])
    if conflicts:
        user += f"\n大师分歧={conflicts}"
    raw = backend.complete_json(JUDGE_SYS, user)
    stance = raw.get("stance") or "观察"
    if stance not in {"研究跟进", "观察", "回避"}:
        stance = "观察"
    long_case = [CitedClaim.model_validate(x) for x in (raw.get("long") or raw.get("long_case") or [])]
    short_case = [CitedClaim.model_validate(x) for x in (raw.get("short") or raw.get("short_case") or [])]
    if not long_case and not short_case:
        raise ValueError("辩论结果缺少带 cite 的多空主张")
    return Judgment(
        symbol=row.symbol,
        name=row.name,
        stance=stance,
        confidence=raw.get("confidence") or 0.0,
        invalidation=str(raw.get("invalidation") or "缺数据"),
        time_stop_days=int(raw.get("time_stop_days") or 20),
        long_case=long_case,
        short_case=short_case,
        formula_score=formula_score,
        formula_rank=formula_rank,
        backend=getattr(backend, "name", ""),
    )


def news_excerpts_for(symbol: str, news: dict[str, Any], *, limit: int = 3) -> list[dict]:
    out: list[dict] = []
    for cluster in (news or {}).get("clusters") or (news or {}).get("items") or []:
        if not isinstance(cluster, dict):
            continue
        if cluster.get("symbol") and cluster.get("symbol") != symbol:
            continue
        out.append(
            {
                "headline": cluster.get("headline"),
                "summary": (cluster.get("summary") or "")[:400],
                "source": cluster.get("source") or cluster.get("source_kind"),
            }
        )
        if len(out) >= limit:
            break
    return out
