"""Greenblatt 神奇公式：ROC + Earnings Yield 排名合成。

来源：Joel Greenblatt, *The Little Book That Beats the Market*（公开方法）。
ROC = EBIT / (净营运资本 + 净固定资产)；Earnings Yield = EBIT / EV。
实现参考了常见开源选股项目（如 myhhub/stock）对 A 股字段的映射思路，未复制其源码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zhiku.models import MissingData
from zhiku.pipeline.settings import GreenblattSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.ranking import drop_missing, minmax_from_ranks, rank_higher_better


def roc(row: Fundamentals) -> float | MissingData:
    if row.ebit is None:
        return MissingData("缺 EBIT", symbol=row.symbol, field="roc")
    nwc = 0.0 if row.nwc is None else row.nwc
    nfa = 0.0 if row.nfa is None else row.nfa
    denom = nwc + nfa
    if denom <= 0:
        return MissingData("净营运资本+净固定资产 ≤ 0，拒绝估算 ROC", symbol=row.symbol)
    return row.ebit / denom


def earnings_yield(row: Fundamentals) -> float | MissingData:
    if row.ebit is None or row.ev is None:
        return MissingData("缺 EBIT 或 EV", symbol=row.symbol, field="earnings_yield")
    if row.ev <= 0:
        return MissingData("EV ≤ 0", symbol=row.symbol)
    return row.ebit / row.ev


@dataclass
class ScreenRow:
    symbol: str
    name: str
    score: float
    breakdown: dict[str, float | None] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def screen_greenblatt(rows: list[Fundamentals], settings: GreenblattSettings) -> list[ScreenRow]:
    roc_map = {r.symbol: roc(r) for r in rows}
    ey_map = {r.symbol: earnings_yield(r) for r in rows}
    roc_ok = drop_missing(roc_map)
    ey_ok = drop_missing(ey_map)
    common = [s for s in roc_ok if s in ey_ok]
    if not common:
        return []
    roc_ranks = rank_higher_better({s: roc_ok[s] for s in common})
    ey_ranks = rank_higher_better({s: ey_ok[s] for s in common})
    combined = {s: roc_ranks[s] + ey_ranks[s] for s in common}
    # 合成排名越低越好
    from zhiku.screens.ranking import rank_lower_better

    scores = minmax_from_ranks(rank_lower_better(combined))
    by_id = {r.symbol: r for r in rows}
    out: list[ScreenRow] = []
    for symbol in common:
        row = by_id[symbol]
        out.append(
            ScreenRow(
                symbol=symbol,
                name=row.name,
                score=scores[symbol],
                breakdown={
                    "roc": roc_ok[symbol],
                    "earnings_yield": ey_ok[symbol],
                    "roc_rank": roc_ranks[symbol],
                    "ey_rank": ey_ranks[symbol],
                    "combined_rank": combined[symbol],
                },
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    out = [r for r in out if r.score >= settings.min_score]
    return out[: settings.top_n]
