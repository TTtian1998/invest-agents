"""低估值高股息：PE/PB 分位 + 股息率 + 分红连续性。"""

from __future__ import annotations

from zhiku.pipeline.settings import DividendSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import ScreenRow
from zhiku.screens.ranking import minmax_from_ranks, rank_higher_better, rank_lower_better


def screen_dividend(rows: list[Fundamentals], settings: DividendSettings) -> list[ScreenRow]:
    eligible: list[Fundamentals] = []
    for row in rows:
        if row.pe_percentile is None or row.pb_percentile is None:
            continue
        if row.dividend_yield is None or row.dividend_years is None:
            continue
        if row.pe_percentile > settings.pe_percentile_max:
            continue
        if row.pb_percentile > settings.pb_percentile_max:
            continue
        if row.dividend_yield < settings.min_yield:
            continue
        if row.dividend_years < settings.min_consecutive_years:
            continue
        eligible.append(row)
    if not eligible:
        return []
    pe_r = rank_lower_better({r.symbol: r.pe_percentile or 0 for r in eligible})
    pb_r = rank_lower_better({r.symbol: r.pb_percentile or 0 for r in eligible})
    y_r = rank_higher_better({r.symbol: r.dividend_yield or 0 for r in eligible})
    c_r = rank_higher_better({r.symbol: float(r.dividend_years or 0) for r in eligible})
    combined = {
        r.symbol: pe_r[r.symbol] + pb_r[r.symbol] + y_r[r.symbol] + c_r[r.symbol] for r in eligible
    }
    from zhiku.screens.ranking import rank_lower_better as rlb

    scores = minmax_from_ranks(rlb(combined))
    out = [
        ScreenRow(
            symbol=r.symbol,
            name=r.name,
            score=scores[r.symbol],
            breakdown={
                "pe_percentile": r.pe_percentile,
                "pb_percentile": r.pb_percentile,
                "dividend_yield": r.dividend_yield,
                "dividend_years": float(r.dividend_years or 0),
            },
        )
        for r in eligible
    ]
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: settings.top_n]
