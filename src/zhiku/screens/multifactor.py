"""质量 + 动量多因子：ROE、毛利率稳定性、OC/NI、12-1 动量、低波。"""

from __future__ import annotations

from statistics import pstdev

from zhiku.pipeline.settings import MultifactorSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import ScreenRow
from zhiku.screens.ranking import minmax_from_ranks, rank_higher_better, rank_lower_better


def _gm_stability(row: Fundamentals) -> float | None:
    series = row.gross_margin_3y or ([row.gross_margin] if row.gross_margin is not None else [])
    series = [x for x in series if x is not None]
    if len(series) < 2:
        return None
    mean = sum(series) / len(series)
    if mean == 0:
        return None
    return 1.0 - min(1.0, pstdev(series) / abs(mean))


def _ocf_ni(row: Fundamentals) -> float | None:
    if row.ocf is None or row.net_income is None or row.net_income == 0:
        return None
    return row.ocf / row.net_income


def screen_multifactor(rows: list[Fundamentals], settings: MultifactorSettings) -> list[ScreenRow]:
    weights = settings.weights
    factor_values: dict[str, dict[str, float]] = {
        "roe": {},
        "gm_stability": {},
        "ocf_ni": {},
        "momentum_12_1": {},
        "low_vol": {},
    }
    for row in rows:
        if row.roe is not None:
            factor_values["roe"][row.symbol] = row.roe
        stab = _gm_stability(row)
        if stab is not None:
            factor_values["gm_stability"][row.symbol] = stab
        ocf_ni = _ocf_ni(row)
        if ocf_ni is not None:
            factor_values["ocf_ni"][row.symbol] = ocf_ni
        if row.momentum_12_1 is not None:
            factor_values["momentum_12_1"][row.symbol] = row.momentum_12_1
        if row.volatility is not None:
            factor_values["low_vol"][row.symbol] = row.volatility

    ranks: dict[str, dict[str, float]] = {
        "roe": rank_higher_better(factor_values["roe"]),
        "gm_stability": rank_higher_better(factor_values["gm_stability"]),
        "ocf_ni": rank_higher_better(factor_values["ocf_ni"]),
        "momentum_12_1": rank_higher_better(factor_values["momentum_12_1"]),
        "low_vol": rank_lower_better(factor_values["low_vol"]),
    }
    symbols = {r.symbol for r in rows}
    combined: dict[str, float] = {}
    for symbol in symbols:
        available = [name for name in ranks if symbol in ranks[name] and name in weights]
        if not available:
            continue
        wsum = sum(weights[name] for name in available)
        if wsum <= 0:
            continue
        combined[symbol] = sum(weights[name] * ranks[name][symbol] for name in available) / wsum

    from zhiku.screens.ranking import rank_lower_better as rlb

    scores = minmax_from_ranks(rlb(combined))
    by_id = {r.symbol: r for r in rows}
    out: list[ScreenRow] = []
    for symbol, score in scores.items():
        row = by_id[symbol]
        out.append(
            ScreenRow(
                symbol=symbol,
                name=row.name,
                score=score,
                breakdown={
                    "roe": row.roe,
                    "gm_stability": _gm_stability(row),
                    "ocf_ni": _ocf_ni(row),
                    "momentum_12_1": row.momentum_12_1,
                    "volatility": row.volatility,
                },
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: settings.top_n]
