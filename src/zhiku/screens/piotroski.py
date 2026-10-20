"""Piotroski F-Score（9 项）。

来源：Piotroski, J. (2000). Value Investing: The Use of Historical Financial
Statement Information to Separate Winners from Losers. 公开学术方法。
缺某一项则该项不得分，并记入 missing，不拿同行均值填。
"""

from __future__ import annotations

from zhiku.pipeline.settings import PiotroskiSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import ScreenRow


def _flag(ok: bool | None) -> int | None:
    if ok is None:
        return None
    return 1 if ok else 0


def fscore_items(row: Fundamentals) -> dict[str, int | None]:
    roa_pos = None if row.roa is None else row.roa > 0
    ocf_pos = None if row.ocf is None else row.ocf > 0
    delta_roa = None
    if row.roa is not None and row.roa_prev is not None:
        delta_roa = row.roa > row.roa_prev
    accrual = None
    if row.ocf is not None and row.net_income is not None:
        accrual = row.ocf > row.net_income
    leverage = None
    if row.lt_debt_asset is not None and row.lt_debt_asset_prev is not None:
        leverage = row.lt_debt_asset < row.lt_debt_asset_prev
    current = None
    if row.current_ratio is not None and row.current_ratio_prev is not None:
        current = row.current_ratio > row.current_ratio_prev
    no_issue = None if row.shares_issued is None else (not row.shares_issued)
    gm = None
    if row.gross_margin is not None and row.gross_margin_prev is not None:
        gm = row.gross_margin > row.gross_margin_prev
    ato = None
    if row.asset_turnover is not None and row.asset_turnover_prev is not None:
        ato = row.asset_turnover > row.asset_turnover_prev
    return {
        "roa_positive": _flag(roa_pos),
        "ocf_positive": _flag(ocf_pos),
        "delta_roa": _flag(delta_roa),
        "accrual_cfo_gt_ni": _flag(accrual),
        "delta_leverage": _flag(leverage),
        "delta_current_ratio": _flag(current),
        "no_new_shares": _flag(no_issue),
        "delta_gross_margin": _flag(gm),
        "delta_asset_turnover": _flag(ato),
    }


def fscore(row: Fundamentals) -> tuple[int, dict[str, int | None], list[str]]:
    items = fscore_items(row)
    missing = [k for k, v in items.items() if v is None]
    score = int(sum(v for v in items.values() if v))
    return score, items, missing


def screen_piotroski(rows: list[Fundamentals], settings: PiotroskiSettings) -> list[ScreenRow]:
    out: list[ScreenRow] = []
    for row in rows:
        raw, items, missing = fscore(row)
        if raw < settings.min_score:
            continue
        breakdown: dict[str, float | None] = {k: (None if v is None else float(v)) for k, v in items.items()}
        breakdown["fscore"] = float(raw)
        out.append(
            ScreenRow(
                symbol=row.symbol,
                name=row.name,
                score=100.0 * raw / 9.0,
                breakdown=breakdown,
                missing=missing,
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: settings.top_n]
