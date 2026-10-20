"""优质门槛过滤器。来源：常见 A 股基本面质地清单（ROE/现金流/负债/流动性/ST），阈值可配。"""

from __future__ import annotations

from dataclasses import dataclass, field

from zhiku.pipeline.settings import QualitySettings, UniverseSettings
from zhiku.screens.fundamentals import Fundamentals


@dataclass
class QualityResult:
    symbol: str
    passed: bool
    reasons: list[str] = field(default_factory=list)


def apply_quality(
    row: Fundamentals,
    quality: QualitySettings,
    universe: UniverseSettings,
) -> QualityResult:
    reasons: list[str] = []
    if universe.exclude_etf and row.instrument == "etf":
        return QualityResult(row.symbol, False, ["ETF/基金不参与个股选股"])
    if universe.exclude_st and (row.st or row.delist_risk):
        return QualityResult(row.symbol, False, ["ST 或退市风险"])

    mktcap_yi = None if row.mktcap_cny is None else row.mktcap_cny / 1e8
    if mktcap_yi is None:
        reasons.append("缺数据：市值")
    elif mktcap_yi < universe.min_mktcap_yi:
        reasons.append(f"小市值 {mktcap_yi:.1f} 亿 < {universe.min_mktcap_yi}")

    amount = row.amount_cny
    min_amt = universe.min_amount(row.market)
    if amount is None:
        reasons.append("缺数据：日均成交额")
    elif amount < min_amt:
        reasons.append(f"日均成交额 {amount:.0f} < {min_amt:.0f}")

    if quality.require_leader_or_roe:
        mean_roe = row.roe_3y_mean
        leader_ok = row.industry_leader
        roe_ok = mean_roe is not None and mean_roe >= quality.roe_3y_min
        if not leader_ok and not roe_ok:
            if mean_roe is None:
                reasons.append("缺数据：ROE 且非行业龙头")
            else:
                reasons.append(f"非龙头且 ROE3Y {mean_roe:.1%} < {quality.roe_3y_min:.1%}")

    if quality.ocf_positive:
        if row.ocf is None:
            reasons.append("缺数据：经营现金流")
        elif row.ocf <= 0:
            reasons.append("经营现金流不为正")

    if row.debt_asset is None:
        reasons.append("缺数据：资产负债率")
    elif row.debt_asset > quality.max_debt_asset:
        reasons.append(f"负债率 {row.debt_asset:.1%} > {quality.max_debt_asset:.1%}")

    if row.pe_percentile is None:
        reasons.append("缺数据：PE 分位")
    elif row.pe_percentile > quality.pe_percentile_max:
        reasons.append(f"PE 分位 {row.pe_percentile:.2f} 处于历史高位")

    if row.pb_percentile is None:
        reasons.append("缺数据：PB 分位")
    elif row.pb_percentile > quality.pb_percentile_max:
        reasons.append(f"PB 分位 {row.pb_percentile:.2f} 处于历史高位")

    return QualityResult(row.symbol, passed=not reasons, reasons=reasons)


def filter_universe(
    rows: list[Fundamentals],
    quality: QualitySettings,
    universe: UniverseSettings,
) -> tuple[list[Fundamentals], list[QualityResult]]:
    passed: list[Fundamentals] = []
    results: list[QualityResult] = []
    for row in rows:
        q = apply_quality(row, quality, universe)
        results.append(q)
        if q.passed:
            passed.append(row)
    return passed, results
