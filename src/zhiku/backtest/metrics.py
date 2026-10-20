"""回测指标。纯 pandas 实现，vectorbt 作为可选加速/对照。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    screen: str
    annualized: float | None
    max_drawdown: float | None
    sharpe: float | None
    win_rate: float | None
    turnover: float | None
    years: dict[str, float] = field(default_factory=dict)
    equity: list[tuple[str, float]] = field(default_factory=list)
    benchmark_annualized: float | None = None
    beat_benchmark: bool | None = None
    engine: str = "pandas"
    notes: list[str] = field(default_factory=list)
    passed_gate: bool = True

    def to_dict(self) -> dict:
        return {
            "screen": self.screen,
            "annualized": self.annualized,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "turnover": self.turnover,
            "years": self.years,
            "equity": self.equity,
            "benchmark_annualized": self.benchmark_annualized,
            "beat_benchmark": self.beat_benchmark,
            "engine": self.engine,
            "notes": self.notes,
            "passed_gate": self.passed_gate,
        }


def _ann_return(nav: pd.Series) -> float | None:
    if len(nav) < 2 or float(nav.iloc[0]) == 0:
        return None
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    return float(nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1


def _max_dd(nav: pd.Series) -> float | None:
    if nav.empty:
        return None
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _sharpe(returns: pd.Series, rf: float, periods_per_year: float) -> float | None:
    if returns.empty or returns.std(ddof=0) == 0:
        return None
    excess = returns.mean() * periods_per_year - rf
    vol = float(returns.std(ddof=0)) * np.sqrt(periods_per_year)
    if vol == 0:
        return None
    return float(excess / vol)


def summarize_nav(
    nav: pd.Series,
    *,
    returns: pd.Series | None = None,
    turnover: float | None = None,
    rf: float = 0.02,
    periods_per_year: float = 12.0,
    benchmark: pd.Series | None = None,
    screen: str = "",
    engine: str = "pandas",
) -> BacktestResult:
    nav = nav.dropna().astype(float)
    nav = nav[nav > 0]
    if returns is None and len(nav) >= 2:
        returns = nav.pct_change().dropna()
    returns = returns if returns is not None else pd.Series(dtype=float)
    years: dict[str, float] = {}
    if not nav.empty:
        grouped = nav.groupby(nav.index.year)
        for year, series in grouped:
            if len(series) >= 2 and float(series.iloc[0]) != 0:
                years[str(year)] = float(series.iloc[-1] / series.iloc[0] - 1)
    bench_ann = _ann_return(benchmark.dropna()) if benchmark is not None else None
    ann = _ann_return(nav)
    beat = None
    if ann is not None and bench_ann is not None:
        beat = ann >= bench_ann
    equity = [(idx.date().isoformat(), float(val)) for idx, val in nav.items()]
    win = None
    if not returns.empty:
        win = float((returns > 0).mean())
    return BacktestResult(
        screen=screen,
        annualized=ann,
        max_drawdown=_max_dd(nav),
        sharpe=_sharpe(returns, rf, periods_per_year),
        win_rate=win,
        turnover=turnover,
        years=years,
        equity=equity,
        benchmark_annualized=bench_ann,
        beat_benchmark=beat,
        engine=engine,
    )


def monthly_equal_weight(
    prices: pd.DataFrame,
    picks: list[str],
    *,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> tuple[pd.Series, float]:
    """每月初等权再平衡 ``picks``。成本/滑点按换手从净值扣除。"""
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    cols = [c for c in picks if c in px.columns]
    if not cols:
        return pd.Series(dtype=float), 0.0
    month_ends = px.resample("ME").last().dropna(how="all")
    if month_ends.empty:
        month_ends = px.resample("M").last().dropna(how="all")
    nav_pts = []
    turnovers = []
    prev_w = pd.Series(0.0, index=cols)
    nav = 1.0
    cost = (cost_bps + slippage_bps) / 10000.0
    for i, (dt, row) in enumerate(month_ends.iterrows()):
        avail = [c for c in cols if pd.notna(row.get(c))]
        if not avail:
            nav_pts.append((dt, nav))
            continue
        w = pd.Series(1.0 / len(avail), index=avail)
        tw = 0.5 * float((w.reindex(cols).fillna(0) - prev_w.reindex(cols).fillna(0)).abs().sum())
        turnovers.append(tw)
        nav *= 1.0 - tw * cost
        if i > 0:
            prev_dt = month_ends.index[i - 1]
            prev_row = month_ends.loc[prev_dt]
            ret = 0.0
            for c in avail:
                p0, p1 = prev_row.get(c), row.get(c)
                if pd.notna(p0) and pd.notna(p1) and float(p0) != 0:
                    ret += (1.0 / len(avail)) * (float(p1) / float(p0) - 1.0)
            nav *= 1.0 + ret
        prev_w = w.reindex(cols).fillna(0)
        nav_pts.append((dt, nav))
    series = pd.Series({d: v for d, v in nav_pts})
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0
    return series, avg_to
