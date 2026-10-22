"""可选仓位：半凯利、单票上限、利弗莫尔止损、热市减仓、连续落后认错。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from zhiku.backtest.metrics import monthly_equal_weight, summarize_nav


def half_kelly(win_rate: float, payoff: float) -> float:
    """f = 1/2 * (p - (1-p)/b)。p 为胜率，b 为均盈/均亏。"""
    if payoff <= 0 or not (0 < win_rate < 1):
        return 0.0
    full = win_rate - (1.0 - win_rate) / payoff
    return float(max(0.0, min(1.0, 0.5 * full)))


def estimate_edge(monthly_rets: pd.Series) -> tuple[float, float]:
    r = monthly_rets.dropna().astype(float)
    if len(r) < 4:
        return 0.5, 1.0
    wins = r[r > 0]
    loss = r[r < 0]
    p = float(len(wins) / len(r))
    if wins.empty or loss.empty:
        return p if wins.empty is False else 0.5, 1.0
    payoff = float(wins.mean() / abs(loss.mean()))
    return p, max(payoff, 1e-6)


@dataclass
class SizingResult:
    name: str
    annualized: float | None
    sharpe: float | None
    max_drawdown: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "annualized": self.annualized,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "notes": self.notes,
        }


def _nav_of(series: pd.Series, name: str, notes: list[str]) -> SizingResult:
    if series is None or len(series) < 2:
        return SizingResult(name, None, None, None, notes + ["净值过短"])
    res = summarize_nav(series, rf=0.02, periods_per_year=12.0, screen=name)
    return SizingResult(name, res.annualized, res.sharpe, res.max_drawdown, notes)


def compare_sizing(
    prices: pd.DataFrame,
    picks: list[str],
    *,
    membership: dict | None = None,
    name_cap: float = 0.10,
    stop_pct: float = 0.08,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> dict[str, SizingResult]:
    """等权满仓 vs 半凯利+单票上限 vs 再加止损。p/b 只用截至上月的收益，不用未来。"""
    cols = [c for c in picks if c in prices.columns]
    notes = []
    if len(cols) < 2:
        return {"equal_weight": SizingResult("equal_weight", None, None, None, ["可交易标的不足"])}
    ew_nav, _ = monthly_equal_weight(
        prices, cols, cost_bps=cost_bps, slippage_bps=slippage_bps, membership=membership
    )
    out = {"equal_weight": _nav_of(ew_nav, "equal_weight", ["满仓等权"])}
    month_ends = prices.resample("ME").last().dropna(how="all")
    if month_ends.empty or len(month_ends) < 6:
        notes.append("月份不足，只报等权")
        return out

    # 半凯利：用已实现等权月收益估计 edge，再把总仓位缩到 min(半凯利, n*name_cap)
    ew_ret = ew_nav.pct_change().dropna()
    kelly_nav = [1.0]
    stop_nav = [1.0]
    prev_dt = None
    prev_w = pd.Series(0.0, index=cols)
    for i, (dt, row) in enumerate(month_ends.iterrows()):
        hist = ew_ret[ew_ret.index < dt]
        p, b = estimate_edge(hist)
        frac = half_kelly(p, b)
        n = max(1, len(cols))
        w_each = min(name_cap, frac / n if frac > 0 else 1.0 / n)
        if frac == 0:
            w_each = min(name_cap, 1.0 / n)
        # 热市减仓：过去 12 个月等权收益 > 40% 则仓位减半
        trail = hist.iloc[-12:] if len(hist) >= 12 else hist
        if len(trail) >= 6 and float((1 + trail).prod() - 1) > 0.40:
            w_each *= 0.5
            notes.append(f"{dt.date()} 热市减仓")
        # 认错：近两月落后 0（这里用自身连续负收益）
        if len(hist) >= 2 and float(hist.iloc[-1]) < 0 and float(hist.iloc[-2]) < 0:
            w_each *= 0.5
            notes.append(f"{dt.date()} 连续两月为负，减仓")
        w = pd.Series(0.0, index=cols)
        avail = [c for c in cols if pd.notna(row.get(c))]
        if avail:
            for c in avail:
                w[c] = w_each
        if i > 0 and prev_dt is not None:
            prev_row = month_ends.loc[prev_dt]
            ret = 0.0
            stop_ret = 0.0
            for c in cols:
                p0, p1 = prev_row.get(c), row.get(c)
                if pd.isna(p0) or pd.isna(p1) or float(p0) == 0:
                    continue
                r = float(p1) / float(p0) - 1.0
                ret += float(prev_w.get(c, 0) or 0) * r
                if r <= -abs(stop_pct):
                    stop_ret += float(prev_w.get(c, 0) or 0) * (-abs(stop_pct))
                else:
                    stop_ret += float(prev_w.get(c, 0) or 0) * r
            cash = max(0.0, 1.0 - float(prev_w.sum()))
            kelly_nav.append(kelly_nav[-1] * (1.0 + ret))
            stop_nav.append(stop_nav[-1] * (1.0 + stop_ret))
            del cash
        prev_w = w
        prev_dt = dt
    idx = month_ends.index[: len(kelly_nav)]
    out["half_kelly_cap"] = _nav_of(pd.Series(kelly_nav, index=idx), "half_kelly_cap", notes[-6:] + [f"单票上限 {name_cap:.0%}"])
    out["half_kelly_stop"] = _nav_of(pd.Series(stop_nav, index=idx), "half_kelly_stop", [f"止损 {stop_pct:.0%}"] + notes[-3:])
    return out
