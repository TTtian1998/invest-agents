"""Deterministic, long-only, whole-lot next-open portfolio accounting.

Targets are decided after the supplied date's close. They execute once at the
next provided session, never at that close and never retried silently. The latest
signal wins if several signals map to one session. Sells precede buys; buys are
allocated in sorted symbol order, capped by cash including costs. Thus there are
no same-day round trips (conservative T+1), even when decisions collide.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd

from zhiku.backtest.metrics import summarize_nav


@dataclass
class TargetBacktestResult:
    nav: pd.Series
    cash: pd.Series
    holdings: pd.DataFrame
    trades: list[dict]
    rejections: list[dict]
    turnover: float
    metrics: dict

    def as_dict(self) -> dict:
        def series(s):
            return {d.date().isoformat(): float(v) for d, v in s.items()}
        return {
            "nav": series(self.nav), "cash": series(self.cash),
            "holdings": {d.date().isoformat(): {s: int(q) for s, q in row.items()}
                         for d, row in self.holdings.iterrows()},
            "trades": self.trades, "rejections": self.rejections,
            "turnover": self.turnover, "metrics": self.metrics,
            "fill": "next_open", "allocation": "sell_then_sorted_symbol_buy",
            "turnover_definition": "sum_gross_executed_notional_over_pretrade_equity",
        }

    def to_dict(self) -> dict:
        return self.as_dict()


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        raise ValueError("Use timezone-naive daily session dates")
    if not result.index.equals(result.index.normalize()):
        raise ValueError("Use daily session dates, not intraday timestamps")
    if result.index.has_duplicates or result.columns.has_duplicates:
        raise ValueError("Duplicate dates or symbols")
    return result.sort_index()


def _price(value) -> bool:
    return pd.notna(value) and math.isfinite(float(value)) and float(value) > 0


def backtest_targets(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    targets_by_date: dict[date, dict[str, float]],
    *,
    initial_cash: float = 100000.,
    cost_bps: float = 10.,
    slippage_bps: float = 5.,
    lot_size: int = 100,
    tradable: pd.DataFrame | None = None,
    limit_up: pd.DataFrame | None = None,
    limit_down: pd.DataFrame | None = None,
) -> TargetBacktestResult:
    """Return daily cash-valued NAV, quantities, cash and an execution audit.

    Closes define the session calendar and security universe. Target quantities
    use pre-trade opening equity and raw opens. A held missing open is valued
    at its *previous* close for sizing only; its order is skipped. No execution
    uses a close. Every held closing mark must be finite and positive or the run
    raises rather than carrying a stale mark. Supplied masks with missing cells
    are fail-closed; absent masks mean unrestricted. Fees apply to each side's
    slipped execution notional; slippage changes the execution price itself.
    Turnover sums both sides' raw notional / pre-trade equity (not half-turnover).
    """
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
        raise ValueError("lot_size must be a positive integer")
    if any(not math.isfinite(x) or x < 0 or x >= 10000 for x in (cost_bps, slippage_bps)):
        raise ValueError("cost_bps and slippage_bps must be finite in [0, 10000)")
    px, op = _panel(closes), _panel(opens)
    symbols = sorted(px.columns)
    op = op.reindex(index=px.index, columns=symbols)
    masks = [None if m is None else _panel(m).reindex(index=px.index, columns=symbols)
             for m in (tradable, limit_up, limit_down)]
    events: dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]] = {}
    rejects: list[dict] = []
    for key, weights in sorted(targets_by_date.items(), key=lambda kv: pd.Timestamp(kv[0])):
        signal = pd.Timestamp(key)
        if signal.tz is not None or signal != signal.normalize():
            raise ValueError("Signal dates must be timezone-naive daily dates")
        unknown = set(weights) - set(symbols)
        if unknown:
            raise ValueError(f"Unknown target symbols: {sorted(unknown)}")
        w = {s: float(v) for s, v in weights.items()}
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in w.values()) or sum(w.values()) > 1 + 1e-12:
            raise ValueError("Target weights must be finite, nonnegative and sum to at most 1")
        loc = px.index.searchsorted(signal, side="right")
        if loc == len(px):
            rejects.append({"signal_date": signal.date().isoformat(), "reason": "no_next_session"})
            continue
        session = px.index[loc]
        if session in events:
            rejects.append({"signal_date": events[session][0].date().isoformat(),
                            "date": session.date().isoformat(), "reason": "superseded_signal"})
        events[session] = (signal, w)
    cash = float(initial_cash)
    qty = dict.fromkeys(symbols, 0)
    last_close: dict[str, float] = {}
    navs, cashes, positions, trades = [], [], [], []
    turnover = 0.
    fee, slip = cost_bps / 10000, slippage_bps / 10000
    for session, close_row in px.iterrows():
        if session in events:
            signal, weights = events[session]
            row = op.loc[session]
            equity = cash + sum(q * (float(row[s]) if _price(row[s]) else last_close[s])
                                for s, q in qty.items() if q)
            desired = {}
            for s in symbols:
                if not _price(row[s]):
                    if qty[s] or weights.get(s, 0):
                        rejects.append({"date": session.date().isoformat(), "signal_date": signal.date().isoformat(),
                                        "symbol": s, "reason": "missing_open"})
                    continue
                desired[s] = math.floor(equity * weights.get(s, 0) / float(row[s]) / lot_size) * lot_size
            for side in ("sell", "buy"):
                for s in symbols:
                    delta = desired.get(s, qty[s]) - qty[s]
                    if (side == "sell" and delta >= 0) or (side == "buy" and delta <= 0):
                        continue
                    reason = None
                    for mask, blocked_value, label in ((masks[0], False, "not_tradable"),
                            (masks[1] if side == "buy" else masks[2], True,
                             "limit_up" if side == "buy" else "limit_down")):
                        if mask is not None:
                            value = mask.loc[session, s]
                            if pd.isna(value) or bool(value) == blocked_value:
                                reason = label
                                break
                    if reason:
                        rejects.append({"date": session.date().isoformat(), "signal_date": signal.date().isoformat(),
                                        "symbol": s, "side": side, "reason": reason})
                        continue
                    raw = float(row[s])
                    fill = raw * (1 + slip if side == "buy" else 1 - slip)
                    amount = abs(delta)
                    if side == "buy":
                        affordable = math.floor(cash / (fill * (1 + fee) * lot_size)) * lot_size
                        amount = min(amount, affordable)
                        if amount < abs(delta):
                            rejects.append({"date": session.date().isoformat(), "signal_date": signal.date().isoformat(),
                                            "symbol": s, "side": side, "reason": "insufficient_cash",
                                            "unfilled_quantity": abs(delta) - amount})
                    if not amount:
                        continue
                    notional = amount * fill
                    costs = notional * fee
                    cash += notional - costs if side == "sell" else -notional - costs
                    qty[s] += -amount if side == "sell" else amount
                    if cash < -1e-8:
                        raise AssertionError("Negative cash")
                    cash = max(0., cash)
                    turnover += amount * raw / equity
                    trades.append({"date": session.date().isoformat(), "signal_date": signal.date().isoformat(),
                                   "symbol": s, "side": side, "quantity": amount,
                                   "open_price": raw, "price": fill, "notional": notional,
                                   "fee": costs, "slippage": amount * abs(fill - raw), "cash_after": cash})
        for s, q in qty.items():
            if q and not _price(close_row[s]):
                raise ValueError(f"Missing valid held close for {s} on {session.date()}")
            if _price(close_row[s]):
                last_close[s] = float(close_row[s])
        navs.append(cash + sum(q * float(close_row[s]) for s, q in qty.items() if q))
        cashes.append(cash)
        positions.append(qty.copy())
    nav = pd.Series(navs, index=px.index, name="nav", dtype=float)
    cash_series = pd.Series(cashes, index=px.index, name="cash", dtype=float)
    holdings = pd.DataFrame(positions, index=px.index, columns=symbols).fillna(0).astype(int)
    # The first session may execute a pre-window signal. An explicit initial
    # cash baseline prevents its transaction costs from disappearing in metrics.
    # This synthetic point is used for metrics only, not the session NAV output.
    baseline_date = px.index[0] - pd.Timedelta(days=1) if len(px) else None
    metric_nav = (pd.concat([pd.Series([initial_cash], index=[baseline_date]), nav])
                  if baseline_date is not None else nav)
    metrics = summarize_nav(metric_nav, turnover=turnover, periods_per_year=252.,
                            engine="target_weights_next_open").to_dict()
    # Attribute each close-to-close return to its ending session's year,
    # including the return across New Year (the legacy summary omits it).
    daily_returns = metric_nav.pct_change().dropna()
    metrics["years"] = {
        str(year): float((1.0 + returns).prod() - 1.0)
        for year, returns in daily_returns.groupby(daily_returns.index.year)
    }
    metrics["metric_baseline_date"] = (baseline_date.date().isoformat()
                                        if baseline_date is not None else None)
    metrics.update(total_return=float(nav.iloc[-1] / initial_cash - 1) if len(nav) else None,
                   initial_cash=initial_cash, final_cash=cash, fill="next_open",
                   close_fallback_fills=0, passed_gate=False, gate_status="RESEARCH_ONLY")
    return TargetBacktestResult(nav, cash_series, holdings, trades, rejects, turnover, metrics)
