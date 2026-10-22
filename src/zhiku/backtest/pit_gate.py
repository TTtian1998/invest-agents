"""时点选股闸门：调仓日 T 只用 T 前数据，T+1 开盘成交，半段超额 + deflated Sharpe。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from zhiku.backtest.metrics import (
    BacktestResult,
    rebalance_equal_weight,
    split_half_excess,
    summarize_nav,
    year_excesses,
)
from zhiku.data.compute import (
    attach_percentiles,
    consecutive_div_years,
    dividend_yield_from_rows,
    snapshot_from_store,
)
from zhiku.data.universe_live import _listing_map, _shares_issued
from zhiku.pipeline.settings import BacktestSettings, PipelineSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.runner import SCREEN_FUNCS, run_screens

_NORM = NormalDist()
_EULER = 0.5772156649015329


def expected_max_sr(n_trials: int, sr_std: float) -> float:
    """N 次独立试验下预期最大 Sharpe（Bailey & López de Prado 2014）。"""
    if n_trials <= 1 or sr_std <= 0:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _EULER) * z1 + _EULER * z2)


def sharpe_sampling_std(sharpe: float, n_obs: int, skew: float = 0.0, excess_kurt: float = 0.0) -> float:
    """σ(SR̂)。kurt 用超额峰度；正态时 excess_kurt=0。"""
    if n_obs <= 1:
        return float("nan")
    kurt = excess_kurt + 3.0
    var = (1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2) / (n_obs - 1)
    return math.sqrt(max(var, 1e-18))


def deflated_sharpe_stats(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    *,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> dict[str, float]:
    """扣除多重检验后的 Sharpe。haircut = SR − E[max SR]；dsr_prob = Φ((SR−SR0)/σ)。"""
    sigma = sharpe_sampling_std(sharpe, n_obs, skew, excess_kurt)
    sr0 = expected_max_sr(n_trials, sigma)
    z = 0.0 if not math.isfinite(sigma) or sigma == 0 else (sharpe - sr0) / sigma
    return {
        "sharpe": float(sharpe),
        "sr0": float(sr0),
        "deflated_sharpe": float(sharpe - sr0),
        "dsr_prob": float(_NORM.cdf(z)),
        "n_trials": float(n_trials),
        "n_obs": float(n_obs),
        "sr_std": float(sigma) if math.isfinite(sigma) else float("nan"),
    }


def month_end_trading_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if index.empty:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    return list(idx.to_series().groupby(idx.to_period("M")).max())


def shift_trading_days(index: pd.DatetimeIndex, dates: list[pd.Timestamp], n: int) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    out: list[pd.Timestamp] = []
    for dt in dates:
        loc = int(idx.searchsorted(pd.Timestamp(dt), side="left"))
        loc = min(max(loc + n, 0), len(idx) - 1)
        out.append(idx[loc])
    return out


def panel_momentum_12_1(close: pd.DataFrame, asof) -> pd.Series:
    hist = close.loc[: pd.Timestamp(asof)]
    if len(hist) < 252:
        return pd.Series(dtype=float)
    m12 = hist.iloc[-252].astype(float)
    m1 = hist.iloc[-22].astype(float)
    out = m1 / m12 - 1.0
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna()


def panel_dist_to_high(close: pd.DataFrame, asof, window: int = 252) -> pd.Series:
    hist = close.loc[: pd.Timestamp(asof)]
    if len(hist) < 20:
        return pd.Series(dtype=float)
    peak = hist.iloc[-window:].max()
    last = hist.iloc[-1].astype(float)
    out = last / peak.replace(0, np.nan) - 1.0
    return out.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class StatementCache:
    by_sym: dict[str, dict[str, list[tuple[pd.Timestamp, str, dict]]]] = field(default_factory=dict)
    dividends: dict[str, pd.DataFrame] = field(default_factory=dict)
    listings: dict[str, dict] = field(default_factory=dict)

    def asof(self, symbol: str, asof: date) -> dict[str, dict[str, Any]]:
        cutoff = pd.Timestamp(asof)
        out: dict[str, dict[str, Any]] = {}
        for kind, rows in self.by_sym.get(symbol, {}).items():
            chosen = None
            for rec in rows:
                if rec[0] <= cutoff:
                    chosen = rec
                else:
                    break
            if chosen is not None:
                out[kind] = {
                    "kind": kind,
                    "period": chosen[1],
                    "announced": chosen[0],
                    "payload": chosen[2],
                }
        return out

    def history(self, symbol: str, kind: str, asof: date) -> list[dict[str, Any]]:
        cutoff = pd.Timestamp(asof)
        rows = [r for r in self.by_sym.get(symbol, {}).get(kind, []) if r[0] <= cutoff]
        return [
            {"period": p, "announced": a, "payload": payload}
            for a, p, payload in reversed(rows)
        ]


def load_statement_cache(store, symbols: list[str] | None = None) -> StatementCache:
    cache = StatementCache()
    cache.listings = _listing_map(store)
    stmts = store.statements_frame(symbols)
    if not stmts.empty:
        grouped: dict[str, dict[str, list]] = {}
        for rec in stmts.itertuples(index=False):
            payload = rec.payload
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if not isinstance(payload, dict):
                payload = {}
            announced = pd.Timestamp(rec.announced)
            if pd.isna(announced):
                continue
            grouped.setdefault(str(rec.symbol), {}).setdefault(str(rec.kind), []).append(
                (announced, str(rec.period), payload)
            )
        for sym, kinds in grouped.items():
            for kind, rows in kinds.items():
                rows.sort(key=lambda x: x[0])
            cache.by_sym[sym] = kinds
    divs = store.dividends_frame(symbols)
    if not divs.empty:
        for sym, part in divs.groupby("symbol"):
            cache.dividends[str(sym)] = part
    return cache


def _overlay_yjbb(stmts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if "yjbb" not in stmts:
        return stmts
    income = dict(stmts.get("income") or {"payload": {}})
    payload = dict(income.get("payload") or {})
    extra = stmts["yjbb"].get("payload") or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if v is not None and v != "":
                payload[k] = v
    income["payload"] = payload
    stmts = dict(stmts)
    stmts["income"] = income
    return stmts


def fundamentals_asof(
    cache: StatementCache,
    asof: date,
    symbols: list[str],
    *,
    close: pd.DataFrame,
    raw_close: pd.DataFrame | None = None,
    amounts: pd.DataFrame | None = None,
) -> list[Fundamentals]:
    """用 announced≤T 的财报 + T 及之前的价格，不用当日 spot（避免把今天的 PE 回填历史）。"""
    from zhiku.data.codes import infer_market
    from zhiku.models import Market

    ts = pd.Timestamp(asof)
    close_h = close.loc[:ts] if not close.empty else close
    raw_h = raw_close.loc[:ts] if raw_close is not None and not raw_close.empty else None
    amt_h = amounts.loc[:ts] if amounts is not None and not amounts.empty else None
    rows: list[Fundamentals] = []
    for symbol in symbols:
        listing = cache.listings.get(symbol, {"symbol": symbol, "name": symbol, "instrument": "stock"})
        dl = listing.get("delist_date")
        if dl:
            try:
                if date.fromisoformat(str(dl)[:10]) <= asof:
                    continue
            except ValueError:
                pass
        stmts = _overlay_yjbb(cache.asof(symbol, asof))
        income_h = cache.history(symbol, "income", asof)
        balance_h = cache.history(symbol, "balance", asof)
        closes = close_h[symbol] if symbol in close_h.columns else None
        raw_closes = raw_h[symbol] if raw_h is not None and symbol in raw_h.columns else None
        amts = amt_h[symbol] if amt_h is not None and symbol in amt_h.columns else None
        div_df = cache.dividends.get(symbol, pd.DataFrame())
        if not div_df.empty and "announced" in div_df.columns:
            ann = pd.to_datetime(div_df["announced"], errors="coerce")
            ex = pd.to_datetime(div_df["ex_date"], errors="coerce")
            keep = (ann.isna() | (ann <= ts)) & (ex.isna() | (ex <= ts))
            div_df = div_df.loc[keep]
        ex_dates: list[date] = []
        if not div_df.empty and "ex_date" in div_df.columns:
            for raw in div_df["ex_date"].tolist():
                try:
                    ex_dates.append(date.fromisoformat(str(raw)[:10]))
                except ValueError:
                    continue
        raw_last = None
        if raw_closes is not None and not raw_closes.dropna().empty:
            raw_last = float(raw_closes.dropna().astype(float).iloc[-1])
        market = infer_market(symbol)
        row = snapshot_from_store(
            symbol=symbol,
            asof=asof,
            listing=listing,
            statements=stmts,
            prev_income=income_h[1]["payload"] if len(income_h) >= 2 else None,
            prev_balance=balance_h[1]["payload"] if len(balance_h) >= 2 else None,
            spot=None,
            closes=closes,
            amounts=amts,
            div_years=consecutive_div_years(ex_dates, asof),
            div_yield=dividend_yield_from_rows(div_df, asof, raw_close=raw_last),
            shares_issued=_shares_issued(balance_h),
            market=market if isinstance(market, Market) else Market.CN,
            raw_closes=raw_closes,
        )
        listed = listing.get("list_date")
        if listed:
            try:
                years = (asof - date.fromisoformat(str(listed)[:10])).days / 365.25
                data = row.model_dump()
                data["listed_years"] = years
                row = Fundamentals.model_validate(data)
            except ValueError:
                pass
        rows.append(row)
    return attach_percentiles(rows)


def monthly_screen_picks(
    settings: PipelineSettings,
    membership: dict,
    *,
    close: pd.DataFrame,
    cache: StatementCache | None = None,
    names: list[str] | None = None,
    decision_dates: list[pd.Timestamp] | None = None,
    raw_close: pd.DataFrame | None = None,
    amounts: pd.DataFrame | None = None,
) -> dict[str, dict[pd.Timestamp, list[str]]]:
    """每个决策日 T 用当时池 + T 前数据重跑公式。"""
    from zhiku.backtest.universe import members_on

    wanted = names or list(SCREEN_FUNCS)
    dates = decision_dates or month_end_trading_days(close.index)
    out: dict[str, dict[pd.Timestamp, list[str]]] = {n: {} for n in wanted}
    only_price = wanted == ["livermore"] and cache is None
    for dt in dates:
        members = [s for s in members_on(membership, dt) if s in close.columns]
        if not members:
            continue
        asof = dt.date() if hasattr(dt, "date") else date.fromisoformat(str(dt)[:10])
        if only_price:
            mom = panel_momentum_12_1(close, dt)
            dth = panel_dist_to_high(close, dt)
            rows = [
                Fundamentals(
                    symbol=s,
                    momentum_12_1=None if s not in mom.index else float(mom[s]),
                    dist_to_high=None if s not in dth.index else float(dth[s]),
                )
                for s in members
            ]
        else:
            if cache is None:
                raise ValueError("基本面公式需要 StatementCache")
            rows = fundamentals_asof(
                cache, asof, members, close=close, raw_close=raw_close, amounts=amounts
            )
        bundle = run_screens(rows, settings, names=wanted)
        for name, srows in bundle.screens.items():
            out[name][pd.Timestamp(dt)] = [r.symbol for r in srows]
    return out


def pick_coverage(picks_by_date: dict, dates: list[pd.Timestamp]) -> dict[str, float | int]:
    """有非空篮子的决策月占比。空仓月单独计数，不把净值停住当成「策略很稳」。"""
    n = len(dates)
    covered = 0
    for dt in dates:
        names = picks_by_date.get(pd.Timestamp(dt)) or picks_by_date.get(dt) or []
        if names:
            covered += 1
    return {
        "n_months": n,
        "n_covered": covered,
        "n_empty": max(n - covered, 0),
        "coverage_ratio": (covered / n) if n else 0.0,
    }


def universe_ew_picks(membership: dict, dates: list[pd.Timestamp], columns: list[str]) -> dict[pd.Timestamp, list[str]]:
    from zhiku.backtest.universe import members_on

    cols = set(columns)
    return {pd.Timestamp(dt): [s for s in members_on(membership, dt) if s in cols] for dt in dates}


def attach_diagnostics(
    result: BacktestResult,
    nav: pd.Series,
    bench: pd.Series | None,
    *,
    extras: dict | None = None,
    n_trials: int = 9,
    rf: float = 0.02,
) -> BacktestResult:
    extras = extras or {}
    result.fill = extras.get("fill")
    result.close_fallback_fills = extras.get("close_fallback_fills")
    result.two_way_turnover = extras.get("two_way_turnover")
    result.one_way_cost_bps = extras.get("one_way_cost_bps")
    result.two_way_cost_bps = extras.get("two_way_cost_bps")
    result.blocked_buys = extras.get("blocked_buys")
    if bench is not None and not bench.empty:
        result.excess_first_half, result.excess_second_half = split_half_excess(nav, bench)
        result.year_excess = year_excesses(nav, bench)
        if result.annualized is not None:
            b_ann = None
            from zhiku.backtest.metrics import _ann_return

            b_ann = _ann_return(bench.reindex(nav.index).dropna())
            if b_ann is not None:
                result.benchmark_annualized = b_ann
                result.excess_annualized = result.annualized - b_ann
                result.beat_benchmark = result.excess_annualized > 0
    monthly = nav.resample("ME").last().dropna()
    if monthly.empty:
        monthly = nav.dropna()
    rets = monthly.pct_change().dropna()
    if result.sharpe is not None and len(rets) >= 4:
        stats = deflated_sharpe_stats(
            result.sharpe,
            len(rets),
            n_trials,
            skew=float(rets.skew()) if len(rets) >= 3 else 0.0,
            excess_kurt=float(rets.kurt()) if len(rets) >= 4 else 0.0,
        )
        result.deflated_sharpe = stats["deflated_sharpe"]
        result.dsr_prob = stats["dsr_prob"]
        result.n_trials = n_trials
    return result


def backtest_timed_picks(
    prices: pd.DataFrame,
    picks_by_date: dict,
    settings: BacktestSettings,
    *,
    screen: str,
    opens: pd.DataFrame | None = None,
    limit_up: pd.DataFrame | None = None,
    limit_down: pd.DataFrame | None = None,
    membership: dict | None = None,
    bench_nav: pd.Series | None = None,
    fill: str | None = None,
    cost_bps: float | None = None,
    slippage_bps: float | None = None,
) -> BacktestResult:
    use_fill = fill or settings.fill or "next_open"
    nav, turnover, extras = rebalance_equal_weight(
        prices,
        picks_by_date,
        opens=opens,
        cost_bps=settings.cost_bps if cost_bps is None else cost_bps,
        slippage_bps=settings.slippage_bps if slippage_bps is None else slippage_bps,
        limit_up=limit_up,
        limit_down=limit_down,
        membership=membership,
        fill=use_fill,
    )
    dates = sorted({pd.Timestamp(k) for k in picks_by_date})
    cov = pick_coverage(picks_by_date, dates)
    if nav.empty:
        empty = BacktestResult(
            screen=screen,
            annualized=None,
            max_drawdown=None,
            sharpe=None,
            win_rate=None,
            turnover=None,
            notes=["时点名单为空或无法成交"],
            passed_gate=False,
            engine="pandas-pit",
            fill=use_fill,
            insufficient_data=True,
            gate_status="数据不足",
        )
        from zhiku.backtest.engine import attach_coverage

        return attach_coverage(empty, n_months=int(cov["n_months"]), n_covered=int(cov["n_covered"]))
    result = summarize_nav(
        nav,
        turnover=turnover,
        rf=settings.rf,
        periods_per_year=12.0,
        screen=screen,
        engine="pandas-pit",
    )
    result.notes = [
        f"调仓用决策日 T 之前的数据；成交={extras.get('fill')}。",
        "涨停不可新买、跌停不可卖；成本=单边换手×(佣金+滑点)。",
    ]
    attach_diagnostics(
        result,
        nav,
        bench_nav,
        extras=extras,
        n_trials=int(settings.n_trials or 9),
        rf=settings.rf,
    )
    from zhiku.backtest.engine import attach_coverage, finalize_gate

    attach_coverage(result, n_months=int(cov["n_months"]), n_covered=int(cov["n_covered"]))
    return finalize_gate(result, settings)


def robustness_suite(
    prices: pd.DataFrame,
    picks_fn,
    settings: BacktestSettings,
    *,
    screen: str,
    opens: pd.DataFrame | None = None,
    limit_up: pd.DataFrame | None = None,
    limit_down: pd.DataFrame | None = None,
    membership: dict | None = None,
    base_dates: list[pd.Timestamp],
) -> dict:
    """半段已在主结果里。这里：成本加倍、调仓日平移、分年（由主结果 year_excess 给出）。"""
    out: dict[str, Any] = {}
    doubled = backtest_timed_picks(
        prices,
        picks_fn(base_dates),
        settings,
        screen=screen,
        opens=opens,
        limit_up=limit_up,
        limit_down=limit_down,
        membership=membership,
        cost_bps=2.0 * settings.cost_bps,
        slippage_bps=2.0 * settings.slippage_bps,
    )
    out["cost_2x"] = {
        "annualized": doubled.annualized,
        "excess_annualized": doubled.excess_annualized,
        "sharpe": doubled.sharpe,
        "turnover": doubled.turnover,
    }
    for shift in (-5, -1, 1, 5):
        shifted = shift_trading_days(prices.index, base_dates, shift)
        res = backtest_timed_picks(
            prices,
            picks_fn(shifted),
            settings,
            screen=screen,
            opens=opens,
            limit_up=limit_up,
            limit_down=limit_down,
            membership=membership,
        )
        out[f"shift_{shift:+d}d"] = {
            "annualized": res.annualized,
            "excess_annualized": res.excess_annualized,
            "sharpe": res.sharpe,
        }
    return out


def evaluate_screens(
    settings: PipelineSettings,
    *,
    close: pd.DataFrame,
    membership: dict,
    opens: pd.DataFrame | None = None,
    raw_close: pd.DataFrame | None = None,
    amounts: pd.DataFrame | None = None,
    limit_up: pd.DataFrame | None = None,
    limit_down: pd.DataFrame | None = None,
    cache: StatementCache | None = None,
    names: list[str] | None = None,
    with_robustness: bool = False,
) -> dict[str, BacktestResult]:
    wanted = names or list(SCREEN_FUNCS)
    dates = month_end_trading_days(close.index)
    picks = monthly_screen_picks(
        settings,
        membership,
        close=close,
        cache=cache,
        names=wanted,
        decision_dates=dates,
        raw_close=raw_close,
        amounts=amounts,
    )
    ew = universe_ew_picks(membership, dates, list(close.columns))
    bench_nav, _, bench_extras = rebalance_equal_weight(
        close, ew, opens=opens, cost_bps=0, slippage_bps=0, membership=membership, fill=settings.backtest.fill
    )
    results: dict[str, BacktestResult] = {}
    n_trials = max(int(settings.backtest.n_trials or len(wanted)), len(wanted))
    settings.backtest.n_trials = n_trials
    for name in wanted:
        bt = backtest_timed_picks(
            close,
            picks.get(name, {}),
            settings.backtest,
            screen=name,
            opens=opens,
            limit_up=limit_up,
            limit_down=limit_down,
            membership=membership,
            bench_nav=bench_nav,
        )
        if with_robustness and cache is not None:

            def _picks_at(decisions, _name=name):
                return monthly_screen_picks(
                    settings,
                    membership,
                    close=close,
                    cache=cache,
                    names=[_name],
                    decision_dates=decisions,
                    raw_close=raw_close,
                    amounts=amounts,
                ).get(_name, {})

            bt.robustness = robustness_suite(
                close,
                _picks_at,
                settings.backtest,
                screen=name,
                opens=opens,
                limit_up=limit_up,
                limit_down=limit_down,
                membership=membership,
                base_dates=dates,
            )
        results[name] = bt
    if not bench_nav.empty:
        ew_bt = summarize_nav(bench_nav, rf=settings.backtest.rf, periods_per_year=12.0, screen="universe_ew")
        ew_bt.notes = ["时点股票池等权，T+1 开盘、成本 0。对照行用此净值，不用月末收盘等权。"]
        ew_bt.engine = "pandas-pit"
        attach_diagnostics(ew_bt, bench_nav, None, extras=bench_extras)
        ew_bt.gate_status = "对照"
        from zhiku.backtest.engine import attach_coverage

        attach_coverage(ew_bt, n_months=len(dates), n_covered=len(dates))
        results["universe_ew"] = ew_bt
    return results
