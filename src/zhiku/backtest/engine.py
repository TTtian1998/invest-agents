"""选股篮子与触发线回测。优先 vectorbt，否则 pandas。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datetime import date

from zhiku.backtest.metrics import (
    BacktestResult,
    monthly_equal_weight,
    split_half_excess,
    summarize_nav,
    year_excesses,
)
from zhiku.pipeline.settings import BacktestSettings


def vectorbt_available() -> bool:
    try:
        import vectorbt  # noqa: F401

        return True
    except ImportError:
        return False


def load_price_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    panel = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return panel


def sanitize_mixed_adjust(
    qfq: pd.DataFrame,
    none: pd.DataFrame | None = None,
    *,
    jump: float = 0.20,
) -> pd.DataFrame:
    """丢掉末日「前复权被不复权价覆盖」的跳变。|涨跌|>jump 且末日 qfq≈none 则置空。"""
    if qfq is None or qfq.empty or len(qfq.index) < 2:
        return qfq
    panel = qfq.copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    last = panel.index[-1]
    prev = panel.index[-2]
    none_p = None
    if none is not None and not none.empty:
        none_p = none.copy()
        none_p.index = pd.to_datetime(none_p.index)
    drop_cols: list[str] = []
    for col in panel.columns:
        p0, p1 = panel.loc[prev, col], panel.loc[last, col]
        if pd.isna(p0) or pd.isna(p1) or float(p0) == 0:
            continue
        ret = float(p1) / float(p0) - 1.0
        if abs(ret) <= jump:
            continue
        mixed = False
        if none_p is not None and col in none_p.columns and last in none_p.index:
            nv = none_p.loc[last, col]
            if pd.notna(nv) and abs(float(p1) - float(nv)) / max(abs(float(nv)), 1e-9) < 0.02:
                mixed = True
        elif abs(ret) > jump:
            mixed = True
        if mixed:
            drop_cols.append(col)
    if drop_cols:
        panel.loc[last, drop_cols] = pd.NA
        if panel.loc[last].isna().all():
            panel = panel.drop(index=last)
    return panel


def load_price_panel_from_store(store, symbols: list[str] | None = None, *, adjust: str = "qfq") -> pd.DataFrame:
    from zhiku.data.store import MarketStore

    if not isinstance(store, MarketStore):
        return pd.DataFrame()
    df = store.bars(symbols, adjust=adjust)
    if df.empty:
        return pd.DataFrame()
    panel = store.price_panel(list(df["symbol"].unique()), adjust=adjust, field="close")
    if adjust == "qfq":
        none = store.price_panel(list(df["symbol"].unique()), adjust="none", field="close")
        panel = sanitize_mixed_adjust(panel, none)
    return panel


def _nav_metrics(nav: pd.Series, *, screen: str, settings: BacktestSettings) -> BacktestResult:
    nav = nav.dropna().astype(float)
    if nav.empty:
        return BacktestResult(screen, None, None, None, None, None, notes=["缺净值"])
    monthly = nav.resample("ME").last().dropna()
    if monthly.empty:
        monthly = nav.resample("M").last().dropna()
    use = monthly if len(monthly) >= 2 else nav
    periods = 12.0 if len(monthly) >= 2 else 252.0
    return summarize_nav(use, rf=settings.rf, periods_per_year=periods, screen=screen)


def compute_benchmarks(
    prices: pd.DataFrame,
    settings: BacktestSettings,
    *,
    membership: dict | None = None,
    universe_note: str | None = None,
) -> dict[str, BacktestResult]:
    """同池时点等权 + 沪深300/中证800（ETF 或指数代码）。成本为 0，作对照不做交易。"""
    out: dict[str, BacktestResult] = {}
    if prices is None or prices.empty:
        return out
    extras = set(settings.benchmark_symbols or [])
    if settings.benchmark and settings.benchmark not in {"universe_ew", "csi300", "csi800"}:
        extras.add(settings.benchmark)
    extras.update({"universe_ew", "csi300", "csi800"})
    if membership:
        members: list[str] = []
        seen: set[str] = set()
        for names in membership.values():
            for symbol in names:
                if symbol in prices.columns and symbol not in extras and symbol not in seen:
                    seen.add(symbol)
                    members.append(symbol)
        nav, _ = monthly_equal_weight(
            prices, members, cost_bps=0, slippage_bps=0, membership=membership
        )
        if not nav.empty:
            out["universe_ew"] = summarize_nav(
                nav, rf=settings.rf, periods_per_year=12.0, screen="universe_ew"
            )
            out["universe_ew"].notes = [universe_note or "时点股票池等权"]
    else:
        members = [c for c in prices.columns if c not in extras]
        if len(members) >= 2:
            nav, _ = monthly_equal_weight(prices, members, cost_bps=0, slippage_bps=0)
            if not nav.empty:
                out["universe_ew"] = summarize_nav(
                    nav, rf=settings.rf, periods_per_year=12.0, screen="universe_ew"
                )
                out["universe_ew"].notes = ["截面股票池等权（无历史时点名单）"]
    aliases = {
        "csi300": ("510300", "000300"),
        "csi800": ("000906",),
    }
    for name, symbols in aliases.items():
        for sym in symbols:
            if sym not in prices.columns:
                continue
            series = prices[sym].dropna()
            if len(series) < 2:
                continue
            last_px = float(series.iloc[-1])
            # 000300/000906 指数点位通常为千级；个位数多半是错把深市股票当指数
            if sym in {"000300", "000906"} and last_px < 100:
                continue
            nav = series / float(series.iloc[0])
            out[name] = _nav_metrics(nav, screen=name, settings=settings)
            out[name].notes = [f"指数对照用 {sym}"]
            break
    return out


def attach_benchmarks(
    result: BacktestResult,
    benches: dict[str, BacktestResult],
    *,
    sanity_max_ann_gap: float = 0.25,
    primary_key: str | None = "universe_ew",
) -> BacktestResult:
    payload = {k: v.to_dict() for k, v in benches.items()}
    result.benchmarks = {
        k: {
            "annualized": v.get("annualized"),
            "max_drawdown": v.get("max_drawdown"),
            "sharpe": v.get("sharpe"),
            "notes": v.get("notes"),
            "engine": v.get("engine"),
            "fill": v.get("fill"),
            "close_fallback_fills": v.get("close_fallback_fills"),
        }
        for k, v in payload.items()
    }
    primary = (
        benches.get(primary_key) if primary_key else None
    ) or benches.get("universe_ew") or benches.get("csi300") or benches.get("csi800")
    if primary is not None and result.annualized is not None and primary.annualized is not None:
        result.benchmark_annualized = primary.annualized
        result.excess_annualized = result.annualized - primary.annualized
        result.beat_benchmark = result.excess_annualized > 0
        nav = _equity_to_series(result)
        bench_nav = _equity_to_series(primary)
        if not nav.empty and not bench_nav.empty:
            if result.excess_first_half is None:
                result.excess_first_half, result.excess_second_half = split_half_excess(nav, bench_nav)
            if not result.year_excess:
                result.year_excess = year_excesses(nav, bench_nav)
    ew = benches.get("universe_ew")
    idx = benches.get("csi300") or benches.get("csi800")
    if ew is not None and idx is not None and ew.annualized is not None and idx.annualized is not None:
        gap = abs(ew.annualized - idx.annualized)
        result.sanity_ok = gap <= sanity_max_ann_gap
        result.sanity_note = (
            f"同池等权年化 {ew.annualized:.2%} vs 指数 {idx.annualized:.2%}，差 {gap:.2%}"
            + ("，超过阈值，回测数字仍需对照时点池口径" if not result.sanity_ok else "")
        )
    return result


def crosscheck_public_index(store, on: date | None = None) -> list[str]:
    """用公开指数点位交叉核对本地 510300/000300。失败写缺数据，不编造。"""
    notes: list[str] = []
    local = None
    for symbol, adjust in (("510300", "none"), ("510300", "qfq"), ("000300", "none"), ("000300", "qfq")):
        bar = store.last_bar(symbol, adjust, before=None)
        if bar and bar.get("close") is not None:
            local = (symbol, adjust, float(bar["close"]), str(bar.get("date")))
            break
    if local is None:
        notes.append("缺数据：本地无 510300/000300，无法与公开点位交叉核对")
        return notes
    try:
        import yfinance as yf
    except ImportError:
        notes.append(f"本地 {local[0]} {local[2]:.4f}（{local[3]}），缺数据：未安装 yfinance 做公开核对")
        return notes
    public = None
    for ticker in ("510300.SS", "000300.SS"):
        try:
            hist = yf.Ticker(ticker).history(period="10d")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"公开点位 {ticker} 失败：{exc}")
            continue
        if hist is None or getattr(hist, "empty", True):
            continue
        close = hist["Close"].dropna()
        if close.empty:
            continue
        public = (ticker, float(close.iloc[-1]), str(close.index[-1])[:10])
        break
    if public is None:
        notes.append(f"本地 {local[0]} 收盘 {local[2]:.4f}（{local[3]}），缺数据：无公开指数点位")
        return notes
    gap = abs(local[2] - public[1]) / max(abs(public[1]), 1e-9)
    notes.append(
        f"交叉核对：本地 {local[0]}/{local[1]} {local[2]:.4f}（{local[3]}）"
        f" vs 公开 {public[0]} {public[1]:.4f}（{public[2]}），相对差 {gap:.2%}"
    )
    if gap > 0.08 and local[0] != "000300":
        notes.append("交叉核对偏离超过 8%，请检查复权或代码映射")
    return notes


def backtest_screen(
    prices: pd.DataFrame,
    picks: list[str],
    settings: BacktestSettings,
    *,
    screen: str,
    benchmark: str | None = None,
    delist: dict[str, date] | None = None,
    asof: date | None = None,
    limit_up: pd.DataFrame | None = None,
    limit_down: pd.DataFrame | None = None,
    membership: dict | None = None,
    universe_note: str | None = None,
) -> BacktestResult:
    if not picks:
        return BacktestResult(
            screen=screen,
            annualized=None,
            max_drawdown=None,
            sharpe=None,
            win_rate=None,
            turnover=None,
            notes=["选股结果为空，无法回测"],
            passed_gate=False,
            engine="pandas",
        )
    panel = prices
    if delist:
        from zhiku.events.pit import drop_survivors

        panel = drop_survivors(panel, delist, asof or date.max)
    bench_col = benchmark or settings.benchmark
    nav, turnover = monthly_equal_weight(
        panel,
        picks,
        cost_bps=settings.cost_bps,
        slippage_bps=settings.slippage_bps,
        limit_up=limit_up,
        limit_down=limit_down,
        membership=membership,
    )
    bench_nav = None
    if bench_col in panel.columns:
        b = panel[bench_col].dropna()
        if len(b) >= 2:
            bench_nav = b / float(b.iloc[0])
    engine = "pandas"
    gate_notes: list[str] = []
    if vectorbt_available():
        engine = "pandas+vectorbt"
        gate_notes.append("已检测到 vectorbt，净值仍按月度等权再平衡计算以便与无 extra 时对齐。")
    else:
        gate_notes.append("未安装 [backtest] extra，使用内置 pandas 月度等权再平衡。")
    gate_notes.append(
        "财务截面须按公告日对齐（zhiku.events.pit.load_asof_universe）；价格面板按退市日截断，避免幸存者偏差。"
    )
    if membership is not None:
        gate_notes.append(universe_note or "再平衡只持有当时股票池内且有日线的样本，不用今天的成分回填历史。")
    if limit_up is not None or limit_down is not None:
        gate_notes.append("涨停不可买入、跌停不可卖出；成本与滑点按换手从净值扣除。")
    result = summarize_nav(
        nav,
        turnover=turnover,
        rf=settings.rf,
        periods_per_year=12.0,
        benchmark=bench_nav,
        screen=screen,
        engine=engine,
    )
    if result.annualized is not None and result.benchmark_annualized is not None:
        result.excess_annualized = result.annualized - result.benchmark_annualized
    result.notes = gate_notes
    result.passed_gate = _gate(result, settings)
    return result


def _equity_to_series(result: BacktestResult) -> pd.Series:
    if not result.equity:
        return pd.Series(dtype=float)
    return pd.Series({pd.Timestamp(d): float(v) for d, v in result.equity})


def attach_coverage(result: BacktestResult, *, n_months: int, n_covered: int) -> BacktestResult:
    result.n_months = int(n_months)
    result.n_covered = int(n_covered)
    result.n_empty = max(int(n_months) - int(n_covered), 0)
    result.coverage_ratio = (float(n_covered) / float(n_months)) if n_months else 0.0
    return result


def _gate(result: BacktestResult, settings: BacktestSettings) -> bool:
    if result.insufficient_data:
        return False
    if result.sharpe is not None and result.sharpe < settings.min_sharpe:
        return False
    if settings.require_beat_benchmark:
        if result.excess_annualized is not None and result.excess_annualized <= 0:
            return False
        if result.beat_benchmark is False:
            return False
        if settings.require_sharpe_ge_benchmark:
            ew = (result.benchmarks or {}).get("universe_ew") or {}
            ew_sharpe = ew.get("sharpe")
            if ew_sharpe is not None and (result.sharpe is None or result.sharpe < float(ew_sharpe)):
                return False
    if getattr(settings, "require_half_excess", False):
        # 没算出来的短样本不挡；算出来则两段都须 > 0
        if result.excess_first_half is not None and result.excess_first_half <= 0:
            return False
        if result.excess_second_half is not None and result.excess_second_half <= 0:
            return False
    return True


def finalize_gate(result: BacktestResult, settings: BacktestSettings) -> BacktestResult:
    if result.deflated_sharpe is None and result.sharpe is not None:
        from zhiku.backtest.pit_gate import deflated_sharpe_stats

        n_obs = max(len(result.equity) - 1, 0)
        n_trials = int(getattr(settings, "n_trials", 9) or 9)
        if n_obs >= 4:
            stats = deflated_sharpe_stats(result.sharpe, n_obs, n_trials)
            result.deflated_sharpe = stats["deflated_sharpe"]
            result.dsr_prob = stats["dsr_prob"]
            result.n_trials = n_trials
    min_cov = float(getattr(settings, "min_coverage", 0.80) or 0.0)
    if (
        result.coverage_ratio is not None
        and min_cov > 0
        and result.coverage_ratio + 1e-12 < min_cov
    ):
        result.insufficient_data = True
        result.passed_gate = False
        result.gate_status = "数据不足"
        return result
    result.passed_gate = _gate(result, settings)
    if result.gate_status in {None, ""}:
        result.gate_status = "通过" if result.passed_gate else "未过闸"
    return result


def backtest_trigger_line(
    prices: pd.Series,
    level: float,
    *,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """收盘跌破 level 则空仓，否则满仓。用于把原有触发线接到回测。"""
    px = prices.dropna().astype(float).sort_index()
    if px.empty:
        return BacktestResult("trigger", None, None, None, None, None, notes=["缺价格"])
    held = (px >= level).astype(float)
    ret = px.pct_change().fillna(0.0)
    strat = held.shift(1).fillna(0.0) * ret
    cost = held.diff().abs().fillna(0.0) * (cost_bps / 10000.0)
    strat = strat - cost
    nav = (1.0 + strat).cumprod()
    engine = "vectorbt" if vectorbt_available() else "pandas"
    return summarize_nav(
        nav, returns=strat, rf=0.0, periods_per_year=252, screen="trigger", engine=engine
    )


def backtest_event_trades(
    prices: pd.DataFrame,
    events: list[dict],
    *,
    cost_bps: float = 10.0,
    delist: dict[str, date] | None = None,
) -> BacktestResult:
    """事件交易回测：下一根 K 线开仓，time_stop 根后或止损出场。与实盘时间止损同一套天数。"""
    if prices is None or prices.empty or not events:
        return BacktestResult("event", None, None, None, None, None, notes=["缺价格或无事件"])
    px = prices.sort_index()
    if delist:
        from zhiku.events.pit import drop_survivors

        px = drop_survivors(px, delist, date.max)
    rets = px.pct_change().fillna(0.0)
    strat = pd.Series(0.0, index=px.index)
    notes = ["事件回测使用 time_stop，不用无限持有。"]
    for ev in events:
        symbol = str(ev.get("symbol") or "")
        if symbol not in px.columns:
            notes.append(f"{symbol} 缺价格")
            continue
        entry = pd.Timestamp(ev.get("entry") or ev.get("date"))
        days = int(ev.get("time_stop") or ev.get("max_hold_days") or 5)
        direction = 1.0 if ev.get("direction") in {"做多", "long", 1, "1"} else -1.0
        stop_pct = float(ev.get("stop_pct") or 0.03)
        after = px.index[px.index > entry]
        if len(after) < 1:
            notes.append(f"{symbol} {entry.date()} 之后无 K 线")
            continue
        start = after[0]
        hold_idx = px.index[(px.index >= start)]
        if len(hold_idx) > days:
            hold_idx = hold_idx[:days]
        entry_px = float(px.loc[start, symbol])
        if entry_px <= 0:
            continue
        for i, ts in enumerate(hold_idx):
            r = float(rets.loc[ts, symbol]) if ts in rets.index else 0.0
            strat.loc[ts] = strat.loc[ts] + direction * r
            if i == 0:
                strat.loc[ts] = strat.loc[ts] - cost_bps / 10000.0
            last_px = float(px.loc[ts, symbol])
            move = (last_px / entry_px - 1.0) * direction
            if move <= -stop_pct:
                break
    nav = (1.0 + strat).cumprod()
    result = summarize_nav(
        nav, returns=strat, rf=0.0, periods_per_year=252, screen="event", engine="pandas"
    )
    result.notes = notes
    return result
