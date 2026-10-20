"""选股篮子与触发线回测。优先 vectorbt，否则 pandas。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datetime import date

from zhiku.backtest.metrics import BacktestResult, monthly_equal_weight, summarize_nav
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


def load_price_panel_from_store(store, symbols: list[str] | None = None, *, adjust: str = "qfq") -> pd.DataFrame:
    from zhiku.data.store import MarketStore

    if not isinstance(store, MarketStore):
        return pd.DataFrame()
    df = store.bars(symbols, adjust=adjust)
    if df.empty:
        return pd.DataFrame()
    return store.price_panel(list(df["symbol"].unique()), adjust=adjust, field="close")


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
    result.notes = gate_notes
    result.passed_gate = _gate(result, settings)
    return result


def _gate(result: BacktestResult, settings: BacktestSettings) -> bool:
    if result.sharpe is not None and result.sharpe < settings.min_sharpe:
        return False
    if settings.require_beat_benchmark and result.beat_benchmark is False:
        return False
    return True


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
