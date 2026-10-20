"""选股篮子与触发线回测。优先 vectorbt，否则 pandas。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def backtest_screen(
    prices: pd.DataFrame,
    picks: list[str],
    settings: BacktestSettings,
    *,
    screen: str,
    benchmark: str | None = None,
) -> BacktestResult:
    bench_col = benchmark or settings.benchmark
    nav, turnover = monthly_equal_weight(
        prices, picks, cost_bps=settings.cost_bps, slippage_bps=settings.slippage_bps
    )
    bench_nav = None
    if bench_col in prices.columns:
        b = prices[bench_col].dropna()
        if len(b) >= 2:
            bench_nav = b / float(b.iloc[0])
    engine = "pandas"
    notes: list[str] = []
    if vectorbt_available():
        engine = "pandas+vectorbt"
        notes.append("已检测到 vectorbt，净值仍按月度等权再平衡计算以便与无 extra 时对齐。")
    else:
        notes.append("未安装 [backtest] extra，使用内置 pandas 月度等权再平衡。")
    notes.append("MVP 用当前截面排名冻结篮子，再在历史价格上再平衡；点-in-time 财务为后续 TODO。")
    result = summarize_nav(
        nav,
        turnover=turnover,
        rf=settings.rf,
        periods_per_year=12.0,
        benchmark=bench_nav,
        screen=screen,
        engine=engine,
    )
    result.notes = notes
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
    cost = (held.diff().abs().fillna(0.0) * (cost_bps / 10000.0))
    strat = strat - cost
    nav = (1.0 + strat).cumprod()
    engine = "vectorbt" if vectorbt_available() else "pandas"
    return summarize_nav(nav, returns=strat, rf=0.0, periods_per_year=252, screen="trigger", engine=engine)
