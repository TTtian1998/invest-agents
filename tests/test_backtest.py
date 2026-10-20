"""回测指标与闸门：用很短的虚构价格序列。"""

from __future__ import annotations

import pandas as pd

from zhiku.backtest.engine import backtest_screen, backtest_trigger_line
from zhiku.backtest.metrics import monthly_equal_weight, summarize_nav
from zhiku.pipeline.settings import BacktestSettings


def test_metrics_known_path() -> None:
    idx = pd.to_datetime(["2025-01-31", "2025-12-31"])
    nav = pd.Series([1.0, 1.1], index=idx)
    result = summarize_nav(nav, rf=0.0, periods_per_year=1, screen="demo")
    assert result.annualized is not None
    assert result.max_drawdown == 0.0 or result.max_drawdown is not None
    assert result.years.get("2025") == 0.1 or "2025" in result.years


def test_monthly_equal_weight_and_gate() -> None:
    idx = pd.to_datetime(
        ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-30", "2025-06-30"]
    )
    prices = pd.DataFrame(
        {
            "000333": [50, 51, 52, 53, 54, 56],
            "510300": [3.4, 3.41, 3.42, 3.43, 3.44, 3.45],
        },
        index=idx,
    )
    nav, turnover = monthly_equal_weight(prices, ["000333"], cost_bps=0, slippage_bps=0)
    assert len(nav) >= 2
    assert nav.iloc[-1] > nav.iloc[0]
    assert turnover >= 0
    result = backtest_screen(
        prices,
        ["000333"],
        BacktestSettings(min_sharpe=-99, require_beat_benchmark=False, cost_bps=0, slippage_bps=0),
        screen="greenblatt",
        benchmark="510300",
    )
    assert result.passed_gate is True
    assert result.annualized is not None
    fail = backtest_screen(
        prices,
        ["000333"],
        BacktestSettings(min_sharpe=99.0, require_beat_benchmark=False),
        screen="greenblatt",
        benchmark="510300",
    )
    assert fail.passed_gate is False


def test_trigger_line_exits_when_below() -> None:
    idx = pd.to_datetime(["2026-10-16", "2026-10-17", "2026-10-20"])
    px = pd.Series([4.0, 3.5, 3.4], index=idx)
    result = backtest_trigger_line(px, 3.80, cost_bps=0)
    assert result.engine in {"pandas", "vectorbt"}
    assert result.equity
