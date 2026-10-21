"""时点闸门：前视、T+1 开盘、涨停、半段超额、deflated Sharpe。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhiku.backtest.engine import _gate
from zhiku.backtest.metrics import BacktestResult, rebalance_equal_weight
from zhiku.backtest.pit_gate import (
    deflated_sharpe_stats,
    expected_max_sr,
    panel_dist_to_high,
    panel_momentum_12_1,
    shift_trading_days,
)
from zhiku.data.compute import dist_to_high, momentum_12_1
from zhiku.pipeline.settings import BacktestSettings


def _close_path() -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=280)
    rng = np.linspace(10.0, 20.0, len(idx))
    spike = pd.Series(rng, index=idx)
    spike.iloc[-5:] = 80.0
    other = pd.Series(np.linspace(10.0, 11.0, len(idx)), index=idx)
    return pd.DataFrame({"AAA": spike, "BBB": other})


def test_asof_momentum_and_high_ignore_future_spike() -> None:
    px = _close_path()
    cutoff = px.index[-10]
    truncated = px["AAA"].loc[:cutoff]
    mom_asof = momentum_12_1(px["AAA"], asof=cutoff)
    dth_asof = dist_to_high(px["AAA"], asof=cutoff)
    assert mom_asof == momentum_12_1(truncated)
    assert dth_asof == dist_to_high(truncated)
    leaked = px.copy()
    leaked.iloc[-1, leaked.columns.get_loc("AAA")] = 999.0
    assert momentum_12_1(leaked["AAA"], asof=cutoff) == mom_asof
    assert dist_to_high(leaked["AAA"], asof=cutoff) == dth_asof
    panel_mom = panel_momentum_12_1(px, cutoff)
    panel_dth = panel_dist_to_high(px, cutoff)
    assert abs(float(panel_mom["AAA"]) - mom_asof) < 1e-12
    assert abs(float(panel_dth["AAA"]) - dth_asof) < 1e-12


def test_t1_open_fill_uses_next_open() -> None:
    idx = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-28", "2025-03-03", "2025-03-04"])
    close = pd.DataFrame({"AAA": [10.0, 10.0, 10.0, 10.0, 10.0]}, index=idx)
    opens = pd.DataFrame({"AAA": [10.0, 10.0, 10.0, 12.0, 12.0]}, index=idx)
    picks = {pd.Timestamp("2025-01-31"): ["AAA"], pd.Timestamp("2025-02-28"): ["AAA"]}
    nav, turnover, extras = rebalance_equal_weight(
        close, picks, opens=opens, cost_bps=0, slippage_bps=0, fill="next_open"
    )
    assert extras["fill"] == "next_open"
    assert len(nav) >= 2
    # 1/31 决策 → 2/3 开盘 10；2/28 决策 → 3/3 开盘 12
    assert abs(float(nav.iloc[-1]) - 1.2) < 1e-9
    assert turnover >= 0


def test_limit_up_blocks_new_buy() -> None:
    idx = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-28", "2025-03-03"])
    close = pd.DataFrame({"AAA": [10.0, 11.0, 11.0, 12.0], "BBB": [10.0, 10.0, 10.0, 10.0]}, index=idx)
    opens = close.copy()
    up = pd.DataFrame(
        {"AAA": [False, True, False, False], "BBB": [False, False, False, False]},
        index=idx,
    )
    picks = {pd.Timestamp("2025-01-31"): ["AAA"], pd.Timestamp("2025-02-28"): ["AAA"]}
    nav, _, extras = rebalance_equal_weight(
        close, picks, opens=opens, cost_bps=0, slippage_bps=0, limit_up=up, fill="next_open"
    )
    assert extras["blocked_buys"] >= 1
    # 第一次成交日涨停买不进，净值应接近 1
    assert float(nav.iloc[0]) == 1.0


def test_gate_requires_both_half_excesses() -> None:
    settings = BacktestSettings(
        min_sharpe=-99,
        require_beat_benchmark=True,
        require_sharpe_ge_benchmark=False,
        require_half_excess=True,
    )
    ok = BacktestResult(
        "demo",
        0.2,
        -0.1,
        1.0,
        0.6,
        0.2,
        excess_annualized=0.1,
        beat_benchmark=True,
        excess_first_half=0.05,
        excess_second_half=0.04,
    )
    assert _gate(ok, settings) is True
    bad = BacktestResult(
        "demo",
        0.2,
        -0.1,
        1.0,
        0.6,
        0.2,
        excess_annualized=0.1,
        beat_benchmark=True,
        excess_first_half=0.15,
        excess_second_half=-0.02,
    )
    assert _gate(bad, settings) is False


def test_deflated_sharpe_haircut_grows_with_trials() -> None:
    one = deflated_sharpe_stats(1.0, n_obs=36, n_trials=1)
    many = deflated_sharpe_stats(1.0, n_obs=36, n_trials=9)
    assert one["deflated_sharpe"] > many["deflated_sharpe"]
    assert many["sr0"] > one["sr0"]
    assert 0 < many["dsr_prob"] < 1
    assert expected_max_sr(9, 0.2) > expected_max_sr(2, 0.2)


def test_next_open_skips_panel_last_bar() -> None:
    idx = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-28"])
    close = pd.DataFrame({"AAA": [10.0, 11.0, 4.0]}, index=idx)
    opens = pd.DataFrame({"AAA": [10.0, 10.0, 1.0]}, index=idx)
    picks = {pd.Timestamp("2025-01-31"): ["AAA"], pd.Timestamp("2025-02-28"): ["AAA"]}
    nav, _, extras = rebalance_equal_weight(
        close, picks, opens=opens, cost_bps=0, slippage_bps=0, fill="next_open"
    )
    assert extras.get("n_rebals") == 1
    assert pd.Timestamp("2025-02-28") not in nav.index


def test_shift_trading_days_moves_along_calendar() -> None:
    idx = pd.bdate_range("2025-01-02", periods=20)
    moved = shift_trading_days(idx, [idx[5]], 2)
    assert moved[0] == idx[7]
