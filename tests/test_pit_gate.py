"""时点闸门：前视、T+1 开盘、涨停、半段超额、deflated Sharpe。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from zhiku.backtest.engine import _gate, attach_coverage, finalize_gate
from zhiku.backtest.metrics import BacktestResult, rebalance_equal_weight
from zhiku.backtest.pit_gate import (
    StatementCache,
    deflated_sharpe_stats,
    expected_max_sr,
    fundamentals_asof,
    monthly_screen_picks,
    panel_dist_to_high,
    panel_momentum_12_1,
    pick_coverage,
    shift_trading_days,
)
from zhiku.data.compute import dist_to_high, momentum_12_1
from zhiku.pipeline.settings import BacktestSettings, PipelineSettings


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


def _cache_two_names() -> StatementCache:
    cache = StatementCache()
    good = {
        "营业利润": 80.0,
        "净利润": 50.0,
        "营业总收入": 400.0,
        "营业成本": 240.0,
        "利润总额": 80.0,
        "利息费用": 2.0,
        "每股收益": 2.0,
    }
    weak = {
        "营业利润": 8.0,
        "净利润": 5.0,
        "营业总收入": 40.0,
        "营业成本": 30.0,
        "利润总额": 8.0,
        "利息费用": 1.0,
        "每股收益": 0.2,
    }
    today = {
        "营业利润": 800.0,
        "净利润": 500.0,
        "营业总收入": 4000.0,
        "营业成本": 400.0,
        "利润总额": 800.0,
        "利息费用": 2.0,
        "每股收益": 20.0,
    }
    bal = {
        "流动资产合计": 200.0,
        "货币资金": 40.0,
        "流动负债合计": 80.0,
        "短期借款": 10.0,
        "固定资产": 90.0,
        "资产总计": 500.0,
        "负债合计": 180.0,
        "股东权益合计": 320.0,
        "长期借款": 50.0,
        "股本": 10.0,
    }
    cash = {"经营活动产生的现金流量净额": 60.0}

    def _rows(income_payload):
        return {
            "income": [(pd.Timestamp("2023-04-20"), "2023-03-31", income_payload)],
            "balance": [(pd.Timestamp("2023-04-20"), "2023-03-31", bal)],
            "cashflow": [(pd.Timestamp("2023-04-20"), "2023-03-31", cash)],
        }

    cache.by_sym["600001"] = _rows(good)
    cache.by_sym["000002"] = _rows(weak)
    # 今天才公告的 000002 暴利：2024-01 调仓日不得用
    cache.by_sym["000002"]["income"].append((pd.Timestamp("2025-12-01"), "2025-09-30", today))
    cache.by_sym["000002"]["income"].sort(key=lambda x: x[0])
    cache.listings = {
        "600001": {"symbol": "600001", "name": "甲", "instrument": "stock"},
        "000002": {"symbol": "000002", "name": "乙", "instrument": "stock"},
    }
    return cache


def test_rebalance_uses_only_announced_asof_not_today_list() -> None:
    """任一调仓日只用当时已公告数据；禁止把今天的名单套到每个月。"""
    cache = _cache_two_names()
    idx = pd.bdate_range("2023-01-03", "2026-02-02")
    close = pd.DataFrame(
        {"600001": np.linspace(10.0, 12.0, len(idx)), "000002": np.linspace(10.0, 11.0, len(idx))},
        index=idx,
    )
    raw = close * 1.0
    membership = {pd.Timestamp(d): ["600001", "000002"] for d in idx}
    settings = PipelineSettings()
    settings.greenblatt.top_n = 1
    settings.universe.min_mktcap_yi = 0
    settings.universe.min_amount_cny = {"CN": 0, "HK": 0, "US": 0}
    early = date(2024, 1, 31)
    today = date(2026, 1, 30)
    early_rows = fundamentals_asof(cache, early, ["600001", "000002"], close=close, raw_close=raw)
    today_rows = fundamentals_asof(cache, today, ["600001", "000002"], close=close, raw_close=raw)
    early_ebit = {r.symbol: r.ebit for r in early_rows}
    today_ebit = {r.symbol: r.ebit for r in today_rows}
    assert early_ebit["000002"] is not None and early_ebit["000002"] < 20
    assert today_ebit["000002"] is not None and today_ebit["000002"] > 100
    leaked = cache.asof("000002", early)
    assert leaked["income"]["payload"]["净利润"] == 5.0
    future = cache.asof("000002", today)
    assert future["income"]["payload"]["净利润"] == 500.0

    picks = monthly_screen_picks(
        settings,
        membership,
        close=close,
        cache=cache,
        names=["greenblatt"],
        decision_dates=[pd.Timestamp(early), pd.Timestamp(today)],
        raw_close=raw,
    )
    early_names = picks["greenblatt"][pd.Timestamp(early)]
    today_names = picks["greenblatt"][pd.Timestamp(today)]
    assert early_names == ["600001"]
    assert today_names == ["000002"]
    # 若误用今日名单，两个月都会是 000002
    assert early_names != today_names


def test_low_coverage_is_insufficient_not_gate_fail() -> None:
    settings = BacktestSettings(
        min_sharpe=-99,
        require_beat_benchmark=True,
        require_sharpe_ge_benchmark=False,
        require_half_excess=True,
        min_coverage=0.80,
    )
    result = BacktestResult(
        "demo",
        0.30,
        -0.10,
        1.5,
        0.6,
        0.2,
        excess_annualized=0.12,
        beat_benchmark=True,
        excess_first_half=0.08,
        excess_second_half=0.06,
    )
    attach_coverage(result, n_months=10, n_covered=5)
    finalize_gate(result, settings)
    assert result.coverage_ratio == 0.5
    assert result.insufficient_data is True
    assert result.passed_gate is False
    assert result.gate_status == "数据不足"

    ok = BacktestResult(
        "demo",
        0.30,
        -0.10,
        1.5,
        0.6,
        0.2,
        excess_annualized=0.12,
        beat_benchmark=True,
        excess_first_half=0.08,
        excess_second_half=0.06,
    )
    attach_coverage(ok, n_months=10, n_covered=9)
    finalize_gate(ok, settings)
    assert ok.insufficient_data is False
    assert ok.passed_gate is True
    assert ok.gate_status == "通过"


def test_pick_coverage_counts_empty_months() -> None:
    dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")]
    picks = {dates[0]: ["AAA"], dates[1]: [], dates[2]: ["BBB"]}
    cov = pick_coverage(picks, dates)
    assert cov["n_months"] == 3
    assert cov["n_covered"] == 2
    assert cov["n_empty"] == 1
    assert abs(float(cov["coverage_ratio"]) - 2 / 3) < 1e-12
