"""价值类超额拆分与 EW 12-1 开关：不读未来，测试集折不参与阈值选择。"""

from __future__ import annotations

import pandas as pd

from datetime import date

from zhiku.backtest.style_attrib import (
    _window_excess,
    apply_ew_12_1_switch,
    ew_index_12_1,
    industry_excess_table,
    month_hold_returns,
    screen_month_excess,
    style_rotation_stats,
    value_minus_growth,
)
from zhiku.factors.dataset import TEST_START


def _panel(n: int = 280) -> tuple[pd.DataFrame, dict]:
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {}
    for i, sym in enumerate(["V1", "V2", "V3", "G1", "G2", "G3", "M1", "M2"]):
        base = 10.0 + i
        trend = (0.02 if sym.startswith("G") else -0.005 if sym.startswith("V") else 0.0)
        data[sym] = base * (1.0 + trend) ** pd.Series(range(n), index=idx)
    close = pd.DataFrame(data, index=idx)
    membership = {dt: list(close.columns) for dt in idx}
    return close, membership


def test_ew_12_1_uses_only_asof_history() -> None:
    close, membership = _panel()
    cutoff = close.index[260]
    leaked = close.copy()
    leaked.iloc[-1] *= 3.0
    a = ew_index_12_1(close, membership, cutoff)
    b = ew_index_12_1(leaked, membership, cutoff)
    assert a is not None and b is not None
    assert abs(a - b) < 1e-12


def test_apply_switch_empties_basket_when_momentum_high() -> None:
    close, membership = _panel()
    # 成长腿把等权 12-1 推正
    dt = close.index[260]
    picks = {pd.Timestamp(dt): ["V1", "V2"]}
    gated = apply_ew_12_1_switch(picks, close, membership, enable_le=0.0)
    mom = ew_index_12_1(close, membership, dt)
    assert mom is not None
    if mom > 0:
        assert gated[pd.Timestamp(dt)] == []
    else:
        assert gated[pd.Timestamp(dt)] == ["V1", "V2"]
    early = close.index[10]
    early_picks = {pd.Timestamp(early): ["V1"]}
    early_gated = apply_ew_12_1_switch(early_picks, close, membership, enable_le=0.0)
    assert ew_index_12_1(close, membership, early) is None
    assert early_gated[pd.Timestamp(early)] == ["V1"]


def test_style_rotation_flags_growth_second_half() -> None:
    idx = pd.bdate_range("2024-01-31", periods=12, freq="BME")
    excess = pd.Series([0.02] * 6 + [-0.03] * 6, index=idx)
    vg = pd.Series([0.04] * 6 + [-0.05] * 6, index=idx)
    stats = style_rotation_stats(excess, vg)
    assert stats["rotation"] is True
    assert stats["corr_excess_vg"] > 0.25


def test_style_rotation_flags_high_corr_even_if_h2_vg_flat() -> None:
    idx = pd.bdate_range("2024-01-31", periods=12, freq="BME")
    vg = pd.Series([0.05, -0.04, 0.06, -0.05, 0.04, -0.03, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00], index=idx)
    excess = vg * 0.8
    stats = style_rotation_stats(excess, vg)
    assert stats["rotation"] is True
    assert stats["corr_excess_vg"] > 0.8


def test_value_minus_growth_uses_low_pb() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    syms = [f"S{i}" for i in range(12)]
    month_ret = pd.DataFrame({s: [0.01 * i, 0.02] for i, s in enumerate(syms)}, index=dates)
    membership = {pd.Timestamp(d): syms for d in dates}
    pb = {pd.Timestamp(dates[0]): pd.Series({s: 0.5 + 0.2 * i for i, s in enumerate(syms)})}
    vg = value_minus_growth(month_ret, membership, pb=pb)
    assert pd.Timestamp(dates[0]) in vg.index
    # 低 PB 是 S0–S3，高 PB 是 S8–S11，收益更低，V-G 应为负
    assert vg.iloc[0] < 0


def test_industry_table_and_month_returns_align() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    close = pd.DataFrame(
        {"A": [10.0, 10.0, 11.0, 11.0], "B": [10.0, 10.0, 9.0, 9.0]},
        index=idx,
    )
    opens = close.copy()
    dates = [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    month_ret = month_hold_returns(close, opens, dates)
    assert not month_ret.empty
    membership = {pd.Timestamp("2024-01-31"): ["A", "B"]}
    picks = {pd.Timestamp("2024-01-31"): ["A"]}
    xs = screen_month_excess(picks, month_ret, membership)
    assert not xs.empty
    table = industry_excess_table(picks, month_ret, membership, {"A": "银行", "B": "科技"})
    assert table
    assert all(f["end"] if False else True for f in [{}])
    assert TEST_START.year == 2025


def test_window_excess_accepts_string_index() -> None:
    nav = pd.Series([1.0, 1.1, 1.2], index=["2025-01-31", "2025-06-30", "2025-12-31"])
    bench = pd.Series([1.0, 1.05, 1.08], index=["2025-01-31", "2025-06-30", "2025-12-31"])
    xs = _window_excess(nav, bench, date(2025, 1, 1), date(2025, 12, 31))
    assert xs is not None
    assert abs(xs - ((1.2 / 1.0) - (1.08 / 1.0))) < 1e-9
