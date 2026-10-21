"""标签、中性化、分割：不看未来公告，测试集不进训练。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from zhiku.data.store import MarketStore
from zhiku.factors.dataset import (
    TEST_START,
    TRAIN_END,
    VALID_END,
    assign_split,
    excess_vs_pool,
    label_open_horizon,
    label_t2_close,
    load_fund_features,
    neutralize,
    split_of,
)
from zhiku.factors.dataset import make_ch_features
from zhiku.factors.experiment import (
    BASELINE,
    CANDIDATES,
    ExperimentConfig,
    ExperimentResult,
    changed_fields,
    decide_keep,
)


def test_t2_label_is_next_day_to_day_after() -> None:
    bars = pd.DataFrame(
        [
            {"date": "2026-10-19", "symbol": "AAA", "open": 10.0, "close": 10.0},
            {"date": "2026-10-20", "symbol": "AAA", "open": 11.0, "close": 12.0},
            {"date": "2026-10-21", "symbol": "AAA", "open": 12.0, "close": 13.0},
        ]
    )
    lab = label_t2_close(bars)
    val = float(lab.xs(pd.Timestamp("2026-10-19"), level="date").iloc[0])
    assert abs(val - (13.0 / 12.0 - 1.0)) < 1e-9


def test_t20_excess_subtracts_same_day_mean() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"] + ["2024-04-30"] * 1)
    # simpler: two names, known 5-day path
    dates = pd.bdate_range("2024-01-02", periods=25)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "symbol": "AAA", "open": 10.0 + i, "close": 10.0 + i})
        rows.append({"date": d, "symbol": "BBB", "open": 20.0, "close": 20.0})
    bars = pd.DataFrame(rows)
    raw = label_open_horizon(bars, 5)
    xs = excess_vs_pool(raw)
    day = raw.index.get_level_values("date")[0]
    part = xs.xs(day, level="date")
    assert abs(float(part.mean())) < 1e-9


def test_neutralize_industry_mean_zero() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-31"]), ["A", "B", "C", "D"]], names=["date", "symbol"]
    )
    y = pd.Series([2.0, 4.0, 10.0, 12.0], index=idx, name="label")
    industry = pd.Series(["X", "X", "Y", "Y"], index=idx)
    out = neutralize(y, industry=industry, size=None)
    # X: 2,4 mean 3 → -1, +1; Y: 10,12 mean 11 → -1,+1
    assert abs(float(out.iloc[0]) + 1.0) < 1e-9
    assert abs(float(out.iloc[1]) - 1.0) < 1e-9


def test_split_is_frozen_and_ordered() -> None:
    assert split_of(pd.Timestamp(TRAIN_END)) == "train"
    assert split_of(pd.Timestamp("2025-07-01")) == "valid"
    assert split_of(pd.Timestamp(TEST_START)) == "test"
    assert TRAIN_END < VALID_END < TEST_START
    s = assign_split(pd.Series(pd.to_datetime(["2024-01-01", "2025-06-01", "2026-01-01"])))
    assert list(s) == ["train", "valid", "test"]


def test_fund_features_ignore_future_filings(tmp_path) -> None:
    store = MarketStore(tmp_path / "data")
    store.upsert_listings([{"symbol": "000333", "market": "CN", "industry": "C 制造业", "instrument": "stock"}])
    store.upsert_statements(
        [
            {
                "symbol": "000333",
                "kind": "income",
                "period": date(2024, 12, 31),
                "announced": date(2025, 3, 31),
                "payload": {"净利润": 10.0, "营业总收入": 100.0, "营业成本": 60.0, "每股收益": 1.0},
            },
            {
                "symbol": "000333",
                "kind": "income",
                "period": date(2025, 6, 30),
                "announced": date(2025, 8, 31),
                "payload": {"净利润": 80.0, "营业总收入": 200.0, "营业成本": 60.0, "每股收益": 8.0},
            },
            {
                "symbol": "000333",
                "kind": "balance",
                "period": date(2024, 12, 31),
                "announced": date(2025, 3, 31),
                "payload": {"股东权益合计": 50.0, "资产总计": 100.0, "负债合计": 50.0},
            },
        ]
    )
    cal = pd.to_datetime(["2025-04-30", "2025-09-01"])
    panel = load_fund_features(store, ["000333"], cal)
    early = panel.xs(pd.Timestamp("2025-04-30"), level="date").iloc[0]
    late = panel.xs(pd.Timestamp("2025-09-01"), level="date").iloc[0]
    assert early["f_roe"] is not None and abs(float(early["f_roe"]) - 0.2) < 1e-9
    assert late["f_roe"] is not None and float(late["f_roe"]) > float(early["f_roe"])


def test_keep_decision_uses_valid_only() -> None:
    base = ExperimentResult(
        name="base",
        valid={"t20_xs": {"rank_icir": 0.10, "rank_ic": 0.02}},
        test={"t20_xs": {"rank_icir": 0.99, "rank_ic": 0.5}},
    )
    worse_test_better_valid = ExperimentResult(
        name="x",
        valid={"t20_xs": {"rank_icir": 0.20, "rank_ic": 0.03}},
        test={"t20_xs": {"rank_icir": 0.01, "rank_ic": 0.0}},
    )
    better_test_worse_valid = ExperimentResult(
        name="y",
        valid={"t20_xs": {"rank_icir": 0.09, "rank_ic": 0.02}},
        test={"t20_xs": {"rank_icir": 0.80, "rank_ic": 0.4}},
    )
    assert decide_keep(base, worse_test_better_valid) is True
    assert decide_keep(base, better_test_worse_valid) is False


def test_ch_features_flip_reversal_turnover_vol() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-31"]), ["AAA"]], names=["date", "symbol"]
    )
    px = pd.DataFrame(
        {"ret_5": [0.10], "ret_20": [0.20], "turnover": [3.0], "std_20": [0.05]},
        index=idx,
    )
    ch = make_ch_features(px)
    assert float(ch["ch_rev_5"].iloc[0]) == -0.10
    assert float(ch["ch_low_turn"].iloc[0]) == -3.0
    assert float(ch["ch_low_vol"].iloc[0]) == -0.05


def test_candidates_change_one_variable() -> None:
    parents = {BASELINE.name: BASELINE}
    for cfg in CANDIDATES:
        parent = parents.get(cfg.versus or BASELINE.name, BASELINE)
        delta = changed_fields(parent, cfg)
        if cfg.name == "univ_alla":
            assert "universe" in delta
            assert set(delta) <= {"universe", "freq"}
        elif cfg.name == "model_rank":
            assert delta == ["model"]
        else:
            assert len(delta) == 1, (cfg.name, delta)
        parents[cfg.name] = cfg


def test_run_one_valid_only_leaves_test_empty(tmp_path) -> None:
    """无行情时 run_one 仍不得填写测试集指标。"""
    from zhiku.factors.experiment import run_one

    store = MarketStore(tmp_path / "data")
    res, _, _ = run_one(store, BASELINE, asof=date(2026, 10, 21), report_splits=("valid",))
    assert res.test == {}
    assert res.n_test == 0
