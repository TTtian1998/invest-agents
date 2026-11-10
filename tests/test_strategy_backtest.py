"""Synthetic accounting fixtures; no providers or network calls."""
import json

import pandas as pd
import pytest

from zhiku.strategies.backtest import backtest_targets


def panel(a, b=None):
    data = {"A": a}
    if b is not None:
        data["B"] = b
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=len(a)))


def run(closes, opens, targets, **kwargs):
    return backtest_targets(closes, opens, targets, initial_cash=10000,
                            cost_bps=0, slippage_bps=0, **kwargs)


def test_drift_daily_marks_and_final_tail():
    close = panel([10, 10, 20, 30], [10, 10, 10, 10])
    result = run(close, panel([10] * 4, [10] * 4),
                 {close.index[0]: {"A": .5, "B": .5}})
    assert result.nav.tolist() == [10000, 10000, 15000, 20000]
    assert result.holdings["A"].tolist() == [0, 500, 500, 500]
    assert result.holdings["B"].tolist() == [0, 500, 500, 500]
    assert result.turnover == 1
    assert result.metrics["total_return"] == 1
    assert result.metrics["passed_gate"] is False
    assert result.metrics["gate_status"] == "RESEARCH_ONLY"
    json.dumps(result.as_dict(), allow_nan=False)


def test_next_open_not_signal_close_and_final_session_execution():
    close = panel([100, 100, 100])
    opens = panel([10, 20, 25])
    result = run(close, opens, {close.index[1]: {"A": 1}})
    assert len(result.trades) == 1
    assert result.trades[0]["date"] == "2024-01-03"
    assert result.trades[0]["price"] == 25
    assert result.nav.tolist() == [10000, 10000, 40000]


def test_missing_open_never_fills_at_close_or_retries():
    close = panel([10, 1, 1])
    result = run(close, panel([10, float("nan"), 1]), {close.index[0]: {"A": 1}})
    assert result.nav.tolist() == [10000] * 3
    assert result.trades == []
    assert result.rejections[0]["reason"] == "missing_open"


def test_missing_held_open_keeps_quantity_and_marks_at_close():
    close = panel([10, 10, 20])
    result = run(close, panel([10, 10, float("nan")]),
                 {close.index[0]: {"A": 1}, close.index[1]: {}})
    assert result.nav.tolist() == [10000, 10000, 20000]
    assert len(result.trades) == 1
    assert result.holdings.iloc[-1]["A"] == 1000


def test_missing_held_close_fails_explicitly():
    close = panel([10, 10, float("nan")])
    with pytest.raises(ValueError, match="held close"):
        run(close, panel([10, 10, 10]), {close.index[0]: {"A": 1}})


def test_limit_up_blocks_add_to_existing_and_limit_down_blocks_sale():
    close = panel([10] * 4)
    result = run(close, close, {close.index[0]: {"A": .5},
                               close.index[1]: {"A": 1}, close.index[2]: {}},
                 limit_up=panel([False, False, True, False]),
                 limit_down=panel([False, False, False, True]))
    assert result.holdings["A"].tolist() == [0, 500, 500, 500]
    assert [r["reason"] for r in result.rejections] == ["limit_up", "limit_down"]


def test_limit_masks_are_side_aware():
    close = panel([10] * 3)
    result = run(close, close, {close.index[0]: {"A": 1}, close.index[1]: {}},
                 limit_up=panel([True, False, True]),
                 limit_down=panel([True, True, False]))
    assert [t["side"] for t in result.trades] == ["buy", "sell"]
    assert result.cash.iloc[-1] == 10000


def test_suspension_and_unknown_mask_are_fail_closed():
    close = panel([10] * 3)
    result = run(close, close, {close.index[0]: {"A": 1}, close.index[1]: {"A": 1}},
                 tradable=panel([True, False, None]))
    assert not result.trades
    assert [r["reason"] for r in result.rejections] == ["not_tradable"] * 2


def test_whole_lots_cash_costs_and_slippage_on_both_sides():
    close = panel([10] * 3)
    result = backtest_targets(close, close, {close.index[0]: {"A": 1}, close.index[1]: {}},
                              initial_cash=10000, cost_bps=100, slippage_bps=100)
    buy, sell = result.trades
    assert buy["quantity"] == sell["quantity"] == 900
    assert buy["price"] == pytest.approx(10.1)
    assert sell["price"] == pytest.approx(9.9)
    assert buy["fee"] == pytest.approx(90.9)
    assert sell["fee"] == pytest.approx(89.1)
    assert result.cash.min() >= 0
    assert result.nav.iloc[-1] == pytest.approx(9640)
    assert result.holdings.iloc[-1]["A"] == 0
    assert result.turnover > 1.8  # Both legs counted; second equity includes first costs.


def test_sells_fund_buys_without_same_day_roundtrip():
    close = panel([10] * 3, [20] * 3)
    result = run(close, close, {close.index[0]: {"A": 1}, close.index[1]: {"B": 1}})
    assert [(t["symbol"], t["side"]) for t in result.trades] == [
        ("A", "buy"), ("A", "sell"), ("B", "buy")]
    assert result.holdings.iloc[-1].to_dict() == {"A": 0, "B": 500}
    assert result.nav.tolist() == [10000] * 3


def test_colliding_decisions_latest_wins_no_same_day_sell_of_purchase():
    close = panel([10, 10]).rename(index={pd.Timestamp("2024-01-02"): pd.Timestamp("2024-01-04")})
    result = run(close, close, {"2024-01-02": {"A": 1}, "2024-01-03": {}})
    assert not result.trades
    assert result.rejections[0]["reason"] == "superseded_signal"


def test_future_close_cannot_change_earlier_orders():
    close = panel([10] * 4)
    changed = panel([10, 999, 999, 999])
    signals = {close.index[0]: {"A": 1}}
    assert run(close, close, signals).trades == run(changed, close, signals).trades


@pytest.mark.parametrize("weights", [{"A": -1}, {"A": float("nan")}, {"A": 1.01}, {"X": .5}])
def test_bad_targets_rejected(weights):
    close = panel([10, 10])
    with pytest.raises(ValueError):
        run(close, close, {close.index[0]: weights})


def test_empty_and_all_cash_portfolios():
    close = panel([10, 10])
    assert run(close, close, {}).nav.tolist() == [10000, 10000]
    empty = close.iloc[:0]
    result = run(empty, empty, {})
    assert result.nav.empty
    assert result.metrics["total_return"] is None


def test_total_weight_validation_and_explicit_cash_residual():
    close = panel([10, 10], [10, 10])
    with pytest.raises(ValueError, match="sum to at most 1"):
        run(close, close, {close.index[0]: {"A": .6, "B": .6}})
    result = run(close, close, {close.index[0]: {"A": .25, "B": .25}})
    assert result.holdings.iloc[-1].to_dict() == {"A": 200, "B": 200}
    assert result.cash.iloc[-1] == 6000
    assert result.nav.iloc[-1] == 10000


def test_sorted_allocation_is_deterministic_and_inputs_unchanged():
    close = panel([10, 10], [10, 10])
    original = close.copy(deep=True)
    targets = {close.index[0]: {"B": .5, "A": .5}}
    first = backtest_targets(close, close, targets, initial_cash=10000, cost_bps=100,
                              slippage_bps=0)
    second = backtest_targets(close[["B", "A"]], close[["B", "A"]],
                               {close.index[0]: {"A": .5, "B": .5}},
                               initial_cash=10000, cost_bps=100, slippage_bps=0)
    assert first.as_dict() == second.as_dict()
    assert first.holdings.iloc[-1].to_dict() == {"A": 500, "B": 400}
    pd.testing.assert_frame_equal(close, original)
    assert targets[close.index[0]] == {"B": .5, "A": .5}


def test_daily_accounting_identity():
    close = panel([10, 11, 12, 9, 15], [20, 21, 19, 22, 24])
    result = backtest_targets(close, close,
                              {close.index[0]: {"A": .5, "B": .4},
                               close.index[2]: {"A": .2, "B": .6}}, initial_cash=10000)
    marked = result.cash + (result.holdings * close).sum(axis=1)
    pd.testing.assert_series_equal(result.nav, marked.rename("nav"))
    assert (result.cash >= 0).all()
    assert (result.holdings.to_numpy() % 100 == 0).all()


def test_first_session_execution_cost_is_in_metric_baseline():
    close = panel([10, 10])
    result = backtest_targets(close, close, {"2023-12-31": {"A": 1}},
                              initial_cash=10000, cost_bps=100, slippage_bps=0)
    assert result.nav.tolist() == [9910, 9910]
    assert result.metrics["total_return"] == pytest.approx(-.009)
    assert result.metrics["max_drawdown"] == pytest.approx(-.009)
    assert result.metrics["annualized"] < 0
    assert result.metrics["metric_baseline_date"] == "2023-12-31"
    assert len(result.nav) == len(close)
    json.dumps(result.as_dict(), allow_nan=False)


def test_yearly_return_includes_new_year_boundary_return():
    close = pd.DataFrame({"A": [10, 10, 20, 20]},
                         index=pd.to_datetime(["2023-12-28", "2023-12-29",
                                               "2024-01-02", "2024-01-03"]))
    result = run(close, close, {"2023-12-28": {"A": 1}})
    assert result.metrics["years"] == {"2023": 0.0, "2024": 1.0}
    assert result.nav.tolist() == [10000, 10000, 20000, 20000]
