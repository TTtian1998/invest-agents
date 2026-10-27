"""Point-in-time invariants for the deterministic ETF pilot."""
from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from zhiku.strategies import StrategySpec, generate_decision, schedule_decisions


def spec(**kwargs):
    return StrategySpec(**{"universe": ["B", "A", "C"],
                           "universe_known_at": date(2020, 1, 1), "lookback": 2, **kwargs})


def prices():
    return pd.DataFrame({"A": [100., 110., 120., 130.],
                         "B": [100., 110., 120., 140.],
                         "C": [100., 90., 80., 70.]},
                        index=pd.bdate_range("2020-01-01", periods=4))


def test_stable_ties_caps_cash_and_frozen_spec():
    config = spec(top_n=1)
    decision = generate_decision(prices(), "2020-01-03", config)
    assert decision.target_weights == {"A": .3}
    assert decision.cash_weight == .7
    assert decision.evidence["below_momentum_threshold"] == ["C"]
    assert config.universe == ("A", "B", "C")
    with pytest.raises(ValidationError):
        config.lookback = 9
    assert decision.model_dump(mode="json")["asof"] == "2020-01-03"


def test_future_prices_ignored_including_fingerprint():
    frame = prices()
    original = generate_decision(frame, "2020-01-03", spec())
    frame.loc["2020-01-06", :] = float("inf")
    assert generate_decision(frame, "2020-01-03", spec()) == original
    assert generate_decision(frame.iloc[:3], "2020-01-03", spec()) == original


def test_equal_weight_uses_same_complete_universe_without_momentum_filter():
    frame = prices()
    momentum = generate_decision(frame, "2020-01-03", spec())
    passive = generate_decision(frame, "2020-01-03", spec(), "equal_weight")
    assert momentum.evidence["eligible_symbols"] == passive.evidence["eligible_symbols"]
    assert passive.target_weights == {"A": .8 / 3, "B": .8 / 3, "C": .8 / 3}
    assert passive.cash_weight == pytest.approx(.2)
    assert momentum.cash_weight == .4


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, "bad"])
def test_invalid_window_price_excluded_explicitly(bad):
    frame = prices().astype(object)
    frame.loc["2020-01-02", "A"] = bad
    d = generate_decision(frame, "2020-01-03", spec())
    assert d.evidence["exclusions"]["A"] == "missing_nonfinite_or_nonpositive_price"
    assert "A" not in d.signals and "A" not in d.target_weights


def test_missing_stale_and_insufficient_prices_never_filled():
    d = generate_decision(prices().drop(columns="A"), "2020-01-03", spec())
    assert d.evidence["exclusions"]["A"] == "missing_column"
    stale = generate_decision(prices(), "2020-01-07", spec())
    assert stale.target_weights == {} and stale.cash_weight == 1
    assert set(stale.evidence["exclusions"].values()) == {"stale_or_missing_asof_session"}
    early = generate_decision(prices(), "2020-01-02", spec())
    assert set(early.evidence["exclusions"].values()) == {"insufficient_window"}


@pytest.mark.parametrize("kind", ["duplicate", "unordered", "timezone", "intraday", "range"])
def test_bad_dates_rejected(kind):
    frame = prices()
    if kind == "duplicate":
        frame.index = [frame.index[0]] * len(frame)
    elif kind == "unordered":
        frame = frame.iloc[::-1]
    elif kind == "timezone":
        frame.index = frame.index.tz_localize("UTC")
    elif kind == "intraday":
        frame.index = frame.index + pd.Timedelta(hours=1)
    else:
        frame.index = pd.RangeIndex(len(frame))
    with pytest.raises(ValueError):
        generate_decision(frame, "2020-01-03", spec())


@pytest.mark.parametrize("kwargs", [
    {"universe": []}, {"universe": ["A", "A"]}, {"universe": [" A"]},
    {"lookback": 0}, {"top_n": 0}, {"rebalance_every": 0}, {"lot_size": 0},
    {"cost_bps": -1}, {"slippage_bps": -1}, {"max_weight": 1.1},
    {"max_gross": 1.1}, {"initial_cash": 0}, {"source_mode": "live"},
    {"min_momentum": float("nan")}, {"lookback": 1.5}, {"unknown": 1},
])
def test_invalid_spec(kwargs):
    with pytest.raises(ValidationError):
        spec(**kwargs)


def test_known_at_guard_and_parameter_fingerprint():
    with pytest.raises(ValueError, match="universe_known_at"):
        generate_decision(prices(), "2020-01-03", spec(universe_known_at="2021-01-01"))
    d = generate_decision(prices(), "2020-01-03", spec())
    changed = generate_decision(prices(), "2020-01-03", spec(top_n=1))
    assert changed.spec_fingerprint != d.spec_fingerprint
    assert changed.fingerprint != d.fingerprint
    assert spec(universe=["C", "A", "B"]).fingerprint() == spec().fingerprint()


def test_schedule_anchors_on_start_not_warmup():
    decisions = schedule_decisions(prices(), spec(rebalance_every=2), "2020-01-02", "2020-01-06")
    assert [d.asof.isoformat() for d in decisions] == ["2020-01-02", "2020-01-06"]
