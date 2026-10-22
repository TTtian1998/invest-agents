"""成长仓暴露闸门：滞回 0.62→FROZEN，0.55→HOLD，0.49→OK。"""

from __future__ import annotations

from datetime import date

from zhiku.engine.exposure import STATUS_FROZEN, STATUS_HOLD, STATUS_OK, ExposureGuard
from zhiku.engine.rules import load_portfolio
from zhiku.models import Market, Quote
from zhiku.data.base import asof_quote


def test_hysteresis(tmp_path) -> None:
    guard = ExposureGuard(tmp_path / "exposure_state.json")
    s1 = guard.evaluate_ratio(0.62)
    guard.save(s1, date(2026, 10, 20))
    assert s1.status == STATUS_FROZEN
    assert s1.frozen is True

    s2 = guard.evaluate_ratio(0.55)
    guard.save(s2, date(2026, 10, 21))
    assert s2.status == STATUS_HOLD
    assert s2.frozen is True

    s3 = guard.evaluate_ratio(0.49)
    guard.save(s3, date(2026, 10, 22))
    assert s3.status == STATUS_OK
    assert s3.frozen is False


def test_middle_band_from_ok_stays_ok(tmp_path) -> None:
    guard = ExposureGuard(tmp_path / "exposure_state.json")
    s1 = guard.evaluate_ratio(0.40)
    guard.save(s1, date(2026, 10, 20))
    assert s1.status == STATUS_OK
    s2 = guard.evaluate_ratio(0.55)
    assert s2.status == STATUS_OK


def test_portfolio_add_blocked_and_cny(tmp_path, examples_dir) -> None:
    portfolio = load_portfolio(examples_dir / "kb" / "portfolio" / "positions.yaml")
    quotes: dict[str, Quote] = {
        "510300": asof_quote("510300", date(2026, 10, 23), 3.62, "close", Market.CN),
        "600519": asof_quote("600519", date(2026, 10, 23), 1480, "close", Market.CN),
        "AAPL": asof_quote("AAPL", date(2026, 10, 23), 164, "close", Market.US),
        "0700.HK": asof_quote("0700.HK", date(2026, 10, 23), 305, "close", Market.HK),
        "2800.HK": asof_quote("2800.HK", date(2026, 10, 23), 17.10, "close", Market.HK),
    }
    guard = ExposureGuard(tmp_path / "state.json")
    snap = guard.evaluate(portfolio, quotes, persist=True, asof=date(2026, 10, 23))
    assert snap.ratio is not None
    assert 0.3 < snap.ratio < 0.5
    assert snap.status == STATUS_OK
    blocked = [p for p in snap.positions if p.add_blocked]
    assert [p.symbol for p in blocked] == ["510300"]
