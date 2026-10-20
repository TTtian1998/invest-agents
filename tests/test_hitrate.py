"""命中率：兑现 / 误报 / 中性 / 待定，以及按市场分组的百分比。"""

from __future__ import annotations

from datetime import date

from zhiku.models import Action, HitLabel, Market, TriggerEvent
from zhiku.review.hitrate import classify_event, group_stats, review_events
from tests.conftest import MemoryProvider, bar


def _event(
    symbol: str,
    market: Market,
    d: str,
    trigger: float,
    level: float,
    *,
    rule_id: str = "r1",
    action: str = "sell_qty",
) -> TriggerEvent:
    kwargs = {"type": action}
    if action == "sell_qty":
        kwargs["qty"] = 10
    elif action == "sell_fraction":
        kwargs["fraction"] = 0.5
    return TriggerEvent(
        rule_id=rule_id,
        symbol=symbol,
        market=market,
        date=date.fromisoformat(d),
        trigger_price=trigger,
        effective_level=level,
        action=Action.model_validate(kwargs),
    )


def test_four_classifications() -> None:
    event = _event("510300", Market.CN, "2026-10-20", trigger=3.65, level=3.70)
    assert classify_event(event, 3.60) is HitLabel.HIT
    assert classify_event(event, 3.80) is HitLabel.FALSE_ALARM  # 3.70 * 1.01 = 3.737
    assert classify_event(event, 3.68) is HitLabel.NEUTRAL
    assert classify_event(event, None) is HitLabel.PENDING


def test_group_percentages_by_market() -> None:
    events = [
        _event("510300", Market.CN, "2026-10-20", 3.65, 3.70, rule_id="cn-a"),
        _event("600519", Market.CN, "2026-10-20", 1410, 1400, rule_id="cn-b"),
        _event("AAPL", Market.US, "2026-10-20", 166, 170, rule_id="us-a"),
    ]
    provider = MemoryProvider(
        [
            bar("510300", "2026-10-21", 3.60),
            bar("600519", "2026-10-21", 1450),  # 误报：>= 1400*1.01
            bar("AAPL", "2026-10-21", 160, market=Market.US),
            bar("510300", "2026-10-23", 3.62),
            bar("600519", "2026-10-23", 1480),
            bar("AAPL", "2026-10-23", 164, market=Market.US),
        ]
    )
    records = review_events(events, provider, horizons=(1, 3))
    stats = group_stats(records, horizon=1)
    by_mkt = {s.market: s for s in stats}
    assert by_mkt["CN"].total == 2
    assert by_mkt["CN"].hit == 1
    assert by_mkt["CN"].false_alarm == 1
    assert by_mkt["CN"].hit_rate == 0.5
    assert by_mkt["US"].hit == 1
    assert by_mkt["US"].hit_rate == 1.0

    pending_event = _event("2800.HK", Market.HK, "2026-10-20", 17.5, 18.0, rule_id="hk-a")
    records2 = review_events([pending_event], MemoryProvider([]), horizons=(1,))
    assert records2[0].outcomes[0].label is HitLabel.PENDING
    stats2 = group_stats(records2, horizon=1)
    assert stats2[0].pending == 1
    assert stats2[0].hit_rate is None
