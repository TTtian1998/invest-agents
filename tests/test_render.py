"""模板能渲染四种报告，缺数据字样与关键字段都在。"""

from __future__ import annotations

from datetime import date, datetime

from zhiku.calendar import SHANGHAI
from zhiku.engine.exposure import ExposureSnapshot, PositionValue, STATUS_OK
from zhiku.models import (
    Action,
    HitLabel,
    Market,
    Signal,
    SignalStatus,
    SymbolBook,
    TriggerEvent,
)
from zhiku.notify.render import render
from zhiku.review.hitrate import GroupStat, HitRecord, HorizonOutcome, WeekReview
from zhiku.engine.triggers import SymbolEvaluation


def _book() -> SymbolBook:
    return SymbolBook.model_validate(
        {
            "symbol": "510300",
            "market": "CN",
            "name": "示例宽基ETF",
            "add_blocked": True,
            "position": {"qty": 10000, "cost": 4.0},
            "rules": [
                {
                    "id": "demo-380-2d",
                    "when": {"field": "close", "op": "lt", "level": 3.8},
                    "action": {"type": "alert"},
                }
            ],
        }
    )


def test_check_and_close_templates_include_evidence() -> None:
    book = _book()
    sig = Signal(
        rule_id="demo-380-2d",
        symbol="510300",
        market=Market.CN,
        status=SignalStatus.MISSING_DATA,
        actual=None,
        effective_level=3.8,
        price_source="close",
        asof=datetime(2026, 10, 20, 15, 0, tzinfo=SHANGHAI),
        reason="无日线",
        action=Action(type="alert"),
    )
    ev = SymbolEvaluation(book=book, signals=[sig], merged_action=None, merged_qty=None)
    check = render(
        "check.md.j2",
        on="2026-10-19",
        provider="fixture",
        kb="/tmp/kb",
        evaluations=[ev],
        warnings=["A 股交易日历不可用，退化为工作日"],
    )
    assert "缺数据" in check
    assert "有效线" in check
    assert "510300" in check
    assert "禁加" in check
    close = render(
        "close.md.j2",
        on="2026-10-20",
        provider="fixture",
        kb="/tmp/kb",
        evaluations=[ev],
        warnings=[],
    )
    assert "未收盘" in close or "缺数据" in close
    assert "收盘确认" in close


def test_exposure_and_weekly_templates() -> None:
    snap = ExposureSnapshot(
        status=STATUS_OK,
        ratio=0.4,
        growth_cny=100.0,
        total_cny=250.0,
        freeze_threshold=0.6,
        unfreeze_threshold=0.5,
        frozen=False,
        positions=[
            PositionValue(
                "510300", "示例", Market.CN, 10000, 3.62, 1.0, 36200, False, True, None
            )
        ],
        notes=[],
    )
    expo = render("exposure.md.j2", on="2026-10-23", snap=snap)
    assert "FROZEN" not in expo or "OK" in expo
    assert "禁加" in expo
    assert "510300" in expo

    event = TriggerEvent(
        rule_id="demo-370-1d",
        symbol="510300",
        market=Market.CN,
        date=date(2026, 10, 20),
        trigger_price=3.65,
        effective_level=3.70,
        action=Action(type="sell_fraction", fraction=0.5),
    )
    rec = HitRecord(
        event=event,
        outcomes=[
            HorizonOutcome(1, date(2026, 10, 21), 3.60, HitLabel.HIT),
            HorizonOutcome(3, date(2026, 10, 23), 3.62, HitLabel.HIT),
        ],
    )
    weekly = render(
        "weekly.md.j2",
        provider="fixture",
        kb="/tmp/kb",
        review=WeekReview(
            start=date(2026, 10, 19),
            end=date(2026, 10, 23),
            records=[rec],
            groups=[GroupStat("CN", "sell_fraction", 1, 1, 0, 0, 0)],
            lessons=["本周 1 次卖出触发在 T+1 兑现"],
        ),
    )
    assert "兑现" in weekly
    assert "候选教训" in weekly
    assert "不会自动写入" in weekly
