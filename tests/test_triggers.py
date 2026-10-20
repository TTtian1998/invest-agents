"""触发线语义：比较符、buffer、连续交易日、缺数据、跳空合并、量比、14:55 预警、港股未收盘。"""

from __future__ import annotations

from datetime import date, datetime

from zhiku.calendar import SHANGHAI
from zhiku.engine.triggers import EvalMode, TriggerEvaluator
from zhiku.models import Market, SignalStatus, SymbolBook
from tests.conftest import MemoryProvider, bar


def _book(rules: list[dict], symbol: str = "510300", market: str = "CN", qty: float = 10000) -> SymbolBook:
    return SymbolBook.model_validate(
        {
            "symbol": symbol,
            "market": market,
            "name": "示例",
            "position": {"qty": qty, "cost": 4.0},
            "rules": rules,
        }
    )


def _eval(book: SymbolBook, bars, on: str, mode: EvalMode, **kwargs) -> list:
    ev = TriggerEvaluator(MemoryProvider(bars), **kwargs)
    return ev.evaluate_book(book, date.fromisoformat(on), mode).signals


def test_ops_lt_le_gt_ge() -> None:
    bars = [
        bar("X", "2026-10-19", 10.0, market=Market.CN),
        bar("X", "2026-10-20", 10.0, market=Market.CN),
    ]
    cases = [
        ("lt", 10.1, SignalStatus.TRIGGERED),
        ("lt", 10.0, SignalStatus.NOT_TRIGGERED),
        ("le", 10.0, SignalStatus.TRIGGERED),
        ("gt", 9.9, SignalStatus.TRIGGERED),
        ("gt", 10.0, SignalStatus.NOT_TRIGGERED),
        ("ge", 10.0, SignalStatus.TRIGGERED),
    ]
    for op, level, expected in cases:
        book = _book(
            [{"id": f"{op}-{level}", "when": {"field": "close", "op": op, "level": level},
              "action": {"type": "alert"}}],
            symbol="X",
        )
        sig = _eval(book, bars, "2026-10-20", EvalMode.CLOSE)[0]
        assert sig.status is expected, (op, level, sig.reason)


def test_buffer_pct_moves_effective_line() -> None:
    bars = [bar("510300", "2026-10-20", 3.78)]
    book = _book(
        [
            {
                "id": "buf",
                "when": {"field": "close", "op": "lt", "level": 3.80, "buffer_pct": 1.0},
                "action": {"type": "alert"},
            }
        ]
    )
    sig = _eval(book, bars, "2026-10-20", EvalMode.CLOSE)[0]
    # 有效线 = 3.80 * 0.99 = 3.762；3.78 未跌破
    assert abs(sig.effective_level - 3.762) < 1e-9
    assert sig.status is SignalStatus.NOT_TRIGGERED

    bars2 = [bar("510300", "2026-10-20", 3.75)]
    sig2 = _eval(book, bars2, "2026-10-20", EvalMode.CLOSE)[0]
    assert sig2.status is SignalStatus.TRIGGERED


def test_consecutive_two_skips_weekend() -> None:
    bars = [
        bar("510300", "2026-10-16", 3.75),  # 周五
        bar("510300", "2026-10-19", 3.76),  # 周一
    ]
    book = _book(
        [
            {
                "id": "demo-380-2d",
                "when": {"field": "close", "op": "lt", "level": 3.80, "consecutive": 2},
                "action": {"type": "sell_fraction", "fraction": 0.5},
            }
        ]
    )
    sig = _eval(book, bars, "2026-10-19", EvalMode.CLOSE)[0]
    assert sig.status is SignalStatus.TRIGGERED


def test_missing_bar_on_weekday_is_missing_data() -> None:
    # consecutive=2，评估周二；周一（交易日）缺 K 线 → 缺数据，不是未触发、也不重置
    bars = [
        bar("510300", "2026-10-16", 3.75),
        bar("510300", "2026-10-20", 3.76),
    ]
    book = _book(
        [
            {
                "id": "demo-380-2d",
                "when": {"field": "close", "op": "lt", "level": 3.80, "consecutive": 2},
                "action": {"type": "sell_fraction", "fraction": 0.5},
            }
        ]
    )
    sig = _eval(book, bars, "2026-10-20", EvalMode.CLOSE)[0]
    assert sig.status is SignalStatus.MISSING_DATA
    assert "缺数据" in sig.reason


def test_supersedes_gap_merges_to_20_shares() -> None:
    bars = [
        bar("AAPL", "2026-10-19", 184, open_=185, market=Market.US),
        bar("AAPL", "2026-10-20", 166, open_=165, market=Market.US),
    ]
    book = _book(
        [
            {
                "id": "aapl-line-a",
                "when": {"field": "close", "op": "lt", "level": 180.0},
                "action": {"type": "sell_qty", "qty": 10},
            },
            {
                "id": "aapl-line-b-close",
                "when": {"field": "close", "op": "lt", "level": 170.0},
                "action": {"type": "sell_qty", "qty": 20},
                "supersedes": ["aapl-line-a"],
            },
            {
                "id": "aapl-line-b-open",
                "when": {"field": "open", "op": "lt", "level": 170.0},
                "action": {"type": "sell_qty", "qty": 20},
                "supersedes": ["aapl-line-a"],
            },
        ],
        symbol="AAPL",
        market="US",
        qty=50,
    )
    ev = TriggerEvaluator(MemoryProvider(bars), replay=True).evaluate_book(
        book, date(2026, 10, 20), EvalMode.CLOSE
    )
    visible_ids = {s.rule_id for s in ev.visible if s.status is SignalStatus.TRIGGERED}
    assert "aapl-line-a" not in visible_ids
    assert ev.merged_qty == 20
    line_a = next(s for s in ev.signals if s.rule_id == "aapl-line-a")
    assert line_a.status is SignalStatus.TRIGGERED
    assert line_a.superseded_by in {"aapl-line-b-close", "aapl-line-b-open"}


def test_volume_ratio_and_condition() -> None:
    bars = [
        bar("0700.HK", "2026-10-19", 312, volume=1000, market=Market.HK),
        bar("0700.HK", "2026-10-20", 290, volume=1500, market=Market.HK),
    ]
    book = _book(
        [
            {
                "id": "hk0700-vol",
                "when": {"field": "close", "op": "lt", "level": 300.0},
                "and": [{"field": "volume_ratio", "op": "ge", "level": 1.0}],
                "action": {"type": "sell_fraction", "fraction": 0.5},
            }
        ],
        symbol="0700.HK",
        market="HK",
        qty=100,
    )
    sig = TriggerEvaluator(MemoryProvider(bars), replay=True).evaluate_book(
        book, date(2026, 10, 20), EvalMode.CLOSE
    ).signals[0]
    assert sig.status is SignalStatus.TRIGGERED
    assert sig.volume_ratio == 1.5

    bars_low = [
        bar("0700.HK", "2026-10-19", 312, volume=2000, market=Market.HK),
        bar("0700.HK", "2026-10-20", 290, volume=1000, market=Market.HK),
    ]
    sig2 = TriggerEvaluator(MemoryProvider(bars_low), replay=True).evaluate_book(
        book, date(2026, 10, 20), EvalMode.CLOSE
    ).signals[0]
    assert sig2.status is SignalStatus.NOT_TRIGGERED
    assert "附加条件未满足" in sig2.reason


def test_check_1455_marks_close_rules_preview() -> None:
    bars = [bar("510300", "2026-10-19", 3.90, price_1455=3.75)]
    book = _book(
        [
            {
                "id": "close-line",
                "when": {"field": "close", "op": "lt", "level": 3.80},
                "action": {"type": "sell_fraction", "fraction": 0.5},
            },
            {
                "id": "p1455",
                "when": {"field": "price_1455", "op": "lt", "level": 3.80},
                "action": {"type": "alert"},
            },
        ]
    )
    ev = TriggerEvaluator(MemoryProvider(bars)).evaluate_book(
        book, date(2026, 10, 19), EvalMode.CHECK_1455
    )
    by_id = {s.rule_id: s for s in ev.signals}
    assert by_id["close-line"].status is SignalStatus.PREVIEW
    assert "预警" in by_id["close-line"].reason
    assert by_id["p1455"].status is SignalStatus.TRIGGERED
    events = TriggerEvaluator(MemoryProvider(bars)).events_from([ev], date(2026, 10, 19))
    assert {e.rule_id for e in events} == {"p1455"}


def test_hk_not_closed_before_1610() -> None:
    bars = [
        bar("2800.HK", "2026-10-19", 17.80, market=Market.HK),
        bar("2800.HK", "2026-10-20", 17.50, market=Market.HK),
    ]
    book = _book(
        [
            {
                "id": "hk2800-2d",
                "when": {"field": "close", "op": "lt", "level": 18.00, "consecutive": 2},
                "action": {"type": "sell_all"},
            }
        ],
        symbol="2800.HK",
        market="HK",
        qty=2000,
    )
    now = datetime(2026, 10, 20, 15, 0, tzinfo=SHANGHAI)
    sig = TriggerEvaluator(MemoryProvider(bars), now=now, replay=False).evaluate_book(
        book, date(2026, 10, 20), EvalMode.CLOSE
    ).signals[0]
    assert sig.status is SignalStatus.NOT_CLOSED
    assert "未收盘" in sig.status_cn
    assert "16:10" in sig.reason

    later = datetime(2026, 10, 20, 16, 10, tzinfo=SHANGHAI)
    sig2 = TriggerEvaluator(MemoryProvider(bars), now=later, replay=False).evaluate_book(
        book, date(2026, 10, 20), EvalMode.CLOSE
    ).signals[0]
    assert sig2.status is SignalStatus.TRIGGERED
