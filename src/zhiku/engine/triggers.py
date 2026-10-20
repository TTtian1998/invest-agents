"""触发线判定：纯 Python、确定性，LLM 不进入执行路径。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from zhiku.calendar import (
    EASTERN,
    HK_CLOSE,
    SHANGHAI,
    is_trading_day,
    now_shanghai,
    session_complete,
    trading_days_on_or_before,
)
from zhiku.data.base import MarketDataProvider
from zhiku.models import (
    Action,
    Bar,
    Condition,
    Market,
    MissingData,
    Rule,
    Signal,
    SignalStatus,
    SymbolBook,
    TriggerEvent,
)


class EvalMode(str, Enum):
    CHECK_1455 = "1455"
    CLOSE = "close"


def compare(op: str, actual: float, threshold: float) -> bool:
    if op == "lt":
        return actual < threshold
    if op == "le":
        return actual <= threshold
    if op == "gt":
        return actual > threshold
    if op == "ge":
        return actual >= threshold
    raise ValueError(f"未知比较符 {op}")


def _bar_map(bars: list[Bar]) -> dict[date, Bar]:
    return {b.date: b for b in bars}


def _asof(on: date, hhmm: time, tz: ZoneInfo = SHANGHAI) -> datetime:
    return datetime.combine(on, hhmm, tzinfo=tz)


@dataclass
class SymbolEvaluation:
    book: SymbolBook
    signals: list[Signal]
    merged_action: Action | None
    merged_qty: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def visible(self) -> list[Signal]:
        return [s for s in self.signals if s.superseded_by is None]


class TriggerEvaluator:
    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        now: datetime | None = None,
        replay: bool = False,
    ) -> None:
        self.provider = provider
        self.now = now_shanghai(now)
        self.replay = replay

    def _history(self, book: SymbolBook, on: date, lookback: int = 20) -> list[Bar] | MissingData:
        start = on - timedelta(days=lookback * 3)
        return self.provider.get_daily_bars(book.symbol, start, on, market=book.market)

    def _field_value(
        self,
        book: SymbolBook,
        bar: Bar | None,
        field: str,
        on: date,
        mode: EvalMode,
    ) -> float | MissingData:
        if field == "volume_ratio":
            return MissingData("volume_ratio 需单独计算", symbol=book.symbol, field=field, on=on)
        if field == "price_1455":
            if bar is not None and bar.price_1455 is not None:
                return float(bar.price_1455)
            return self.provider.get_price_1455(book.symbol, on)
        if mode is EvalMode.CHECK_1455 and field == "close":
            if bar is not None and bar.price_1455 is not None:
                return float(bar.price_1455)
            return self.provider.get_price_1455(book.symbol, on)
        if bar is None:
            return MissingData("无日线", symbol=book.symbol, field=field, on=on)
        return bar.value_of(field)

    def _volume_ratio(
        self, bars: dict[date, Bar], on: date, market: Market, symbol: str
    ) -> float | MissingData:
        today = bars.get(on)
        if today is None:
            return MissingData("无当日成交量", symbol=symbol, field="volume_ratio", on=on)
        prev_days = trading_days_on_or_before(market, on, 2)
        if len(prev_days) < 2:
            return MissingData("无前一交易日", symbol=symbol, field="volume_ratio", on=on)
        prev = bars.get(prev_days[-2])
        if prev is None:
            return MissingData("无前一交易日成交量", symbol=symbol, field="volume_ratio", on=on)
        if prev.volume <= 0:
            return MissingData("前一交易日成交量为 0，拒绝估算量比", symbol=symbol, on=on)
        return today.volume / prev.volume

    def _eval_condition(
        self,
        book: SymbolBook,
        condition: Condition,
        on: date,
        bars: dict[date, Bar],
        mode: EvalMode,
    ) -> tuple[SignalStatus, float | None, str]:
        days = trading_days_on_or_before(book.market, on, condition.consecutive)
        if len(days) < condition.consecutive:
            return (
                SignalStatus.MISSING_DATA,
                None,
                f"交易日不足 consecutive={condition.consecutive}",
            )
        threshold = condition.effective_level()
        last_actual: float | None = None
        for d in days:
            bar = bars.get(d)
            if bar is None and is_trading_day(book.market, d):
                return (
                    SignalStatus.MISSING_DATA,
                    None,
                    f"{d.isoformat()} 缺数据（交易日无 K 线，不估算）",
                )
            actual = self._field_value(book, bar, condition.field, d, mode)
            if isinstance(actual, MissingData):
                return SignalStatus.MISSING_DATA, None, f"{d.isoformat()} {actual.reason}"
            last_actual = actual
            if not compare(condition.op, actual, threshold):
                return (
                    SignalStatus.NOT_TRIGGERED,
                    last_actual,
                    f"{d.isoformat()} {condition.field}={actual} 未满足 {condition.op} {threshold}",
                )
        return (
            SignalStatus.TRIGGERED,
            last_actual,
            f"最近 {condition.consecutive} 个交易日均满足 "
            f"{condition.field} {condition.op} {threshold}",
        )

    def _hk_unclosed(self, book: SymbolBook, on: date, mode: EvalMode) -> bool:
        if book.market is not Market.HK or mode is not EvalMode.CLOSE:
            return False
        if self.replay:
            return False
        if session_complete(Market.HK, on, self.now):
            return False
        return self.now.date() == on and self.now.time() < HK_CLOSE

    def evaluate_book(self, book: SymbolBook, on: date, mode: EvalMode) -> SymbolEvaluation:
        hist = self._history(book, on)
        warnings: list[str] = []
        hhmm = time(14, 55) if mode is EvalMode.CHECK_1455 else (
            HK_CLOSE if book.market is Market.HK else time(15, 0)
        )
        tz = EASTERN if book.market is Market.US else SHANGHAI
        if book.market is Market.US:
            hhmm = time(16, 0)
        asof = _asof(on, hhmm, tz)

        if isinstance(hist, MissingData):
            signals = [
                Signal(
                    rule_id=rule.id,
                    symbol=book.symbol,
                    market=book.market,
                    status=SignalStatus.MISSING_DATA,
                    actual=None,
                    effective_level=rule.when.effective_level(),
                    price_source=rule.when.field,
                    asof=asof,
                    reason=hist.reason,
                    action=rule.action,
                )
                for rule in book.rules
            ]
            return SymbolEvaluation(book, signals, None, None, warnings)

        bars = _bar_map(hist)
        raw_signals: list[Signal] = []
        for rule in book.rules:
            if mode is EvalMode.CHECK_1455:
                # 14:55 只对 CN 有意义：price_1455 正式判定；CN 的 close 规则做预警。
                if rule.when.field == "price_1455":
                    pass
                elif rule.when.field == "close" and book.market is Market.CN:
                    pass
                else:
                    continue
            elif rule.when.field == "price_1455":
                continue

            if self._hk_unclosed(book, on, mode) and rule.when.field in {"close", "open", "low"}:
                raw_signals.append(
                    Signal(
                        rule_id=rule.id,
                        symbol=book.symbol,
                        market=book.market,
                        status=SignalStatus.NOT_CLOSED,
                        actual=None,
                        effective_level=rule.when.effective_level(),
                        price_source=rule.when.field,
                        asof=self.now,
                        reason="港股 16:10 尚未收盘",
                        action=rule.action,
                    )
                )
                continue

            status, actual, reason = self._eval_condition(book, rule.when, on, bars, mode)
            volume_ratio = None
            if status is SignalStatus.TRIGGERED and rule.and_conditions:
                status, actual, reason, volume_ratio = self._apply_and(
                    book, rule, on, bars, mode, actual, reason
                )

            price_source: str = rule.when.field
            if mode is EvalMode.CHECK_1455 and rule.when.field == "close":
                if status is SignalStatus.TRIGGERED:
                    status = SignalStatus.PREVIEW
                    reason = "盘中预警（以 14:55 价套用 close 规则，不算触发）"
                price_source = "price_1455 (预警)"
            if book.market is Market.US:
                price_source = f"{price_source} · 美东交易日 {on.isoformat()}"

            preview_or_hit = status in {SignalStatus.TRIGGERED, SignalStatus.PREVIEW}
            raw_signals.append(
                Signal(
                    rule_id=rule.id,
                    symbol=book.symbol,
                    market=book.market,
                    status=status,
                    actual=actual,
                    effective_level=rule.when.effective_level(),
                    price_source=price_source,
                    asof=asof,
                    reason=reason,
                    action=rule.action,
                    volume_ratio=volume_ratio,
                    sell_qty=(
                        rule.action.sell_qty(book.position.qty) if preview_or_hit else None
                    ),
                )
            )

        hits = [s for s in raw_signals if s.status is SignalStatus.TRIGGERED]
        hit_ids = {s.rule_id for s in hits}
        for sig in raw_signals:
            if sig.status is not SignalStatus.TRIGGERED:
                continue
            for other in book.rules:
                if sig.rule_id in other.supersedes and other.id in hit_ids:
                    sig.superseded_by = other.id

        evaluation = SymbolEvaluation(book, raw_signals, None, None, warnings)
        apply_cumulative(book, evaluation)
        return evaluation

    def _apply_and(
        self,
        book: SymbolBook,
        rule: Rule,
        on: date,
        bars: dict[date, Bar],
        mode: EvalMode,
        actual: float | None,
        reason: str,
    ) -> tuple[SignalStatus, float | None, str, float | None]:
        volume_ratio: float | None = None
        for extra in rule.and_conditions:
            if extra.field == "volume_ratio":
                vr = self._volume_ratio(bars, on, book.market, book.symbol)
                if isinstance(vr, MissingData):
                    return SignalStatus.MISSING_DATA, actual, vr.reason, None
                volume_ratio = vr
                if not compare(extra.op, vr, extra.effective_level()):
                    return (
                        SignalStatus.NOT_TRIGGERED,
                        actual,
                        f"附加条件未满足: volume_ratio={vr} {extra.op} {extra.effective_level()}",
                        volume_ratio,
                    )
            else:
                extra_status, extra_actual, extra_reason = self._eval_condition(
                    book, extra, on, bars, mode
                )
                if extra_status is not SignalStatus.TRIGGERED:
                    return extra_status, extra_actual, extra_reason, volume_ratio
        return SignalStatus.TRIGGERED, actual, reason, volume_ratio

    def evaluate(self, books: list[SymbolBook], on: date, mode: EvalMode) -> list[SymbolEvaluation]:
        return [self.evaluate_book(book, on, mode) for book in books]

    def events_from(self, evaluations: list[SymbolEvaluation], on: date) -> list[TriggerEvent]:
        events: list[TriggerEvent] = []
        for ev in evaluations:
            for sig in ev.visible:
                if sig.status is not SignalStatus.TRIGGERED:
                    continue
                if sig.action is None or sig.actual is None or sig.effective_level is None:
                    continue
                events.append(
                    TriggerEvent(
                        rule_id=sig.rule_id,
                        symbol=ev.book.symbol,
                        market=ev.book.market,
                        date=on,
                        trigger_price=sig.actual,
                        effective_level=sig.effective_level,
                        action=sig.action,
                        sell_qty=sig.sell_qty,
                        price_source=sig.price_source,
                        name=ev.book.name,
                    )
                )
        return events


def apply_cumulative(book: SymbolBook, evaluation: SymbolEvaluation) -> None:
    """去掉被 supersede 的规则后，动作数量取更大者；cumulative 则与被覆盖规则相加。"""
    best_qty = -1.0
    best_action: Action | None = None
    for sig in evaluation.visible:
        if sig.status is not SignalStatus.TRIGGERED or not sig.action:
            continue
        extra = 0.0
        if sig.action.cumulative:
            for other in evaluation.signals:
                if other.superseded_by == sig.rule_id and other.action:
                    extra += other.action.sell_qty(book.position.qty)
        qty = sig.action.sell_qty(book.position.qty, extra)
        sig.sell_qty = qty
        if qty > best_qty:
            best_qty = qty
            best_action = sig.action
    evaluation.merged_action = best_action
    evaluation.merged_qty = best_qty if best_qty >= 0 else None
