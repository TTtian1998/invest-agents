"""命中率：卖出类动作在 T+h 的兑现 / 误报 / 中性 / 待定。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from zhiku.calendar import add_trading_days, week_bounds
from zhiku.data.base import MarketDataProvider
from zhiku.models import (
    HIT_CN,
    SELL_ACTIONS,
    Bar,
    HitLabel,
    MissingData,
    TriggerEvent,
)


def classify_event(
    event: TriggerEvent,
    close_th: float | None,
    *,
    rebound_pct: float = 0.01,
) -> HitLabel:
    if close_th is None:
        return HitLabel.PENDING
    if event.action.type not in SELL_ACTIONS:
        return HitLabel.NEUTRAL
    if close_th <= event.trigger_price:
        return HitLabel.HIT
    if close_th >= event.effective_level * (1.0 + rebound_pct):
        return HitLabel.FALSE_ALARM
    return HitLabel.NEUTRAL


@dataclass
class HorizonOutcome:
    horizon: int
    on: date | None
    close: float | None
    label: HitLabel
    missing: str | None = None

    @property
    def label_cn(self) -> str:
        return HIT_CN[self.label]


@dataclass
class HitRecord:
    event: TriggerEvent
    outcomes: list[HorizonOutcome]


@dataclass
class GroupStat:
    market: str
    rule_type: str
    total: int
    hit: int
    false_alarm: int
    neutral: int
    pending: int

    @property
    def resolved(self) -> int:
        return self.hit + self.false_alarm + self.neutral

    @property
    def hit_rate(self) -> float | None:
        if self.resolved == 0:
            return None
        return self.hit / self.resolved


@dataclass
class WeekReview:
    start: date
    end: date
    records: list[HitRecord]
    groups: list[GroupStat]
    lessons: list[str] = field(default_factory=list)


def _close_on(
    provider: MarketDataProvider, event: TriggerEvent, on: date
) -> tuple[float | None, str | None]:
    bars = provider.get_daily_bars(event.symbol, on, on, market=event.market)
    if isinstance(bars, MissingData):
        return None, bars.reason
    match: Bar | None = next((b for b in bars if b.date == on), None)
    if match is None:
        return None, "无当日收盘"
    return match.close, None


def review_events(
    events: list[TriggerEvent],
    provider: MarketDataProvider,
    *,
    horizons: tuple[int, ...] = (1, 3),
    rebound_pct: float = 0.01,
) -> list[HitRecord]:
    records: list[HitRecord] = []
    for event in events:
        outcomes: list[HorizonOutcome] = []
        for h in horizons:
            future = add_trading_days(event.market, event.date, h)
            if isinstance(future, MissingData):
                outcomes.append(
                    HorizonOutcome(h, None, None, HitLabel.PENDING, future.reason)
                )
                continue
            close, missing = _close_on(provider, event, future)
            label = classify_event(event, close, rebound_pct=rebound_pct)
            outcomes.append(HorizonOutcome(h, future, close, label, missing))
        records.append(HitRecord(event=event, outcomes=outcomes))
    return records


def _group_key(event: TriggerEvent) -> tuple[str, str]:
    return event.market.value, event.action.type


def group_stats(records: list[HitRecord], horizon: int = 1) -> list[GroupStat]:
    buckets: dict[tuple[str, str], GroupStat] = {}
    for rec in records:
        market, rule_type = _group_key(rec.event)
        key = (market, rule_type)
        if key not in buckets:
            buckets[key] = GroupStat(market, rule_type, 0, 0, 0, 0, 0)
        stat = buckets[key]
        stat.total += 1
        outcome = next((o for o in rec.outcomes if o.horizon == horizon), None)
        label = outcome.label if outcome else HitLabel.PENDING
        if label is HitLabel.HIT:
            stat.hit += 1
        elif label is HitLabel.FALSE_ALARM:
            stat.false_alarm += 1
        elif label is HitLabel.NEUTRAL:
            stat.neutral += 1
        else:
            stat.pending += 1
    return sorted(buckets.values(), key=lambda s: (s.market, s.rule_type))


def candidate_lessons(records: list[HitRecord]) -> list[str]:
    """纯规则化文本，不自动写入任何 lessons 文件。"""
    lessons: list[str] = []
    by_label: dict[HitLabel, int] = defaultdict(int)
    reclaim = 0
    for rec in records:
        # 以 h=1 为主
        o1 = next((o for o in rec.outcomes if o.horizon == 1), None)
        if o1:
            by_label[o1.label] += 1
            if (
                rec.event.action.type in SELL_ACTIONS
                and o1.label is HitLabel.FALSE_ALARM
            ):
                reclaim += 1
        o3 = next((o for o in rec.outcomes if o.horizon == 3), None)
        if (
            o1
            and o3
            and o1.label is HitLabel.HIT
            and o3.label is HitLabel.FALSE_ALARM
        ):
            lessons.append(
                f"{rec.event.symbol} 规则 {rec.event.rule_id}：T+1 兑现但 T+3 反弹穿越有效线"
            )
    n_hit = by_label[HitLabel.HIT]
    n_fa = by_label[HitLabel.FALSE_ALARM]
    n_pending = by_label[HitLabel.PENDING]
    n_neutral = by_label[HitLabel.NEUTRAL]
    if n_hit:
        lessons.append(f"本周 {n_hit} 次卖出触发在 T+1 兑现")
    if n_fa:
        lessons.append(f"本周 {n_fa} 次盘中/收盘触发后出现反弹（误报）")
    if reclaim:
        lessons.append(f"本周 {reclaim} 次刺破后收盘附近收回并反弹穿越有效线")
    if n_neutral:
        lessons.append(f"本周 {n_neutral} 次结果落在触发价与有效线之间（中性）")
    if n_pending:
        lessons.append(f"本周 {n_pending} 条事件因缺数据待定，不要用估算补齐")
    if not lessons:
        lessons.append("本周没有足够的已兑现样本，候选教训从略")
    return lessons


def review_week(
    events: list[TriggerEvent],
    provider: MarketDataProvider,
    end: date,
    *,
    rebound_pct: float = 0.01,
) -> WeekReview:
    start, friday = week_bounds(end)
    window = [e for e in events if start <= e.date <= friday]
    records = review_events(window, provider, rebound_pct=rebound_pct)
    return WeekReview(
        start=start,
        end=friday,
        records=records,
        groups=group_stats(records, horizon=1),
        lessons=candidate_lessons(records),
    )
