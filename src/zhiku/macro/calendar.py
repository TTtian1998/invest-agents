"""核心宏观日历：时间统一 Asia/Shanghai，并标出影响 A/H/美哪个交易时段。"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from zhiku.calendar import SHANGHAI, now_shanghai
from zhiku.macro.models import EVENT_TYPE_CN, MacroEvent

# 上海时间下的交易时段近似（夏令时美股开盘会提前 1 小时，报告里注明）。
A_OPEN, A_CLOSE = time(9, 15), time(15, 0)
H_OPEN, H_CLOSE = time(9, 15), time(16, 10)
US_OPEN_SH = time(21, 30)  # 美东 09:30 冬令时约 22:30；用 21:30 覆盖夏令时


def to_shanghai(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SHANGHAI)
    return dt.astimezone(SHANGHAI)


def affected_sessions(when: datetime | None) -> list[str]:
    """发布时刻影响哪些现金/期货交易时段。"""
    if when is None:
        return []
    sh = to_shanghai(when)
    if sh is None:
        return []
    t = sh.time()
    sessions: list[str] = []
    if t < A_OPEN:
        sessions.append("A当日盘")
    elif t <= A_CLOSE:
        sessions.append("A当日盘")
    else:
        sessions.append("次日A")
    if t < H_OPEN:
        sessions.append("H当日盘")
    elif t <= H_CLOSE:
        sessions.append("H当日盘")
    else:
        sessions.append("次日H")
    if t >= time(20, 0) or t <= time(5, 0):
        sessions.append("美股当日盘")
    else:
        sessions.append("美股当日晚盘")
    return sessions


def decorate_event(event: MacroEvent) -> MacroEvent:
    event.scheduled_at = to_shanghai(event.scheduled_at)
    if not event.name:
        event.name = EVENT_TYPE_CN.get(event.event_type, event.event_type)
    if not event.sessions:
        event.sessions = affected_sessions(event.scheduled_at)
    return event


def upcoming(events: list[MacroEvent], asof: datetime, days: int = 7) -> list[MacroEvent]:
    start = now_shanghai(asof)
    end = start + timedelta(days=days)
    out: list[MacroEvent] = []
    for ev in events:
        when = to_shanghai(ev.scheduled_at)
        if when is None:
            continue
        if start.date() <= when.date() <= end.date():
            out.append(decorate_event(ev))
    out.sort(key=lambda e: (e.scheduled_at or start, -e.importance))
    return out


def released_on_or_before(
    events: list[MacroEvent], asof: datetime, *, lookback_days: int = 1
) -> list[MacroEvent]:
    """默认「昨日」已发布；无昨日样本时取 asof 当日已发布。"""
    clock = now_shanghai(asof)
    yesterday = clock.date() - timedelta(days=lookback_days)
    picked: list[MacroEvent] = []
    same_day: list[MacroEvent] = []
    for ev in events:
        when = to_shanghai(ev.scheduled_at)
        if when is None:
            continue
        if ev.actual is None and ev.decision is None:
            continue
        if when.date() == yesterday:
            picked.append(decorate_event(ev))
        elif when.date() == clock.date() and when <= clock:
            same_day.append(decorate_event(ev))
    if picked:
        return sorted(picked, key=lambda e: e.scheduled_at or clock)
    return sorted(same_day, key=lambda e: e.scheduled_at or clock)
