"""重大事件前后的交易纪律窗口。可配置禁加。"""

from __future__ import annotations

from datetime import date, timedelta

from zhiku.calendar import is_trading_day, now_shanghai
from zhiku.macro.calendar import to_shanghai
from zhiku.macro.models import EventWindow, MacroEvent
from zhiku.models import Market
from zhiku.pipeline.settings import MacroSettings


WINDOW_HINT = "事件窗口内，新开仓降仓位/止损放宽或等落地"


def _shift_trading_day(start: date, n: int, market: Market = Market.US) -> date:
    if n == 0:
        return start
    step = 1 if n > 0 else -1
    remain = abs(n)
    cur = start
    guard = 0
    while remain > 0 and guard < 40:
        guard += 1
        cur += timedelta(days=step)
        if is_trading_day(market, cur):
            remain -= 1
    return cur


def build_windows(
    events: list[MacroEvent],
    settings: MacroSettings,
) -> list[EventWindow]:
    out: list[EventWindow] = []
    default = settings.window_for("default")
    allowed = set(settings.window_event_types or [])
    for ev in events:
        if allowed and ev.event_type not in allowed:
            continue
        if ev.importance < settings.min_window_importance:
            continue
        when = to_shanghai(ev.scheduled_at)
        if when is None:
            continue
        rule = settings.window_for(ev.event_type) or default
        before = int(rule.get("before", 1))
        after = int(rule.get("after", 1))
        day = when.date()
        start = _shift_trading_day(day, -before)
        end = _shift_trading_day(day, after)
        out.append(
            EventWindow(
                event_id=ev.id,
                event_type=ev.event_type,
                name=ev.display_name,
                start=start,
                end=end,
                freeze_adds=settings.freeze_adds_in_window,
                hint=settings.window_hint or WINDOW_HINT,
                importance=ev.importance,
            )
        )
    out.sort(key=lambda w: (w.start, w.end, w.event_id))
    return out


def active_windows(windows: list[EventWindow], on: date) -> list[EventWindow]:
    return [w for w in windows if w.covers(on)]


def window_banner(windows: list[EventWindow], on: date | None = None) -> list[str]:
    day = on or now_shanghai().date()
    active = active_windows(windows, day)
    lines: list[str] = []
    for w in active:
        extra = "；窗口内禁加" if w.freeze_adds else ""
        lines.append(f"{w.name} 窗口 {w.start.isoformat()}～{w.end.isoformat()}：{w.hint}{extra}")
    return lines


def apply_to_evaluations(evaluations, windows: list[EventWindow], on: date) -> list[str]:
    """把窗口提示写进每条评估的 warnings；返回全局横幅。"""
    banners = window_banner(windows, on)
    freeze = any(w.freeze_adds for w in active_windows(windows, on))
    for ev in evaluations:
        ev.warnings.extend(banners)
        if freeze:
            ev.warnings.append("事件窗口配置为禁加（不改触发线判定）")
    return banners
