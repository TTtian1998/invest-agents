"""预期差：前值 / 一致预期 / 实际值 → 标准化 surprise；缺预期不编造。"""

from __future__ import annotations

from zhiku.macro.models import (
    HAWKISH_IF_HIGHER,
    HAWKISH_IF_LOWER,
    RATE_EVENTS,
    MacroEvent,
    Scenario,
)


def surprise_z(actual: float | None, consensus: float | None, sigma: float | None) -> tuple[float | None, str | None]:
    if actual is None:
        return None, "缺数据：尚无实际值"
    if consensus is None:
        return None, "缺数据：无市场一致预期，不编造"
    if sigma is None or sigma == 0:
        return None, "缺数据：无历史预期差标准差，不编造 z"
    return (actual - consensus) / sigma, None


def scenario_from_print(event: MacroEvent) -> Scenario | None:
    """利率决议走 decision+tone；其它走标准化预期差。"""
    if event.event_type in RATE_EVENTS:
        if event.decision == "hike":
            return "hawkish"
        if event.decision == "cut":
            return "dovish"
        if event.decision == "hold":
            return event.tone or "inline"
        if event.tone is not None:
            return event.tone
        return None
    z = event.surprise_z
    if z is None:
        return None
    et = event.event_type
    if et in HAWKISH_IF_LOWER:
        if z <= -0.5:
            return "hawkish"
        if z >= 0.5:
            return "dovish"
        return "inline"
    if et == "EIA":
        # 库存超预期偏空油价，不映射鹰鸽
        return "inline"
    if z >= 0.5:
        return "hawkish" if et in HAWKISH_IF_HIGHER or et not in HAWKISH_IF_LOWER else "dovish"
    if z <= -0.5:
        return "dovish" if et in HAWKISH_IF_HIGHER or et not in HAWKISH_IF_LOWER else "hawkish"
    return "inline"


def attach_surprise(events: list[MacroEvent]) -> list[MacroEvent]:
    for ev in events:
        z, missing = surprise_z(ev.actual, ev.consensus, ev.sigma)
        ev.surprise_z = z
        ev.surprise_missing = missing
        ev.scenario = scenario_from_print(ev)
        if ev.actual is None and ev.decision is None and ev.consensus is None:
            ev.notes.append("缺数据：前值/预期/实际均未提供")
        elif ev.consensus is None and ev.actual is not None:
            ev.notes.append("实际值已公布，但无一致预期，不计算预期差")
    return events
