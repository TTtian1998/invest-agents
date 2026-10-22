"""宏观事件分析层：落地与即将发生的事件都给出结构化情景分析。"""

from __future__ import annotations

from datetime import date, timedelta

from zhiku.analyze import (
    ANALYSIS_FIELDS,
    DEGRADED_MARK,
    AnalysisCard,
    Claim,
    call_llm_json,
    card_from_llm,
    ensure_claim_kinds,
    llm_enabled,
)
from zhiku.macro.models import ANCHOR_LABELS, SCENARIO_CN, MacroEvent


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "缺数据"
    return f"{value:.{digits}g}"


def _script_impact(script: dict | None) -> str:
    if not script:
        return "【事实】缺数据：无对应事件剧本历史样本。"
    bits: list[str] = []
    for pb in script.get("playbooks") or []:
        sc = pb.get("scenario")
        label = SCENARIO_CN.get(sc, sc)
        n = pb.get("n") or 0
        if n == 0:
            bits.append(f"{label}情景缺历史样本")
            continue
        winners = "、".join(pb.get("winners") or []) or "未列"
        losers = "、".join(pb.get("losers") or []) or "未列"
        bits.append(f"{label}（n={n}）：受益 {winners}；受损 {losers}")
        holds = pb.get("holding_impact") or []
        if holds:
            bits.append("持仓暴露：" + "；".join(holds[:6]))
    return "【解读】历史剧本：" + "。".join(bits)


def _priced_in(event: MacroEvent, rate: dict | None) -> str:
    if event.actual is None and event.decision is None:
        implied = None
        if rate:
            implied = rate.get("hold_prob")
        extra = ""
        if implied is not None:
            extra = f" 当前利率预期 hold≈{implied}（来源 {(rate or {}).get('source')}）。"
        return "【解读】事件尚未落地，市场按隐含路径交易，不能视为已完全 priced-in。" + extra
    if event.surprise_missing:
        return f"【事实】{event.surprise_missing}。无法判断预期差是否被消化。"
    z = event.surprise_z
    if z is None:
        return "【事实】缺数据：无标准化预期差。"
    rx = event.reactions.get("US10Y") or event.reactions.get("CSI300")
    if rx is None or rx.t0 is None:
        return f"【解读】surprise_z={z:.2f}，但缺锚定资产 T0 反应，无法确认消化程度。"
    return (
        f"【解读】surprise_z={z:.2f}，锚定资产已出现 T0 反应"
        f"（示例 {ANCHOR_LABELS.get('US10Y', 'US10Y')} t0={_fmt(event.reactions['US10Y'].t0 if 'US10Y' in event.reactions else None)}）。"
        "方向若与剧本一致，视为部分消化；幅度是否过度需对照历史均值。"
    )


def _so_what(event: MacroEvent) -> str:
    when = event.scheduled_at.strftime("%Y-%m-%d %H:%M") if event.scheduled_at else "时间缺数据"
    sessions = "、".join(event.sessions) or "时段缺数据"
    if event.actual is None and event.decision is None:
        return (
            f"【解读】{event.display_name} 将于 {when}（Asia/Shanghai）发布，影响 {sessions}。"
            "这是风险窗口而非交易信号：三种情景（偏鹰/符合/偏鸽）会改写利率、美元、权益久期与商品路径。"
        )
    sc = SCENARIO_CN.get(event.scenario or "", "情景未定")
    return (
        f"【解读】{event.display_name} 已公布：前值 {_fmt(event.previous)} / 预期 {_fmt(event.consensus)}"
        f" / 实际 {_fmt(event.actual)}，判定 {sc}。"
        "对定价的含义是利率路径与风险偏好可能被改写，而不是标题本身。"
    )


def _counter(event: MacroEvent) -> str:
    if event.event_type == "FOMC":
        return "【推测】最强反方：利率不变但点阵图/记者会更鹰或更鸽，盘中路径与决议方向相反。"
    if event.actual is None:
        return "【推测】最强反方：数据符合预期，真正的交易发生在分项/修订/官员讲话而非头条。"
    return "【推测】最强反方：预期差被归因于一次性因素，或被其它同日事件对冲，锚定资产回归。"


def _next_check(event: MacroEvent, on: date) -> str:
    if event.scheduled_at and event.scheduled_at.date() > on:
        nxt = event.scheduled_at.date()
        return f"【事实】验证节点：{nxt.isoformat()} 发布时刻及次日 A/H 开盘（Asia/Shanghai）。"
    nxt = on + timedelta(days=1)
    return f"【事实】验证节点：T+1（约 {nxt.isoformat()}）锚定资产是否延续 T0 方向。"


def _confidence(event: MacroEvent) -> float:
    score = 0.4
    if event.importance >= 5:
        score += 0.15
    if event.surprise_z is not None:
        score += 0.15
    if event.reactions:
        score += 0.1
    if event.consensus is None:
        score -= 0.2
    return max(0.05, min(0.9, score))


def _claims(event: MacroEvent, so_what: str, counter: str) -> list[Claim]:
    facts = (
        f"{event.display_name}：前值 {_fmt(event.previous)}，预期 {_fmt(event.consensus)}，"
        f"实际 {_fmt(event.actual)}，单位 {event.unit or '缺数据'}。"
    )
    return [
        Claim(text=facts, kind="事实"),
        Claim(text=so_what, kind="解读"),
        Claim(text=counter, kind="推测"),
    ]


def rule_card(
    event: MacroEvent,
    *,
    on: date,
    script: dict | None,
    rate: dict | None,
    chain_notes: list[str],
) -> AnalysisCard:
    so_what = _so_what(event)
    counter = _counter(event)
    impact = _script_impact(script)
    if chain_notes:
        impact += " 传导链：" + "；".join(chain_notes)
    card = AnalysisCard(
        subject=event.display_name,
        important=True,
        so_what=so_what,
        priced_in=_priced_in(event, rate),
        impact=impact,
        counter_view=counter,
        next_check=_next_check(event, on),
        confidence=_confidence(event),
        claims=_claims(event, so_what, counter),
        deep=False,
        engine="rules",
        mark=DEGRADED_MARK,
        sources=[],
        catalyst=event.event_type,
    )
    return ensure_claim_kinds(card)


def analyze_event(
    event: MacroEvent,
    *,
    on: date,
    script: dict | None = None,
    rate: dict | None = None,
    chain_notes: list[str] | None = None,
) -> AnalysisCard:
    fallback = rule_card(event, on=on, script=script, rate=rate, chain_notes=chain_notes or [])
    if not llm_enabled():
        return fallback
    prompt = (
        "对宏观事件做投研分析，不要复述日历标题。只输出 JSON，字段："
        f"{list(ANALYSIS_FIELDS)} 以及 claims（覆盖 事实/解读/推测）。不要编造未提供的预期或实际值。\n"
        f"event: {event.display_name} type={event.event_type} importance={event.importance}\n"
        f"previous={event.previous} consensus={event.consensus} actual={event.actual} z={event.surprise_z}\n"
        f"规则草稿: {fallback.model_dump(include=set(ANALYSIS_FIELDS))}\n"
    )
    payload = call_llm_json(prompt)
    if not payload:
        return fallback
    merged = card_from_llm(payload, fallback=fallback)
    return ensure_claim_kinds(merged or fallback)
