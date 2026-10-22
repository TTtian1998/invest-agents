"""历史事件研究 → 即将发生事件的三种情景剧本，并回看预判是否兑现。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from zhiku.macro.models import (
    ANCHOR_LABELS,
    SCENARIO_CN,
    EventScript,
    HistorySample,
    HorizonMove,
    MacroEvent,
    Scenario,
    ScenarioPlaybook,
    ScriptHit,
    ScriptReaction,
)
from zhiku.screens.fundamentals import Fundamentals

SCENARIOS: tuple[Scenario, ...] = ("hawkish", "inline", "dovish")


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _winrate(xs: list[float]) -> float | None:
    if not xs:
        return None
    # 方向胜率：与均值同号计胜；均值缺失则看正收益比例
    mean = _avg(xs)
    if mean is None:
        return None
    if mean == 0:
        return sum(1 for x in xs if x == 0) / len(xs)
    return sum(1 for x in xs if x * mean > 0) / len(xs)


def _collect(samples: list[HistorySample], event_type: str) -> dict[Scenario, list[HistorySample]]:
    by: dict[Scenario, list[HistorySample]] = {s: [] for s in SCENARIOS}
    for s in samples:
        if s.event_type == event_type:
            by[s.scenario].append(s)
    return by


def _playbook(
    scenario: Scenario,
    rows: list[HistorySample],
    holdings: list[Fundamentals] | None,
) -> ScenarioPlaybook:
    asset_map: dict[str, list[tuple[float | None, float | None, float | None]]] = defaultdict(list)
    winners: list[str] = []
    losers: list[str] = []
    for row in rows:
        winners.extend(row.winners)
        losers.extend(row.losers)
        for asset, mv in row.reactions.items():
            asset_map[asset].append((mv.t0, mv.t1, mv.t5))
    reactions: list[ScriptReaction] = []
    scored: list[tuple[str, float]] = []
    for asset, triples in sorted(asset_map.items()):
        t0 = [x[0] for x in triples if x[0] is not None]
        t1 = [x[1] for x in triples if x[1] is not None]
        t5 = [x[2] for x in triples if x[2] is not None]
        avg0 = _avg(t0)
        reactions.append(
            ScriptReaction(
                asset=asset,
                label=ANCHOR_LABELS.get(asset, asset),
                avg_t0=avg0,
                avg_t1=_avg(t1),
                avg_t5=_avg(t5),
                winrate_t0=_winrate(t0),
                n=len(triples),
            )
        )
        if avg0 is not None:
            scored.append((asset, avg0))
    scored.sort(key=lambda x: x[1], reverse=True)
    if not winners and scored:
        winners = [ANCHOR_LABELS.get(a, a) for a, v in scored if v > 0][:4]
        losers = [ANCHOR_LABELS.get(a, a) for a, v in scored if v < 0][-4:]
    winners = list(dict.fromkeys(winners))
    losers = list(dict.fromkeys(losers))
    impact = _holding_impact(scenario, winners, losers, holdings)
    return ScenarioPlaybook(
        scenario=scenario,
        n=len(rows),
        assets=reactions,
        winners=winners,
        losers=losers,
        holding_impact=impact,
    )


def _holding_impact(
    scenario: Scenario,
    winners: list[str],
    losers: list[str],
    holdings: list[Fundamentals] | None,
) -> list[str]:
    if holdings is None:
        return []
    if not holdings:
        return ["未找到私有持仓文件，跳过暴露提示"]
    lines: list[str] = []
    win_l = {w.lower() for w in winners}
    lose_l = {w.lower() for w in losers}
    label = SCENARIO_CN.get(scenario, scenario)
    for h in holdings:
        industry = (h.industry or "").strip()
        name = h.name or h.symbol
        if industry and industry.lower() in win_l:
            lines.append(f"{h.symbol} {name}（{industry}）在{label}情景下历史典型受益")
        elif industry and industry.lower() in lose_l:
            lines.append(f"{h.symbol} {name}（{industry}）在{label}情景下历史典型受损")
        elif industry:
            lines.append(f"{h.symbol} {name}（{industry}）行业未直接命中该情景赢家/输家名单")
        else:
            lines.append(f"{h.symbol} {name}：缺行业字段，无法映射情景暴露")
    return lines


def build_scripts(
    upcoming_events: list[MacroEvent],
    history: list[HistorySample],
    holdings: list[Fundamentals] | None,
) -> list[EventScript]:
    scripts: list[EventScript] = []
    for ev in upcoming_events:
        by = _collect(history, ev.event_type)
        playbooks = [_playbook(sc, by[sc], holdings) for sc in SCENARIOS]
        notes: list[str] = []
        if all(p.n == 0 for p in playbooks):
            notes.append(f"缺数据：无 {ev.event_type} 历史样本，剧本为空")
        scripts.append(
            EventScript(
                event_id=ev.id,
                event_type=ev.event_type,
                name=ev.display_name,
                scheduled_at=ev.scheduled_at,
                playbooks=playbooks,
                notes=notes,
            )
        )
    return scripts


def _direction(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def script_hit_for_event(
    event: MacroEvent,
    history: list[HistorySample],
) -> ScriptHit:
    if event.scenario is None:
        return ScriptHit(
            event_id=event.id,
            event_type=event.event_type,
            name=event.display_name,
            scenario=None,
            missing="缺数据：无法判定情景（无预期差或决议）",
        )
    hist = [s for s in history if s.event_type == event.event_type and s.scenario == event.scenario]
    if not hist:
        return ScriptHit(
            event_id=event.id,
            event_type=event.event_type,
            name=event.display_name,
            scenario=event.scenario,
            missing="缺数据：无该情景历史样本，无法核对剧本",
        )
    typical: dict[str, float] = {}
    for sample in hist:
        for asset, mv in sample.reactions.items():
            if mv.t0 is None:
                continue
            typical.setdefault(asset, 0.0)
            typical[asset] += mv.t0
    for asset in list(typical):
        n = sum(1 for s in hist if asset in s.reactions and s.reactions[asset].t0 is not None)
        if n:
            typical[asset] /= n
    matched = 0
    total = 0
    for asset, typical_t0 in typical.items():
        actual = event.reactions.get(asset)
        if actual is None or actual.t0 is None:
            continue
        total += 1
        if _direction(typical_t0) == 0 or _direction(actual.t0) == 0:
            continue
        if _direction(typical_t0) == _direction(actual.t0):
            matched += 1
    if total == 0:
        return ScriptHit(
            event_id=event.id,
            event_type=event.event_type,
            name=event.display_name,
            scenario=event.scenario,
            missing="缺数据：无锚定资产 T0 反应",
        )
    return ScriptHit(
        event_id=event.id,
        event_type=event.event_type,
        name=event.display_name,
        scenario=event.scenario,
        matched=matched,
        total=total,
        hit=matched / total >= 0.5,
    )


def week_script_hits(
    events: list[MacroEvent],
    history: list[HistorySample],
    start: date,
    end: date,
) -> list[ScriptHit]:
    hits: list[ScriptHit] = []
    for ev in events:
        when = ev.scheduled_at
        if when is None:
            continue
        if not (start <= when.date() <= end):
            continue
        if ev.actual is None and ev.decision is None:
            continue
        hits.append(script_hit_for_event(ev, history))
    return hits


def empty_move() -> HorizonMove:
    return HorizonMove(missing="缺数据")
