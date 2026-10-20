"""宏观锚点流水线：日历、预期差、剧本、传导链、事件窗口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from zhiku.calendar import now_shanghai, week_bounds
from zhiku.config import Paths
from zhiku.engine.rules import RulesSchemaError, load_portfolio
from zhiku.macro.anchors import snapshots_from_raw
from zhiku.macro.calendar import decorate_event, released_on_or_before, to_shanghai, upcoming
from zhiku.macro.chains import chain_summary, load_chains, match_text
from zhiku.macro.models import GeoHeat, MacroEvent
from zhiku.macro.rates import from_fixture, live_fedwatch_approx
from zhiku.macro.scripts import build_scripts, week_script_hits
from zhiku.macro.sources import load_calendar, load_geo, load_history, load_yaml, live_calendar
from zhiku.macro.surprise import attach_surprise
from zhiku.macro.windows import active_windows, build_windows, window_banner
from zhiku.models import MissingData
from zhiku.notify.render import render, write_report
from zhiku.pipeline.settings import PipelineSettings, load_pipeline_settings
from zhiku.screens.fundamentals import Fundamentals, load_fundamentals


@dataclass
class MacroBundle:
    asof: str
    notes: list[str] = field(default_factory=list)
    calendar_7d: list[dict] = field(default_factory=list)
    released: list[dict] = field(default_factory=list)
    rate_expectation: dict = field(default_factory=dict)
    anchors: list[dict] = field(default_factory=list)
    geo: dict = field(default_factory=dict)
    scripts: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    active_windows: list[dict] = field(default_factory=list)
    window_banners: list[str] = field(default_factory=list)
    freeze_adds: bool = False
    script_hits: list[dict] = field(default_factory=list)
    chains_fired: list[dict] = field(default_factory=list)
    analyses: list[dict] = field(default_factory=list)
    analysis_engine: str = "rules"
    low_value_count: int = 0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2, default=str)


def _holdings_fundamentals(paths: Paths) -> list[Fundamentals] | None:
    if not paths.positions.is_file():
        return None
    try:
        port = load_portfolio(paths.positions)
    except RulesSchemaError:
        return None
    symbols = {p.symbol for p in port.positions}
    fund_path = paths.fixtures / "fundamentals.yaml"
    funds: list[Fundamentals] = []
    if fund_path.is_file():
        all_rows = load_fundamentals(fund_path)
        funds = [f for f in all_rows if f.symbol in symbols]
    if not funds:
        # 只有代码、没有行业时仍给出「缺行业」提示，绝不输出数量/成本
        funds = [
            Fundamentals(symbol=p.symbol, name=p.name, market=p.market) for p in port.positions
        ]
    return funds


def _event_dict(ev: MacroEvent) -> dict:
    when = to_shanghai(ev.scheduled_at)
    return {
        "id": ev.id,
        "event_type": ev.event_type,
        "name": ev.display_name,
        "region": ev.region,
        "importance": ev.importance,
        "scheduled_at": when.isoformat() if when else None,
        "scheduled_shanghai": when.strftime("%Y-%m-%d %H:%M") if when else "缺数据",
        "previous": ev.previous,
        "consensus": ev.consensus,
        "actual": ev.actual,
        "unit": ev.unit,
        "sigma": ev.sigma,
        "surprise_z": None if ev.surprise_z is None else round(ev.surprise_z, 3),
        "surprise_missing": ev.surprise_missing,
        "scenario": ev.scenario,
        "decision": ev.decision,
        "tone": ev.tone,
        "sessions": ev.sessions,
        "reactions": {k: v.model_dump() for k, v in ev.reactions.items()},
        "notes": ev.notes,
    }


def run_macro(
    paths: Paths,
    on: date,
    *,
    provider_kind: str = "fixture",
    settings: PipelineSettings | None = None,
    now: datetime | None = None,
) -> MacroBundle:
    settings = settings or load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    notes: list[str] = []
    asof_dt = now_shanghai(now) if now is not None else datetime.combine(on, datetime.min.time())
    asof_dt = now_shanghai(asof_dt)

    cal_path = paths.fixtures / "macro_calendar.yaml"
    events = load_calendar(cal_path)
    if not events:
        notes.append("缺数据：无 macro_calendar.yaml，宏观日历为空。")
    if provider_kind == "live" and settings.macro.enabled:
        live_events, live_notes = live_calendar(on)
        notes.extend(live_notes)
        events.extend(live_events)
        notes.append("live 宏观接口仅作补充；预期差仍以可解析样本为准，不编造一致预期。")
    else:
        notes.append("使用离线虚构宏观样本（不访问网络）。")

    attach_surprise(events)
    for ev in events:
        decorate_event(ev)

    horizon = settings.macro.horizon_days
    future = upcoming(events, asof_dt, days=horizon)
    released = released_on_or_before(events, asof_dt, lookback_days=1)

    history = load_history(paths.fixtures / "macro_history.yaml")
    holdings = _holdings_fundamentals(paths)
    if holdings is None:
        notes.append("未找到私有持仓文件，事件剧本不输出持仓暴露。")
    scripts = build_scripts(future, history, holdings)

    windows = build_windows(events, settings.macro)
    active = active_windows(windows, on)
    banners = window_banner(windows, on)
    freeze = any(w.freeze_adds for w in active)

    fed_raw = load_yaml(paths.fixtures / "fedwatch.yaml")
    rate = from_fixture(fed_raw)
    if provider_kind == "live" and settings.macro.fedwatch_approx:
        live_rate = live_fedwatch_approx(on)
        if isinstance(live_rate, MissingData):
            notes.append(f"利率期货：{live_rate.reason}")
        else:
            rate = live_rate

    anchors = snapshots_from_raw(load_yaml(paths.fixtures / "anchor_prices.yaml"))
    geo = load_geo(paths.fixtures / "geopolitical.yaml")
    chains = load_chains(paths.kb / "config" / "chains.yaml")
    fired: list[dict] = []
    texts: list[str] = [e.headline for e in geo.recent_events]
    for ev in released + future:
        texts.append(ev.display_name)
    for text in texts:
        for chain in match_text(text, chains):
            payload = chain_summary(chain)
            if payload not in fired:
                fired.append(payload)
    if geo.active_chains:
        by_id = {c.id: c for c in chains}
        for cid in geo.active_chains:
            if cid not in by_id:
                continue
            payload = chain_summary(by_id[cid])
            if payload not in fired:
                fired.append(payload)

    start, friday = week_bounds(on)
    hits = week_script_hits(events, history, start, friday)

    if geo.temperature == "缺数据":
        notes.append("地缘风险：缺数据")

    from zhiku.macro.analyze import analyze_event

    script_by_id = {s["event_id"]: s for s in [_script_dict(x) for x in scripts]}
    chain_note_list = [f"{c['name']}→{','.join(c.get('commodities') or [])}" for c in fired]
    analyses = []
    low_macro = 0
    for ev in released + future:
        if ev.importance < 3:
            low_macro += 1
            continue
        analyses.append(
            analyze_event(
                ev,
                on=on,
                script=script_by_id.get(ev.id),
                rate=rate.model_dump(),
                chain_notes=chain_note_list if ev.event_type in {"OPEC", "EIA"} or "地缘" in ev.display_name else [],
            ).to_dict()
        )
    if analyses and all(not a.get("deep") for a in analyses):
        notes.append("未配置 LLM key 或分析降级：未做深度分析（规则模板）。")

    return MacroBundle(
        asof=on.isoformat(),
        notes=notes,
        calendar_7d=[_event_dict(e) for e in future],
        released=[_event_dict(e) for e in released],
        rate_expectation=rate.model_dump(),
        anchors=[a.model_dump() for a in anchors],
        geo=_geo_dict(geo),
        scripts=[_script_dict(s) for s in scripts],
        windows=[w.model_dump(mode="json") for w in windows],
        active_windows=[w.model_dump(mode="json") for w in active],
        window_banners=banners,
        freeze_adds=freeze,
        script_hits=[h.model_dump() for h in hits],
        chains_fired=fired,
        analyses=analyses,
        analysis_engine="llm" if any(a.get("deep") for a in analyses) else "rules",
        low_value_count=low_macro,
    )


def _geo_dict(geo: GeoHeat) -> dict:
    return {
        "asof": geo.asof,
        "gpr": geo.gpr,
        "gpr_percentile": geo.gpr_percentile,
        "temperature": geo.temperature,
        "active_chains": geo.active_chains,
        "recent_events": [e.model_dump(mode="json") for e in geo.recent_events],
        "missing": geo.missing,
    }


def _script_dict(script) -> dict:
    return {
        "event_id": script.event_id,
        "event_type": script.event_type,
        "name": script.name,
        "scheduled_at": script.scheduled_at.isoformat() if script.scheduled_at else None,
        "notes": script.notes,
        "playbooks": [
            {
                "scenario": p.scenario,
                "n": p.n,
                "assets": [a.model_dump() for a in p.assets],
                "winners": p.winners,
                "losers": p.losers,
                "holding_impact": p.holding_impact,
            }
            for p in script.playbooks
        ],
    }


def write_macro_reports(paths: Paths, on: date, bundle: MacroBundle) -> list[Path]:
    paths.ensure_runtime_dirs()
    md = render("macro.md.j2", macro=bundle, disclaimer="仅为研究锚点，不构成投资建议")
    written = [write_report(paths.reports, on, "macro", md)]
    folder = written[0].parent
    json_path = folder / f"{on.isoformat()}-zk-macro.json"
    payload = bundle.to_json() + "\n"
    json_path.write_text(payload, encoding="utf-8")
    (paths.reports / "latest-macro.json").write_text(payload, encoding="utf-8")
    written.append(json_path)
    return written


def load_latest_macro(paths: Paths) -> MacroBundle | None:
    path = paths.reports / "latest-macro.json"
    if not path.is_file():
        files = sorted(paths.reports.glob("*/*-zk-macro.json"))
        path = files[-1] if files else path
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MacroBundle(**data)
