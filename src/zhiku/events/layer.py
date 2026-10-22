"""把事件驱动层挂到新闻簇上，并写决策日志。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from zhiku.config import Paths
from zhiku.events.blackswan import detect_emergency, load_playbook, load_shock_fixture, live_shock
from zhiku.events.crowding import attach_crowding, load_crowding, summarize_by_crowding
from zhiku.events.graph import attach_graph
from zhiku.events.log import DecisionLog
from zhiku.events.models import Emergency
from zhiku.events.novelty import attach_novelty
from zhiku.events.surprise import asof_filter, attach_surprise, load_consensus, load_implied_moves
from zhiku.news.models import ClusteredNews
from zhiku.news.study import summarize_samples
from zhiku.pipeline.settings import PipelineSettings


@dataclass
class EventLayerResult:
    emergency: Emergency
    crowding_stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def apply_event_layer(
    clusters: list[ClusteredNews],
    *,
    paths: Paths,
    on: date,
    settings: PipelineSettings,
    holdings: set[str] | None,
    samples: list | None = None,
    provider_kind: str = "fixture",
    store=None,
) -> EventLayerResult:
    ev = settings.events
    notes: list[str] = []
    if not ev.enabled:
        return EventLayerResult(emergency=Emergency(), notes=["事件驱动层关闭"])

    if provider_kind == "live":
        cards = []
        implied = {}
        notes.append("live 事件层：一致预期/隐含波动/拥挤度无真实源，记缺数据，不回落 fixture。")
    else:
        cards = asof_filter(load_consensus(paths.fixtures / "earnings_consensus.yaml"), on)
        implied = load_implied_moves(paths.fixtures / "implied_moves.yaml")
    attach_surprise(clusters, cards, implied)

    stats = summarize_samples(list(samples or []))
    hist_avgs = {k: (v.avg_t1, v.avg_t5) for k, v in stats.items()}
    attach_novelty(
        clusters,
        on=on,
        settings=ev,
        journal=paths.journal / "first_seen.jsonl",
        hist_avgs=hist_avgs,
    )
    if provider_kind == "live":
        snaps = {}
    else:
        snaps = load_crowding(paths.fixtures / "crowding.yaml")
    attach_crowding(clusters, snaps, ev)
    supply_cands = [
        paths.kb / "config" / "supply_chains.yaml",
        paths.kb / "config" / "supply_chains.csv",
    ]
    if provider_kind != "live":
        supply_cands.extend(
            [paths.fixtures / "supply_chains.yaml", paths.fixtures / "supply_chains.csv"]
        )
    attach_graph(
        clusters,
        supply_path=_first_existing(*supply_cands),
        geo_path=paths.kb / "config" / "chains.yaml",
    )
    if provider_kind == "live":
        shock = live_shock(store=store)
        notes.append("应急价格确认来自 VIX/沪深300，不读 blackswan.yaml。")
    else:
        shock = load_shock_fixture(paths.fixtures / "blackswan.yaml")
    playbook = load_playbook(paths.kb / "config" / "playbook.yaml") or ev.playbook
    emergency = detect_emergency(
        clusters=clusters, settings=ev, shock=shock, holdings=holdings, playbook=playbook
    )
    if emergency.active:
        notes.append("应急模式：" + "；".join(emergency.reasons))
    elif emergency.reasons:
        notes.append("应急未开启：" + "；".join(emergency.reasons))
        notes.extend(emergency.playbook)
        if emergency.freeze_adds:
            notes.append("应急模式：触发线禁加")
        if ev.alert_on_emergency:
            try:
                from zhiku.news.alerts import dispatch

                body = "\n".join(emergency.reasons + emergency.playbook)
                notes.extend(dispatch("zhiku 应急模式", body))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"应急推送跳过：{exc}")
    crowd_stats = summarize_by_crowding(samples or [])
    log = DecisionLog(paths.journal / "decisions.jsonl")
    log.append(
        "event_layer",
        {
            "asof": on.isoformat(),
            "n_clusters": len(clusters),
            "emergency": emergency.to_dict(),
            "crowding_stats": crowd_stats,
        },
    )
    log.append(
        "judgment",
        {
            "asof": on.isoformat(),
            "clusters": [
                {
                    "fingerprint": c.fingerprint,
                    "symbol": c.symbol,
                    "headline": c.headline,
                    "catalyst": c.catalyst,
                    "surprise_z": c.surprise_z,
                    "surprise_missing": c.surprise_missing,
                    "novelty": c.novelty,
                    "session": c.session,
                    "stale": c.stale,
                    "crowding": c.crowding,
                    "crowding_note": c.crowding_note,
                }
                for c in clusters
            ],
        },
    )
    return EventLayerResult(emergency=emergency, crowding_stats=crowd_stats, notes=notes)


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None
