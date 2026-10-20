"""新闻流水线：扫描 → 聚合 → 价量确认 → 打分 → 机会 → 报告。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from zhiku.config import Paths
from zhiku.engine.rules import RulesSchemaError, load_portfolio
from zhiku.news.aggregate import apply_freshness, cluster_items, load_previous_fingerprints
from zhiku.news.confirm import attach_reactions
from zhiku.news.models import ClusteredNews, MarketReaction, NewsItem, SourceError
from zhiku.news.opportunities import build_opportunities
from zhiku.news.sources import fetch_live, load_fixture_news
from zhiku.news.study import load_event_history, load_weights, save_weights, summarize_samples, update_weights
from zhiku.notify.render import render, write_report
from zhiku.pipeline.settings import PipelineSettings, load_pipeline_settings
from zhiku.screens.fundamentals import Fundamentals, load_fundamentals
from zhiku.screens.quality import filter_universe


@dataclass
class NewsBundle:
    asof: str
    kind: str
    notes: list[str] = field(default_factory=list)
    source_errors: list[dict] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    opportunities: list[dict] = field(default_factory=list)
    by_group: dict[str, list[dict]] = field(default_factory=dict)
    by_tier: dict[str, list[dict]] = field(default_factory=dict)
    fingerprints: list[str] = field(default_factory=list)
    catalyst_stats: dict[str, dict] = field(default_factory=dict)
    analyses: list[dict] = field(default_factory=list)
    low_value_count: int = 0
    links: list[dict] = field(default_factory=list)
    window_banners: list[str] = field(default_factory=list)
    analysis_engine: str = "rules"
    source_hits: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2, default=str)


def _holdings(paths: Paths) -> set[str] | None:
    if not paths.positions.is_file():
        return None
    try:
        port = load_portfolio(paths.positions)
    except RulesSchemaError:
        return None
    return {p.symbol for p in port.positions}


def _prices_from_fund(rows: list[Fundamentals], extra: Path | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if extra is not None and extra.is_file():
        import pandas as pd

        df = pd.read_csv(extra)
        if {"symbol", "close"} <= set(df.columns):
            last = df.sort_values("date" if "date" in df.columns else df.columns[0])
            for _, row in last.groupby("symbol").tail(1).iterrows():
                try:
                    out[str(row["symbol"])] = float(row["close"])
                except (TypeError, ValueError):
                    continue
    return out


def run_news(
    paths: Paths,
    on: date,
    *,
    kind: str = "all",
    provider_kind: str = "fixture",
    screen_scores: dict[str, dict[str, float]] | None = None,
    qlib_scores: dict[str, float] | None = None,
    quality_passed: set[str] | None = None,
    settings: PipelineSettings | None = None,
    macro_bundle=None,
) -> NewsBundle:
    settings = settings or load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    notes: list[str] = []
    items: list[NewsItem] = []
    errors: list[SourceError] = []
    reactions: list[MarketReaction] = []

    fixture_news = paths.fixtures / "news.yaml"
    if provider_kind == "fixture" or not settings.news.enabled:
        if fixture_news.is_file():
            items, reactions = load_fixture_news(fixture_news)
            notes.append("使用离线虚构新闻样本（不访问网络）。")
        else:
            notes.append("缺数据：无 news.yaml，新闻雷达为空。")
    else:
        from zhiku.data.cache import SQLiteCache

        live_items, errors = fetch_live(on, settings.news, cache=SQLiteCache(paths.cache))
        items.extend(live_items)
        if fixture_news.is_file():
            extra, extra_rx = load_fixture_news(fixture_news)
            items.extend(extra)
            reactions.extend(extra_rx)
            notes.append("live 源已拉取；fixture 样本仅作补充。")
        for err in errors:
            notes.append(f"源 {err.source} 失败：{err.reason}（已跳过，不阻塞）")

    holdings = _holdings(paths)
    if holdings is None:
        notes.append("未找到私有持仓文件，机会中不输出与持仓的相关性。")

    clusters = cluster_items(items, holdings)
    rx_map = {(r.symbol, r.on): r for r in reactions}
    attach_reactions(clusters, rx_map, on)
    prev = load_previous_fingerprints(paths.reports / "latest-news.json")
    apply_freshness(clusters, prev)

    from zhiku.macro.chains import load_chains, match_text

    chains = load_chains(paths.kb / "config" / "chains.yaml")
    chain_notes = {c.id: f"{c.name}→商品 {','.join(c.commodities)} / A股 {','.join(c.a_share)}" for c in chains}
    for cluster in clusters:
        blob = cluster.headline + " " + " ".join(i.summary for i in cluster.items)
        matched = match_text(blob, chains)
        cluster.chain_ids = [c.id for c in matched]
        if matched and "地缘" not in cluster.groups:
            cluster.groups = list(cluster.groups) + ["地缘"]

    from zhiku.news.grades import (
        apply_followups,
        load_followups,
        load_institution_track,
        review_institution,
        source_hitrate,
    )

    followups = load_followups(fixture_news)
    apply_followups(clusters, followups)
    track = load_institution_track(paths.fixtures / "institution_track.yaml")
    journal_track = load_institution_track(paths.journal / "institution_track.json")
    track = {**track, **journal_track}
    for cluster in clusters:
        if cluster.source_grade == 3:
            cluster.institution_review = review_institution(cluster, track)

    funds: list[Fundamentals] = []
    fund_path = paths.fixtures / "fundamentals.yaml"
    if fund_path.is_file():
        funds = load_fundamentals(fund_path)
    fund_map = {f.symbol: f for f in funds}
    if quality_passed is None and funds:
        passed, _ = filter_universe(funds, settings.quality, settings.universe)
        quality_passed = {p.symbol for p in passed}
    quality_passed = quality_passed or set()

    hist_path = paths.fixtures / "event_history.yaml"
    stats = {}
    if hist_path.is_file():
        stats = summarize_samples(load_event_history(hist_path))
    hist_wr = {
        k: (v.winrate_t5 if v.winrate_t5 is not None else v.winrate_t1) or 0.5 for k, v in stats.items()
    }
    weight_path = paths.journal / "news_weights.json"
    weights = load_weights(weight_path)
    weights = update_weights(weights, stats)
    save_weights(weight_path, weights)

    prices = _prices_from_fund(funds, paths.fixtures / "monthly_prices.csv")
    window_note = None
    freeze_adds = False
    upcoming: list[dict] = []
    if macro_bundle is None and settings.macro.enabled:
        from zhiku.macro.runner import run_macro

        macro_bundle = run_macro(paths, on, provider_kind=provider_kind, settings=settings)
    if macro_bundle is not None:
        banners = list(getattr(macro_bundle, "window_banners", None) or [])
        window_note = banners[0] if banners else None
        freeze_adds = bool(getattr(macro_bundle, "freeze_adds", False))
        upcoming = list(getattr(macro_bundle, "calendar_7d", None) or [])
        notes.extend(banners)
    from zhiku.news.analyze import analyze_clusters

    cards, low_n = analyze_clusters(
        clusters,
        on=on,
        holdings=holdings,
        hist=hist_wr,
        chain_notes_by_id=chain_notes,
        upcoming=upcoming,
    )
    if cards and all(not c.deep for c in cards):
        notes.append("未配置 LLM key 或分析降级：未做深度分析（规则模板）。")
    opps = build_opportunities(
        clusters,
        funds=fund_map,
        quality_passed=quality_passed,
        screen_scores=screen_scores or {},
        qlib_scores=qlib_scores or {},
        holding_symbols=holdings,
        prices=prices,
        hist=hist_wr,
        weights=weights,
        settings=settings.news,
        event_window_note=window_note,
        freeze_adds=freeze_adds,
        chain_names={c.id: c.name for c in chains},
    )
    card_by_subject = {c.subject: c for c in cards}
    for o in opps:
        hit = card_by_subject.get(o.trigger_headline)
        if hit is not None:
            o.analysis = hit.to_dict()

    by_group: dict[str, list[dict]] = {}
    for c in clusters:
        d = _cluster_dict(c)
        for g in c.groups:
            by_group.setdefault(g, []).append(d)
    by_tier: dict[str, list[dict]] = {"质量档": [], "交易档": [], "只观察": []}
    for o in opps:
        by_tier.setdefault(o.tier, []).append(o.to_dict())

    links: list[dict] = []
    seen_url: set[str] = set()
    for card in cards:
        for src in card.sources:
            url = src.get("url") or ""
            if url and url not in seen_url:
                seen_url.add(url)
                links.append(src)
    engine = "llm" if any(c.deep for c in cards) else "rules"
    return NewsBundle(
        asof=on.isoformat(),
        kind=kind,
        notes=notes,
        source_errors=[e.model_dump() for e in errors],
        clusters=[_cluster_dict(c) for c in clusters],
        opportunities=[o.to_dict() for o in opps],
        by_group=by_group,
        by_tier=by_tier,
        fingerprints=[c.fingerprint for c in clusters],
        catalyst_stats={k: v.model_dump() for k, v in stats.items()},
        analyses=[c.to_dict() for c in cards],
        low_value_count=low_n,
        links=links,
        window_banners=list(getattr(macro_bundle, "window_banners", None) or []) if macro_bundle else [],
        analysis_engine=engine,
        source_hits=source_hitrate(followups),
    )


def _cluster_dict(c: ClusteredNews) -> dict:
    return {
        "fingerprint": c.fingerprint,
        "symbol": c.symbol,
        "name": c.name,
        "headline": c.headline,
        "catalyst": c.catalyst,
        "tags": c.tags,
        "freshness": c.freshness,
        "groups": c.groups,
        "reaction": c.reaction,
        "excess_return": c.excess_return,
        "volume_ratio": c.volume_ratio,
        "main_net": c.main_net,
        "reaction_missing": c.reaction_missing,
        "chain_ids": c.chain_ids,
        "source_grade": c.source_grade,
        "source_grade_label": c.source_grade_label,
        "unofficial": c.unofficial,
        "exclusive": c.exclusive,
        "official_url": c.official_url,
        "official_followup": c.official_followup,
        "institution_review": c.institution_review.to_dict() if c.institution_review else None,
        "sources": [
            {
                "source": i.source,
                "kind": i.source_kind,
                "grade": i.source_grade,
                "grade_label": i.source_grade_label,
                "url": i.url,
                "official_url": i.official_url,
                "headline": i.headline,
                "attribution": i.attribution,
                "exclusive": i.exclusive,
            }
            for i in c.items
        ],
    }


def write_news_reports(paths: Paths, on: date, bundle: NewsBundle) -> list[Path]:
    paths.ensure_runtime_dirs()
    written: list[Path] = []
    kinds = ["premarket_ah", "premarket_us", "midday"] if bundle.kind == "all" else [bundle.kind]
    mapping = {
        "premarket_ah": "premarket_ah",
        "premarket-ah": "premarket_ah",
        "premarket_us": "premarket_us",
        "premarket-us": "premarket_us",
        "midday": "midday",
    }
    for kind in kinds:
        tmpl = mapping.get(kind, kind)
        md = render(f"{tmpl}.md.j2", news=bundle, disclaimer="仅为机会与逻辑，不构成投资建议")
        written.append(write_report(paths.reports, on, tmpl.replace("_", "-"), md))
    folder = paths.reports / on.strftime("%Y-%m")
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{on.isoformat()}-zk-news.json"
    json_path.write_text(bundle.to_json() + "\n", encoding="utf-8")
    (paths.reports / "latest-news.json").write_text(bundle.to_json() + "\n", encoding="utf-8")
    written.append(json_path)
    _append_opportunity_log(paths.journal / "opportunities.jsonl", on, bundle.opportunities)
    return written


def _append_opportunity_log(path: Path, on: date, opps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in opps:
            payload = dict(row)
            payload["date"] = on.isoformat()
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
