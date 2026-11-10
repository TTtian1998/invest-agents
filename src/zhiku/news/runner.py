"""新闻流水线：扫描 → 聚合 → 价量确认 → 打分 → 机会 → 报告。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from zhiku.calendar import calendar_network, now_shanghai
from zhiku.config import Paths
from zhiku.engine.rules import RulesSchemaError, load_portfolio
from zhiku.news.aggregate import apply_freshness, cluster_items, load_previous_fingerprints
from zhiku.news.confirm import attach_reactions
from zhiku.news.models import ClusteredNews, MarketReaction, NewsItem, SourceError
from zhiku.news.opportunities import build_opportunities
from zhiku.news.sources import fetch_live, load_fixture_news
from zhiku.news.study import (
    default_weights,
    load_event_history,
    load_weights,
    save_weights,
    summarize_samples,
    update_weights,
)
from zhiku.notify.render import render, write_report
from zhiku.pipeline.settings import PipelineSettings, load_pipeline_settings
from zhiku.pipeline.sources import resolve_source_mode
from zhiku.events.pit import load_asof_universe
from zhiku.screens.fundamentals import Fundamentals
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
    kol_feed: list[dict] = field(default_factory=list)
    kol_resonance: list[dict] = field(default_factory=list)
    kol_skipped: str = ""
    x_hits: dict = field(default_factory=dict)
    emergency: dict = field(default_factory=dict)
    crowding_stats: dict = field(default_factory=dict)
    source_mode: str = "unknown"

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
    funds: list[Fundamentals] | None = None,
    prices: dict[str, float] | None = None,
    name_map: dict[str, str] | None = None,
    include_research: bool = False,
) -> NewsBundle:
    """Collect context with source-scoped calendar access, never ambient opt-in."""
    settings = settings or load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    mode = resolve_source_mode(provider_kind, enabled=settings.news.enabled)
    with calendar_network(enabled=mode == "live"):
        return _run_news(
            paths, on, kind=kind, provider_kind=mode, screen_scores=screen_scores,
            qlib_scores=qlib_scores, quality_passed=quality_passed, settings=settings,
            macro_bundle=macro_bundle, funds=funds, prices=prices, name_map=name_map,
            include_research=include_research,
        )


def _run_news(
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
    funds: list[Fundamentals] | None = None,
    prices: dict[str, float] | None = None,
    name_map: dict[str, str] | None = None,
    include_research: bool = False,
) -> NewsBundle:
    """Build deterministic news context; historical fitting is explicit opt-in.

    ``fixture`` remains a compatibility alias for ``demo``. Disabling news
    returns before source, history, event-layer, persistence, or analysis work.
    LLM analysis belongs to the review stage, not this context collector.
    """
    settings = settings or load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    source_mode = resolve_source_mode(provider_kind, enabled=settings.news.enabled)
    if source_mode == "disabled":
        return NewsBundle(
            asof=on.isoformat(),
            kind=kind,
            source_mode=source_mode,
            notes=["新闻源已关闭：未采集新闻，也未加载离线样本。"],
            analysis_engine="disabled",
            kol_skipped="新闻源已关闭",
        )
    if source_mode == "live" and on != now_shanghai().date():
        # Several adapters return current feeds and assign the requested date;
        # they cannot truthfully reconstruct historical/future news snapshots.
        return NewsBundle(
            asof=on.isoformat(),
            kind=kind,
            source_mode=source_mode,
            notes=["缺数据：live 新闻源不支持历史/未来快照，未调用当前新闻源，不回退离线样本。"],
            source_errors=[{"source": "news", "reason": "报告日不是当前上海日期，缺少 point-in-time 新闻源"}],
        )
    notes: list[str] = []
    items: list[NewsItem] = []
    errors: list[SourceError] = []
    reactions: list[MarketReaction] = []

    fixture_news = paths.fixtures / "news.yaml"
    from zhiku.news.xwatch import (
        SKIP_NO_TOKEN,
        annotate_x_clusters,
        append_x_calls,
        bearer_token,
        key_calls_from_clusters,
        kol_feed as build_kol_feed,
        load_fixture_posts,
        load_watchlist,
        load_x_calls,
        summarize_x_hits,
        update_x_weights,
    )

    watchlist = load_watchlist(paths=paths, settings=settings.news)
    notes.extend(watchlist.warnings)
    kol_skipped = ""
    if source_mode == "demo":
        if fixture_news.is_file():
            items, reactions = load_fixture_news(fixture_news)
            notes.append("使用离线虚构新闻样本（不访问网络）。")
        else:
            notes.append("缺数据：无 news.yaml，新闻雷达为空。")
        x_items = load_fixture_posts(paths.fixtures / "x_posts.yaml", watchlist)
        if x_items:
            items.extend(x_items)
            notes.append("使用离线虚构 X 帖子（不访问 X API）。")
        else:
            notes.append("缺数据：无 x_posts.yaml，大V动态为空。")
    else:
        from zhiku.data.cache import SQLiteCache

        # A broken cache must not silently switch this run to fictional data.
        try:
            cache = SQLiteCache(paths.cache)
        except Exception as exc:  # noqa: BLE001
            cache = None
            errors.append(SourceError(source="cache", reason=str(exc)))
        try:
            live_items, live_errors = fetch_live(on, settings.news, cache=cache)
            errors.extend(live_errors)
        except Exception as exc:  # noqa: BLE001
            live_items = []
            errors.append(SourceError(source="news", reason=str(exc)))
        future_count = sum(
            item.published is not None and now_shanghai(item.published).date() > on
            for item in live_items
        )
        live_items = [
            item for item in live_items
            if item.published is None or now_shanghai(item.published).date() <= on
        ]
        if future_count:
            notes.append(f"跳过报告日之后发布的 live 新闻 {future_count} 条。")
        items.extend(live_items)
        notes.append(f"live 新闻 {len(live_items)} 条（不混入离线样本）。")
        if not live_items:
            notes.append("缺数据：live 新闻为空（各源失败或未配置），不回退虚构样本。")
        for err in errors:
            if err.source == "twitter":
                kol_skipped = err.reason
                notes.append(err.reason)
            else:
                notes.append(f"源 {err.source} 失败：{err.reason}（已跳过，不阻塞）")
        if settings.news.sources.twitter and not bearer_token() and not kol_skipped:
            kol_skipped = SKIP_NO_TOKEN
            notes.append(kol_skipped)

    holdings = _holdings(paths)
    if holdings is None:
        notes.append("未找到私有持仓文件，机会中不输出与持仓的相关性。")

    clusters = cluster_items(items, holdings)
    if source_mode == "live" and name_map:
        from zhiku.news.tickers import attach_cn_tickers

        n_cn = attach_cn_tickers(clusters, name_map)
        if n_cn:
            notes.append(f"A 股新闻挂上代码 {n_cn} 条（按股票池名称/6 位代码）。")
    rx_map = {(r.symbol, r.on): r for r in reactions}
    attach_reactions(clusters, rx_map, on)
    prev = load_previous_fingerprints(paths.reports / "latest-news.json")
    apply_freshness(clusters, prev)

    from zhiku.macro.chains import load_chains, match_text

    chains = load_chains(paths.kb / "config" / "chains.yaml")
    chain_notes = {
        c.id: f"{c.name}→商品 {','.join(c.commodities)} / A股 {','.join(c.a_share)}" for c in chains
    }
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

    followups = load_followups(fixture_news) if source_mode != "live" else []
    apply_followups(clusters, followups)
    if source_mode == "live":
        track = load_institution_track(paths.journal / "institution_track.json")
    else:
        track = load_institution_track(paths.fixtures / "institution_track.yaml")
        journal_track = load_institution_track(paths.journal / "institution_track.json")
        track = {**track, **journal_track}
    for cluster in clusters:
        if cluster.source_grade == 3:
            cluster.institution_review = review_institution(cluster, track)

    if funds is None:
        if source_mode == "live":
            funds = []
            notes.append("live 新闻基本面：用当日股票池，不读 fundamentals.yaml。")
        else:
            funds = load_asof_universe(
                fundamentals_path=paths.fixtures / "fundamentals.yaml",
                pit_path=paths.fixtures / "pit_fundamentals.yaml",
                delist_path=paths.fixtures / "delisted.yaml",
                asof=on,
            )
    fund_map = {f.symbol: f for f in funds}
    if quality_passed is None and funds:
        passed, _ = filter_universe(funds, settings.quality, settings.universe)
        quality_passed = {p.symbol for p in passed}
    quality_passed = quality_passed or set()

    stats = {}
    samples = []
    if include_research:
        hist_path = paths.fixtures / "event_history.yaml"
        if source_mode == "live":
            notes.append("live 事件历史胜率：缺数据，不回落 event_history.yaml。")
        elif hist_path.is_file():
            samples = load_event_history(hist_path)
            stats = summarize_samples(samples)
    else:
        notes.append("历史事件研究与权重拟合未运行（需显式开启 research）。")
    hist_wr = {
        k: (v.winrate_t5 if v.winrate_t5 is not None else v.winrate_t1) or 0.5
        for k, v in stats.items()
    }
    # Demonstration research must never train the weights used by live runs.
    research_dir = paths.journal / "demo" if source_mode == "demo" else paths.journal
    weight_path = research_dir / "news_weights.json"
    weights = load_weights(weight_path) if include_research else default_weights()
    if include_research:
        weights = update_weights(weights, stats)

    if prices is None:
        if source_mode == "live":
            prices = {}
            notes.append("live 机会现价来自本地库，不读 monthly_prices.csv。")
        else:
            prices = _prices_from_fund(funds, paths.fixtures / "monthly_prices.csv")
    window_note = None
    freeze_adds = False
    upcoming: list[dict] = []
    if macro_bundle is None and settings.macro.enabled:
        from zhiku.macro.runner import run_macro

        macro_bundle = run_macro(paths, on, provider_kind=source_mode, settings=settings)
    if macro_bundle is not None:
        banners = list(getattr(macro_bundle, "window_banners", None) or [])
        window_note = banners[0] if banners else None
        freeze_adds = bool(getattr(macro_bundle, "freeze_adds", False))
        upcoming = list(getattr(macro_bundle, "calendar_7d", None) or [])
        notes.extend(banners)

    resonance = annotate_x_clusters(
        clusters,
        watchlist,
        settings=settings.news,
        macro_bundle=macro_bundle,
        on=on,
        track=track,
    )
    x_hits = {}
    if include_research:
        hist_calls = [] if source_mode == "live" else load_x_calls(paths.fixtures / "x_calls.yaml")
        journal_calls = load_x_calls(research_dir / "x_calls.jsonl")
        today_calls = key_calls_from_clusters(clusters, on)
        x_hits = summarize_x_hits(hist_calls + journal_calls + today_calls)
        weights = update_x_weights(weights, x_hits)
        save_weights(weight_path, weights)
        append_x_calls(research_dir / "x_calls.jsonl", today_calls)
    feed = build_kol_feed(clusters)

    from zhiku.events.layer import apply_event_layer

    store = None
    if source_mode == "live" and settings.events.enabled:
        from zhiku.data.store import MarketStore

        store = MarketStore(paths.data_dir)
    layer = apply_event_layer(
        clusters,
        paths=paths,
        on=on,
        settings=settings,
        holdings=holdings,
        samples=list(samples),
        provider_kind=source_mode,
        store=store,
    )
    notes.extend(layer.notes)
    if layer.emergency.freeze_adds:
        freeze_adds = True
        window_note = (window_note or "") + "；应急模式禁加"

    from zhiku.news.analyze import is_important, rule_card

    cards = []
    low_n = 0
    for cluster in clusters:
        if not is_important(cluster):
            low_n += 1
            continue
        cards.append(
            rule_card(
                cluster,
                on=on,
                holdings=holdings,
                hist_winrate=hist_wr.get(cluster.catalyst),
                chain_notes=[chain_notes[cid] for cid in cluster.chain_ids if cid in chain_notes],
                script_note="",
                upcoming=upcoming,
            )
        )
    if cards:
        notes.append("新闻分析使用确定性规则，未做深度分析；LLM 统一由候选审核层执行。")
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
    from zhiku.events.invalidation import attach_invalidation
    from zhiku.events.log import DecisionLog

    attach_invalidation(
        opps, clusters, on=on, event_settings=settings.events, news_settings=settings.news
    )
    log = DecisionLog(paths.journal / "decisions.jsonl")
    log.append_many("opportunity", [o.to_dict() for o in opps])
    log.append_many("analysis", [c.to_dict() for c in cards])
    _append_opportunity_log(paths.journal / "opportunities.jsonl", on, [o.to_dict() for o in opps])
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
        source_mode=source_mode,
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
        window_banners=list(getattr(macro_bundle, "window_banners", None) or [])
        if macro_bundle
        else [],
        analysis_engine=engine,
        source_hits=source_hitrate(followups),
        kol_feed=feed,
        kol_resonance=resonance,
        kol_skipped=kol_skipped,
        x_hits=x_hits,
        emergency=layer.emergency.to_dict(),
        crowding_stats=layer.crowding_stats,
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
        "x_handles": c.x_handles,
        "x_category": c.x_category,
        "tickers": c.tickers,
        "resonance": c.resonance,
        "reverse_risk": c.reverse_risk,
        "newswire_matched": c.newswire_matched,
        "fomc_boost": c.fomc_boost,
        "first_seen": c.first_seen.isoformat() if c.first_seen else None,
        "first_source": c.first_source,
        "novelty": c.novelty,
        "session": c.session,
        "session_market": c.session_market,
        "earliest_executable": c.earliest_executable,
        "decay": c.decay,
        "stale": c.stale,
        "surprise_z": c.surprise_z,
        "surprise_missing": c.surprise_missing,
        "implied_move": c.implied_move,
        "vs_implied": c.vs_implied,
        "crowding": c.crowding,
        "crowding_bucket": c.crowding_bucket,
        "crowding_note": c.crowding_note,
        "graph_impacts": [
            g.to_dict() if hasattr(g, "to_dict") else g for g in (c.graph_impacts or [])
        ],
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
                "x_handle": i.x_handle,
                "x_category": i.x_category,
                "tickers": i.tickers,
                "x_direction": i.x_direction,
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
    return written


def _append_opportunity_log(path: Path, on: date, opps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in opps:
            payload = dict(row)
            payload["date"] = on.isoformat()
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
