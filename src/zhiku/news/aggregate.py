"""去重聚合 + 与上一份报告对比新鲜度。"""

from __future__ import annotations

import json
from pathlib import Path

from zhiku.news.classify import fingerprint, groups_for, infer_catalyst, tag_cluster
from zhiku.news.grades import decorate_cluster, link_media_to_official
from zhiku.news.models import ClusteredNews, NewsItem


def cluster_items(items: list[NewsItem], holding_symbols: set[str] | None) -> list[ClusteredNews]:
    buckets: dict[str, list[NewsItem]] = {}
    for item in items:
        cat = item.catalyst if item.catalyst and item.catalyst != "other" else infer_catalyst(
            item.headline + " " + item.summary
        )
        item.catalyst = cat
        fp = fingerprint(item.symbol, cat, item.headline)
        buckets.setdefault(fp, []).append(item)
    clusters: list[ClusteredNews] = []
    for fp, group in buckets.items():
        head = group[0]
        cluster = ClusteredNews(
            fingerprint=fp,
            symbol=head.symbol,
            name=head.name,
            market=head.market,
            headline=head.headline,
            catalyst=head.catalyst,
            items=group,
            tags=tag_cluster(group),
        )
        decorate_cluster(cluster)
        cluster.groups = groups_for(cluster, holding_symbols)
        clusters.append(cluster)
    link_media_to_official(clusters)
    return clusters


def apply_freshness(clusters: list[ClusteredNews], previous: set[str]) -> None:
    for c in clusters:
        if c.fingerprint in previous:
            c.freshness = "旧闻翻炒"
        elif c.reaction in {"已反应", "过度反应"}:
            c.freshness = "已price-in"
        else:
            c.freshness = "NEW"


def load_previous_fingerprints(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    fps = data.get("fingerprints") or data.get("news_fingerprints") or []
    if isinstance(data, dict) and "clusters" in data:
        fps = [c.get("fingerprint") for c in data["clusters"] if c.get("fingerprint")]
    return {str(x) for x in fps if x}
