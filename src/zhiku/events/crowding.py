"""持仓拥挤度（反身性）：融资/北向/主力/龙虎榜/换手；美股 put-call、做空、零售热度。"""

from __future__ import annotations

from pathlib import Path

import yaml

from zhiku.events.models import CrowdingBucket, CrowdingCard
from zhiku.news.models import ClusteredNews
from zhiku.pipeline.settings import EventSettings


def _clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def bucket_of(score: float | None, settings: EventSettings) -> CrowdingBucket:
    if score is None:
        return "unknown"
    if score >= settings.crowding_high:
        return "high"
    if score <= settings.crowding_low:
        return "low"
    return "mid"


def compute_score(card: CrowdingCard) -> CrowdingCard:
    parts: list[float] = []
    mapping = [
        card.margin_change,
        card.northbound,
        card.main_net,
        card.lhb_concentration,
        card.turnover_pct,
        card.put_call,
        card.short_interest,
        card.retail_heat,
    ]
    for raw in mapping:
        v = _clip01(raw)
        if v is not None:
            parts.append(v)
    if not parts:
        card.score = None
        card.missing = card.missing or "缺数据：无拥挤度输入，不编造"
        return card
    card.score = round(sum(parts) / len(parts), 4)
    if card.missing and parts:
        # 部分缺数据仍给分，但保留备注
        pass
    return card


def load_crowding(path: Path) -> dict[str, CrowdingCard]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("items") if isinstance(raw, dict) else raw
    out: dict[str, CrowdingCard] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        card = compute_score(CrowdingCard.model_validate(row))
        if card.symbol:
            out[card.symbol] = card
    return out


def crowding_adjust(
    score: float | None,
    *,
    good_news: bool,
    settings: EventSettings,
) -> tuple[float, str]:
    """拥挤时好消息降权（利好出尽）；冷清加权。缺数据不调。"""
    if score is None:
        return 1.0, "缺数据：无拥挤度"
    b = bucket_of(score, settings)
    if b == "high" and good_news:
        return 0.55, "利好出尽风险"
    if b == "high" and not good_news:
        return 0.75, "拥挤且偏空，踩踏风险"
    if b == "low":
        return 1.18, "拥挤度低，加权"
    return 1.0, ""


def is_good_news(cluster: ClusteredNews) -> bool:
    if cluster.catalyst in {"rating_down", "regulator", "unlock"}:
        return False
    if cluster.excess_return is not None and cluster.excess_return < -0.02:
        return False
    return True


def attach_crowding(
    clusters: list[ClusteredNews],
    snapshots: dict[str, CrowdingCard],
    settings: EventSettings,
) -> None:
    for cluster in clusters:
        snap = snapshots.get(cluster.symbol or "")
        if snap is None:
            cluster.crowding = None
            cluster.crowding_bucket = "unknown"
            cluster.crowding_note = "缺数据：无该标的拥挤度"
            continue
        snap.bucket = bucket_of(snap.score, settings)
        factor, note = crowding_adjust(snap.score, good_news=is_good_news(cluster), settings=settings)
        snap.note = note
        cluster.crowding = snap.score
        cluster.crowding_bucket = snap.bucket
        cluster.crowding_note = note
        cluster.crowding_factor = factor
        cluster.crowding_card = snap


def summarize_by_crowding(samples: list) -> dict[str, dict]:
    """事件研究按拥挤度分组。"""
    by: dict[str, list] = {"high": [], "mid": [], "low": [], "unknown": []}
    for s in samples:
        b = getattr(s, "crowding_bucket", None) or "unknown"
        if b not in by:
            b = "unknown"
        by[b].append(s)

    def _wr(rows: list, attr: str) -> float | None:
        xs = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        if not xs:
            return None
        return sum(1 for x in xs if x > 0) / len(xs)

    out: dict[str, dict] = {}
    for key, rows in by.items():
        if not rows:
            continue
        out[key] = {
            "n": len(rows),
            "winrate_t1": _wr(rows, "t1"),
            "winrate_t5": _wr(rows, "t5"),
        }
    return out
