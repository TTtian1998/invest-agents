"""信源分级：YAML 配置等级，参与打分、标签与机构辩证审查。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from zhiku.news.models import ClusteredNews, InstitutionReview, NewsItem, NewsTag

PACKAGE_GRADES = Path(__file__).resolve().parent / "grades.yaml"
GRADE_LABELS = {
    1: "一级·官方第一手",
    2: "二级·大媒体",
    3: "三级·投资机构",
    4: "情绪·社交/自媒体",
}
REVIEW_FIELDS = (
    "institution_stance",
    "track_record",
    "key_assumptions",
    "consensus_gap",
    "counter_view",
    "known_holdings",
    "conflicts",
)


def load_grade_book(path: Path | None = None) -> dict:
    data: dict = {}
    if PACKAGE_GRADES.is_file():
        loaded = yaml.safe_load(PACKAGE_GRADES.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    if path is not None and path.is_file():
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(override, dict):
            data = {**data, **override, "grades": {**data.get("grades", {}), **(override.get("grades") or {})}}
    return data


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def unofficial_markers(book: dict | None = None) -> tuple[str, ...]:
    book = book or load_grade_book()
    return tuple(str(x) for x in (book.get("unofficial_markers") or []))


def looks_unofficial(text: str, book: dict | None = None) -> bool:
    blob = text or ""
    return any(m.lower() in blob.lower() for m in unofficial_markers(book))


def _spec(book: dict, level: int) -> dict:
    grades = book.get("grades") or {}
    return grades.get(level) or grades.get(str(level)) or {}


def grade_for(source: str, source_kind: str = "", url: str = "", book: dict | None = None) -> tuple[int, str, float]:
    """返回 (等级, 标签, 基础权重)。优先精确匹配 source，避免 eastmoney 误伤 eastmoney_hot。"""
    book = book or load_grade_book()
    src = _norm(source)
    blob = " ".join([src, _norm(source_kind), _norm(url)])
    exact: list[tuple[int, dict]] = []
    fuzzy: list[tuple[int, dict, int]] = []
    for level in (1, 2, 3, 4):
        spec = _spec(book, level)
        aliases = [_norm(a) for a in (spec.get("aliases") or []) if a]
        if src in aliases:
            exact.append((level, spec))
            continue
        hit = next((a for a in sorted(aliases, key=len, reverse=True) if a and a in blob), None)
        if hit:
            fuzzy.append((level, spec, len(hit)))
    if exact:
        level, spec = exact[0]
    elif fuzzy:
        fuzzy.sort(key=lambda x: x[2], reverse=True)
        level, spec, _ = fuzzy[0]
    else:
        if source_kind in {"official"}:
            return 1, GRADE_LABELS[1], 1.45
        if source_kind in {"rating"}:
            return 3, GRADE_LABELS[3], 0.55
        if source_kind in {"social", "lhb"}:
            return 4, GRADE_LABELS[4], 0.28
        if source_kind in {"media", "rss"}:
            return 2, GRADE_LABELS[2], 1.05
        return 4, GRADE_LABELS[4], 0.28
    label = spec.get("label") or GRADE_LABELS[level]
    weight = float(spec.get("weight") or 1.0)
    return level, str(label), weight


def is_exclusive(source: str, headline: str, book: dict | None = None) -> bool:
    book = book or load_grade_book()
    spec = (book.get("grades") or {}).get(2) or {}
    aliases = [_norm(a) for a in (spec.get("exclusive_aliases") or [])]
    src = _norm(source)
    if not any(a and a in src for a in aliases):
        return False
    text = headline.lower()
    return any(k in text for k in ("exclusive", "独家", "快讯", "breaking", "sources say", "据知情人士"))


def is_paywall(source: str, book: dict | None = None) -> bool:
    book = book or load_grade_book()
    spec = (book.get("grades") or {}).get(2) or {}
    aliases = [_norm(a) for a in (spec.get("paywall_aliases") or [])]
    src = _norm(source)
    return any(a and a in src for a in aliases)


def annotate_item(item: NewsItem, book: dict | None = None) -> NewsItem:
    book = book or load_grade_book()
    if item.source_grade is None:
        level, label, _w = grade_for(item.source, item.source_kind, item.url, book)
        item.source_grade = level
        item.source_grade_label = label
    elif not item.source_grade_label:
        item.source_grade_label = GRADE_LABELS.get(item.source_grade, "")
    blob = f"{item.headline} {item.summary}"
    if not item.attribution:
        if looks_unofficial(blob, book):
            item.attribution = "people_familiar"
        elif item.source_grade == 1:
            item.attribution = "official"
    if not item.exclusive:
        item.exclusive = is_exclusive(item.source, item.headline, book)
    if not item.paywall:
        item.paywall = is_paywall(item.source, book)
    return item


def annotate_items(items: list[NewsItem], book: dict | None = None) -> list[NewsItem]:
    book = book or load_grade_book()
    return [annotate_item(i, book) for i in items]


def _best_grade(items: list[NewsItem]) -> int:
    nums = [i.source_grade for i in items if i.source_grade]
    return min(nums) if nums else 4


def backfill_official_url(cluster: ClusteredNews) -> None:
    """媒体转述官方时，尽量挂上同簇官方原文链接。"""
    official = next((i for i in cluster.items if i.source_grade == 1 and i.url), None)
    if official is None:
        return
    cluster.official_url = official.url
    for item in cluster.items:
        if item.source_grade == 2 and not item.official_url:
            item.official_url = official.url


def decorate_cluster(cluster: ClusteredNews, book: dict | None = None) -> ClusteredNews:
    book = book or load_grade_book()
    annotate_items(cluster.items, book)
    cluster.source_grade = _best_grade(cluster.items)
    spec = (book.get("grades") or {}).get(cluster.source_grade) or {}
    cluster.source_grade_label = str(spec.get("label") or GRADE_LABELS.get(cluster.source_grade, ""))
    cluster.unofficial = any(i.attribution == "people_familiar" for i in cluster.items) or looks_unofficial(
        cluster.headline, book
    )
    cluster.exclusive = any(i.exclusive for i in cluster.items)
    backfill_official_url(cluster)
    if cluster.source_grade == 3:
        cluster.institution_review = review_institution(cluster)
    return cluster


def grade_weight(cluster: ClusteredNews, book: dict | None = None) -> float:
    book = book or load_grade_book()
    spec = (book.get("grades") or {}).get(cluster.source_grade or 4) or {}
    weight = float(spec.get("weight") or 1.0)
    if cluster.exclusive and cluster.source_grade == 2:
        weight += float(spec.get("exclusive_bonus") or 0.25)
    if cluster.unofficial and cluster.official_followup in {None, "pending"}:
        weight *= 0.7
    if cluster.official_followup == "confirmed":
        weight *= 1.1
    if cluster.official_followup == "denied":
        weight *= 0.4
    return weight


def review_institution(
    cluster: ClusteredNews,
    track: dict[str, dict] | None = None,
    account: Any = None,
) -> InstitutionReview:
    inst = next((i.source for i in cluster.items if i.source_grade == 3), cluster.items[0].source if cluster.items else "unknown")
    x_dir = next((i.x_direction for i in cluster.items if i.x_direction in {"看多", "看空"}), "")
    direction = x_dir or (
        "看多" if cluster.catalyst in {"rating_up", "buyback", "earnings_beat"} else (
            "看空" if cluster.catalyst in {"rating_down", "regulator"} else "观点不明"
        )
    )
    handle = next((i.x_handle for i in cluster.items if i.x_handle), "")
    who = f"{inst}" + (f"（@{handle}）" if handle else "")
    default_conflict = (
        "潜在利益冲突：卖方可能有承销、做市或持仓（缺数据，不读取真实账户），且有持续发片惯性。"
    )
    acct_conflicts = getattr(account, "conflicts", "") if account is not None else ""
    conflicts = acct_conflicts or default_conflict
    stance = f"{who} 立场偏{direction}。{conflicts}"
    rec = (track or {}).get(inst) or (track or {}).get(handle) or (track or {}).get("ratings") or {}
    hits = rec.get("hits")
    misses = rec.get("misses")
    if hits is None and misses is None:
        track_s = "缺数据：尚无该机构/分析师的周复盘样本，不编造命中率。"
    else:
        h, m = int(hits or 0), int(misses or 0)
        total = h + m
        rate = f"{h / total:.0%}" if total else "缺数据"
        track_s = f"历史命中 {h}、落空 {m}，命中率 {rate}（来自本地复盘，不是官方排名）。"
    assumptions = "核心假设通常是盈利预测与估值倍数；本条未提供假设原文则标缺数据，不把标题当模型。"
    if "目标价" in cluster.headline or "target" in cluster.headline.lower():
        assumptions = "核心假设含目标价/倍数；若无假设明细，目标价本身不能当事实。"
    consensus = "缺数据：无该标的一致预期样本，无法判断是否已在预期内。"
    counter = "最强反方：卖方观点常滞后于价格，可能是交易需求或关系维护，而非增量信息；三级不能单独构成事实。"
    holdings_note = "缺数据：无已核对的公开持仓快照（不编造 13F 明细，不写数量）。"
    raw_holdings = list(getattr(account, "known_holdings", None) or []) if account is not None else []
    if raw_holdings:
        holdings_note = "已知公开持仓代码：" + "、".join(str(x) for x in raw_holdings) + "（无数量，非真实账户）。"
    return InstitutionReview(
        institution=inst,
        institution_stance=stance,
        track_record=track_s,
        key_assumptions=assumptions,
        consensus_gap=consensus,
        counter_view=counter,
        known_holdings=holdings_note,
        conflicts=conflicts,
    )


def link_media_to_official(clusters: list[ClusteredNews]) -> None:
    """媒体转述官方消息时，跨簇回溯官方原文链接。"""
    officials = [c for c in clusters if c.source_grade == 1]
    keys = ("fomc", "fed", "federal reserve", "bls", "cpi", "pce", "nonfarm", "非农", "议息", "降准", "lpr")
    for cluster in clusters:
        if cluster.source_grade not in {2, 3}:
            continue
        blob = cluster.headline.lower()
        for official in officials:
            ohead = official.headline.lower()
            if any(k in blob and k in ohead for k in keys):
                url = official.official_url or next((i.url for i in official.items if i.url), "")
                if url:
                    cluster.official_url = url
                    for item in cluster.items:
                        if not item.official_url:
                            item.official_url = url
                break


def apply_followups(clusters: list[ClusteredNews], followups: list[dict]) -> None:
    for cluster in clusters:
        if not cluster.unofficial:
            cluster.official_followup = cluster.official_followup or "不适用"
            continue
        cluster.official_followup = cluster.official_followup or "pending"
        for row in followups:
            needle = str(row.get("headline_contains") or row.get("fingerprint") or "")
            if not needle:
                continue
            if needle in cluster.headline or needle == cluster.fingerprint:
                cluster.official_followup = str(row.get("status") or "pending")
                if cluster.official_followup == "confirmed" and cluster.source_grade == 1:
                    if "事实" not in cluster.tags:
                        cluster.tags = ["事实", *[t for t in cluster.tags if t != "未经官方证实"]]


def load_followups(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict):
        return list(raw.get("followups") or [])
    return []


def load_institution_track(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {str(k): dict(v) for k, v in (data.items() if isinstance(data, dict) else [])}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inst = raw.get("institutions") if isinstance(raw, dict) else raw
    return {str(k): dict(v) for k, v in (inst or {}).items()}


def source_hitrate(followups: list[dict]) -> dict[str, float | None]:
    confirmed = sum(1 for r in followups if r.get("status") == "confirmed")
    denied = sum(1 for r in followups if r.get("status") == "denied")
    pending = sum(1 for r in followups if r.get("status") == "pending")
    resolved = confirmed + denied
    return {
        "confirmed": confirmed,
        "denied": denied,
        "pending": pending,
        "hit_rate": (confirmed / resolved) if resolved else None,
    }


def tag_with_grades(items: list[NewsItem]) -> list[NewsTag]:
    """事实仅来自一级或「二级转述且已回溯官方」；三级不能单独成事实。"""
    annotate_items(items)
    has_t1 = any(i.source_grade == 1 for i in items)
    t2_sources = {i.source for i in items if i.source_grade == 2}
    has_official_link = any(i.official_url or (i.source_grade == 1 and i.url) for i in items)
    unofficial = any(i.attribution == "people_familiar" for i in items)
    only_t3 = bool(items) and all((i.source_grade or 4) == 3 for i in items)
    only_social = bool(items) and all((i.source_grade or 4) >= 4 for i in items)
    tags: list[NewsTag] = []
    if unofficial and not has_t1:
        tags.append("未经官方证实")
        tags.append("未经证实")
    elif has_t1:
        tags.append("事实")
    elif only_t3:
        tags.append("市场解读")
    elif only_social:
        tags.append("未经证实")
        tags.append("情绪")
    elif has_official_link and t2_sources:
        tags.append("事实")
    elif len(t2_sources) >= 2 and not unofficial:
        tags.append("事实")
    else:
        tags.append("未经证实")
    blob = " ".join(i.headline + i.summary for i in items)
    if any((i.source_grade or 4) >= 4 or i.source_kind in {"social", "lhb"} for i in items) or any(
        k in blob for k in ("热度", "人气", "情绪")
    ):
        if "情绪" not in tags:
            tags.append("情绪")
    if only_t3 or any(i.source_kind == "rating" or (i.catalyst or "").startswith("rating") for i in items):
        if "市场解读" not in tags:
            tags.append("市场解读")
    seen: list[NewsTag] = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen
