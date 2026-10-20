"""日历外重大冲击：规则识别应急模式，输出应对清单，可选禁加与推送。"""

from __future__ import annotations

from pathlib import Path

import yaml

from zhiku.events.models import Emergency
from zhiku.news.models import ClusteredNews
from zhiku.pipeline.settings import EventSettings

PACKAGE_PLAYBOOK = Path(__file__).resolve().parent / "playbook.yaml"


def load_playbook(path: Path | None = None) -> list[str]:
    target = path if path is not None and path.is_file() else PACKAGE_PLAYBOOK
    if not target.is_file():
        return []
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return [str(x) for x in (raw.get("playbook") or [])]


def load_shock_fixture(path: Path) -> dict:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def detect_emergency(
    *,
    clusters: list[ClusteredNews],
    settings: EventSettings,
    shock: dict | None = None,
    holdings: set[str] | None = None,
    playbook: list[str] | None = None,
) -> Emergency:
    reasons: list[str] = []
    shock = shock or {}
    vix = shock.get("vix_change")
    index_drop = shock.get("index_intraday")
    try:
        vix_f = float(vix) if vix is not None else None
    except (TypeError, ValueError):
        vix_f = None
    try:
        idx_f = float(index_drop) if index_drop is not None else None
    except (TypeError, ValueError):
        idx_f = None
    if vix_f is not None and vix_f >= settings.vix_spike:
        reasons.append(f"VIX 单日涨幅 {vix_f:.0%} 超过阈值 {settings.vix_spike:.0%}")
    elif vix is None:
        pass
    if idx_f is not None and idx_f <= -abs(settings.index_intraday_drop):
        reasons.append(f"指数盘中跌幅 {idx_f:.0%} 超过阈值 {settings.index_intraday_drop:.0%}")
    keys = tuple(k.lower() for k in settings.emergency_keywords)
    hits: list[str] = []
    sources: set[str] = set()
    for cluster in clusters:
        blob = (cluster.headline + " " + " ".join(i.summary for i in cluster.items)).lower()
        if any(k in blob for k in keys):
            srcs = {i.source for i in cluster.items} | {i.x_handle for i in cluster.items if i.x_handle}
            sources |= {s for s in srcs if s}
            hits.append(cluster.headline)
    if len(sources) >= settings.emergency_min_sources and hits:
        reasons.append(f"{len(sources)} 个快讯源同时出现冲突/制裁/监管关键词")
    active = bool(reasons)
    book = list(playbook or settings.playbook)
    affected = sorted(holdings) if holdings else []
    hedges = [
        "美股：认沽期权或反向 ETF（权限缺数据）",
        "A 股：反向 ETF 或降低新开仓（融券权限缺数据）",
        "不输出任何真实账户或数量",
    ]
    return Emergency(
        active=active,
        reasons=reasons,
        playbook=book if active else [],
        affected=affected,
        hedges=hedges if active else [],
        freeze_adds=bool(active and settings.freeze_adds_on_emergency),
    )
