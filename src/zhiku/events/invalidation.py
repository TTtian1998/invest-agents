"""失效条件与时间止损：机会必带 invalidation / time_stop，接入触发线与回测。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from zhiku.calendar import add_trading_days
from zhiku.events.models import ExpiryAlert
from zhiku.models import Market, MissingData
from zhiku.news.models import ClusteredNews, Direction, Opportunity
from zhiku.pipeline.settings import EventSettings, NewsSettings


def time_stop_days(catalyst: str, settings: EventSettings, tier: str | None = None) -> int:
    table = settings.time_stop_days or {}
    if catalyst in table:
        return int(table[catalyst])
    return int(table.get("default") or 5)


def invalidation_text(
    cluster: ClusteredNews,
    *,
    price: float | None,
    direction: Direction,
    stop_pct: float,
) -> str:
    bits: list[str] = []
    if price is not None and price > 0:
        if direction == "做空":
            level = round(price * (1.0 + stop_pct), 4)
            bits.append(f"价格：收盘涨破 {level} 则逻辑失效")
        else:
            level = round(price * (1.0 - stop_pct), 4)
            bits.append(f"价格：收盘跌破 {level} 则逻辑失效")
    else:
        bits.append("价格：缺数据，无法给出价格失效线")
    bits.append("事实：官方否认、预期差被证伪或正式文件大幅下修则失效")
    if cluster.unofficial:
        bits.append("事实：知情人士后续若被官方否认，立即失效")
    return "；".join(bits)


def time_stop_on(on: date, days: int, market: Market | None) -> date:
    mkt = market or Market.CN
    future = add_trading_days(mkt, on, days)
    if isinstance(future, MissingData):
        return on + timedelta(days=days)
    return future


def attach_invalidation(
    opps: list[Opportunity],
    clusters: list[ClusteredNews],
    *,
    on: date,
    event_settings: EventSettings,
    news_settings: NewsSettings,
) -> None:
    by_head = {c.headline: c for c in clusters}
    for opp in opps:
        cluster = by_head.get(opp.trigger_headline)
        days = time_stop_days(opp.catalyst, event_settings, opp.tier)
        if cluster and cluster.stale:
            days = min(days, 1)
            opp.stale = True
        opp.time_stop = days
        mkt = cluster.market if cluster is not None else None
        opp.time_stop_on = time_stop_on(on, days, mkt)
        if cluster is not None:
            opp.invalidation = invalidation_text(
                cluster, price=opp.price, direction=opp.direction, stop_pct=news_settings.stop_pct
            )
            opp.earliest_executable = cluster.earliest_executable
        else:
            opp.invalidation = "事实：逻辑被证伪则失效；价格失效线缺数据"
        if cluster is not None and cluster.stale:
            opp.risk = f"{opp.risk}；已过时（超过半衰期）" if opp.risk else "已过时（超过半衰期）"


def load_open_opportunities(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    import json

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def apply_event_discipline(evaluations, *, emergency, expiries: list[ExpiryAlert]) -> list[str]:
    """写入触发线评估：到期撤、应急禁加。卖出纪律保留，只禁新开/加仓。"""
    banners: list[str] = []
    freeze = bool(
        emergency is not None
        and getattr(emergency, "active", False)
        and getattr(emergency, "freeze_adds", False)
    )
    if emergency is not None and getattr(emergency, "active", False):
        banners.append("应急模式：" + "；".join(emergency.reasons or []))
        banners.extend(list(emergency.playbook or []))
        if freeze:
            banners.append("应急模式：禁加")
    for exp in expiries:
        banners.append(f"{exp.symbol} 事件机会到期未兑现，撤（{exp.time_stop_on}）")
    for ev in evaluations or []:
        ev.warnings.extend(banners)
        if freeze:
            ev.freeze_adds = True
    return banners


def expiry_alerts(path: Path, on: date) -> list[ExpiryAlert]:
    """到期未兑现 → 「未兑现，撤」。只读日志，不改历史行。"""
    alerts: list[ExpiryAlert] = []
    seen: set[str] = set()
    for row in load_open_opportunities(path):
        stop_on = row.get("time_stop_on")
        if not stop_on:
            continue
        try:
            day = date.fromisoformat(str(stop_on)[:10])
        except ValueError:
            continue
        symbol = str(row.get("symbol") or "")
        key = f"{symbol}:{stop_on}:{row.get('trigger_headline')}"
        if key in seen:
            continue
        if day <= on:
            seen.add(key)
            alerts.append(
                ExpiryAlert(
                    symbol=symbol,
                    headline=str(row.get("trigger_headline") or ""),
                    time_stop_on=day,
                    reason="未兑现，撤",
                    kind="time_stop",
                )
            )
    return alerts
