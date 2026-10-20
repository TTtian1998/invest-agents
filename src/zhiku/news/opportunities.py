"""从新闻簇生成做多/做空/对冲机会。没有合格标的就输出空列表（报告写「无」）。"""

from __future__ import annotations

from zhiku.events.scoring import composite_score
from zhiku.news.grades import grade_weight
from zhiku.news.xwatch import account_weight, fomc_weight
from zhiku.news.models import ClusteredNews, Direction, Opportunity, Tier
from zhiku.pipeline.settings import NewsSettings
from zhiku.screens.fundamentals import Fundamentals


def _direction(cluster: ClusteredNews) -> Direction:
    cat = cluster.catalyst
    if cat in {"rating_down", "regulator", "unlock"}:
        return "做空"
    if cluster.excess_return is not None and cluster.excess_return <= -0.03 and cluster.reaction == "过度反应":
        return "做空"
    if cat in {"policy"} and "降" not in cluster.headline:
        return "对冲"
    return "做多"


def _tier(cluster: ClusteredNews, quality_ok: bool, settings: NewsSettings) -> Tier | None:
    if "未经证实" in cluster.tags and "事实" not in cluster.tags:
        return "只观察"
    if "未经官方证实" in cluster.tags and cluster.official_followup != "confirmed":
        return "只观察"
    if cluster.source_grade == 3 and "事实" not in cluster.tags:
        # 三级观点不能单独进入质量档
        return "只观察"
    if (cluster.source_grade or 4) >= 4:
        return "只观察"
    if cluster.reaction == "过度反应":
        return "只观察"
    if quality_ok and "事实" in cluster.tags and (cluster.source_grade or 9) <= 2:
        return "质量档"
    if quality_ok and cluster.source_grade == 3:
        return "只观察"
    return "交易档"


def _stop(price: float | None, direction: Direction, pct: float) -> float | None:
    if price is None or price <= 0:
        return None
    if direction == "做空":
        return round(price * (1.0 + pct), 4)
    return round(price * (1.0 - pct), 4)


def _short_vehicle(market: str | None) -> str:
    if market == "US":
        return "认沽期权或反向 ETF（是否可交易：缺数据，视账户权限）"
    if market == "HK":
        return "反向 ETF 或融券（可否融券：缺数据）"
    return "融券或反向 ETF（可否融券：缺数据）"


def score_cluster(
    cluster: ClusteredNews,
    *,
    hist_winrate: float | None,
    quality_score: float | None,
    screen_score: float | None,
    qlib_score: float | None,
    holding_related: bool | None,
    source_weight: float,
    catalyst_weight: float,
    settings: NewsSettings,
) -> float:
    return composite_score(
        cluster,
        hist_winrate=hist_winrate,
        quality_score=quality_score,
        screen_score=screen_score,
        qlib_score=qlib_score,
        holding_related=holding_related,
        source_weight=source_weight,
        catalyst_weight=catalyst_weight,
        settings=settings,
    )


def build_opportunities(
    clusters: list[ClusteredNews],
    *,
    funds: dict[str, Fundamentals],
    quality_passed: set[str],
    screen_scores: dict[str, dict[str, float]],
    qlib_scores: dict[str, float],
    holding_symbols: set[str] | None,
    prices: dict[str, float],
    hist: dict[str, float],
    weights: dict[str, dict[str, float]],
    settings: NewsSettings,
    event_window_note: str | None = None,
    freeze_adds: bool = False,
    chain_names: dict[str, str] | None = None,
) -> list[Opportunity]:
    out: list[Opportunity] = []
    for cluster in clusters:
        if not cluster.symbol:
            if cluster.tickers:
                cluster.symbol = cluster.tickers[0]
            else:
                continue
        symbol = cluster.symbol
        fund = funds.get(symbol)
        name = cluster.name or (fund.name if fund else symbol)
        direction = _direction(cluster)
        quality_ok = symbol in quality_passed
        tier = _tier(cluster, quality_ok, settings)
        if tier is None:
            continue
        price = prices.get(symbol)
        if price is None and fund is None:
            # 没有现价也可以进观察档，交易档必须能算止损
            if tier == "交易档":
                continue
        stop = _stop(price, direction, settings.stop_pct)
        if tier == "交易档" and stop is None:
            continue
        if tier == "质量档" and not quality_ok:
            continue
        holding = None if holding_symbols is None else (symbol in holding_symbols)
        srcs = [i.source for i in cluster.items]
        learned = sum(weights.get("source", {}).get(s, 1.0) for s in srcs) / max(1, len(srcs))
        src_w = grade_weight(cluster) * learned * account_weight(cluster, weights) * fomc_weight(
            cluster, settings
        )
        cat_w = weights.get("catalyst", {}).get(cluster.catalyst, 1.0)
        screens = screen_scores.get(symbol) or {}
        best_screen = max(screens.values()) if screens else None
        sc = score_cluster(
            cluster,
            hist_winrate=hist.get(cluster.catalyst),
            quality_score=1.0 if quality_ok else 0.35,
            screen_score=best_screen,
            qlib_score=qlib_scores.get(symbol),
            holding_related=holding,
            source_weight=src_w,
            catalyst_weight=cat_w,
            settings=settings,
        )
        lot = None
        if price is not None:
            board = 100 if (cluster.market and cluster.market.value == "CN") else 1
            lot = round(price * board, 2)
        short_v = _short_vehicle(cluster.market.value if cluster.market else None) if direction != "做多" else None
        perm = None
        if direction != "做多":
            perm = "缺数据：是否有融券/期权/反向 ETF 权限（不读取任何真实账户）"
        risk = _risk(cluster, tier)
        if event_window_note:
            risk = f"{risk}；{event_window_note}" if risk else event_window_note
            if freeze_adds and tier in {"质量档", "交易档"}:
                tier = "只观察"
                risk += "；事件窗口配置为禁加，机会降为只观察"
        if cluster.resonance:
            risk = f"{risk}；多账号共振" if risk else "多账号共振"
        if cluster.reverse_risk:
            risk = f"{risk}；情绪极度一致，提示反向拥挤风险"
        if cluster.crowding_note:
            risk = f"{risk}；{cluster.crowding_note}" if risk else cluster.crowding_note
        if cluster.stale:
            risk = f"{risk}；已过时（超过半衰期）" if risk else "已过时（超过半衰期）"
            if tier in {"质量档", "交易档"}:
                tier = "只观察"
        names = chain_names or {}
        chain_labels = [names.get(cid, cid) for cid in cluster.chain_ids]
        impacts = []
        for g in cluster.graph_impacts or []:
            impacts.append(g.to_dict() if hasattr(g, "to_dict") else g)
        out.append(
            Opportunity(
                symbol=symbol,
                name=name,
                direction=direction,
                tier=tier,
                score=round(sc, 2),
                catalyst=cluster.catalyst,
                trigger_headline=cluster.headline,
                sources=list(dict.fromkeys(srcs)),
                source_urls=[i.url for i in cluster.items if i.url],
                price=price,
                lot_notional=lot,
                entry_logic=_logic(cluster, direction),
                stop=stop,
                max_hold_days=settings.max_hold_days.get(tier, 5),
                holding_related=holding,
                overlap_screens=sorted(screens),
                short_vehicle=short_v,
                special_permission=perm,
                risk=risk,
                reaction=cluster.reaction,
                hist_winrate=hist.get(cluster.catalyst),
                event_window=event_window_note,
                chains=chain_labels,
                source_grade=cluster.source_grade,
                source_grade_label=cluster.source_grade_label,
                institution_review=(
                    cluster.institution_review.to_dict() if cluster.institution_review else None
                ),
                novelty=cluster.novelty,
                crowding=cluster.crowding,
                surprise_z=cluster.surprise_z,
                vs_implied=cluster.vs_implied,
                graph_impacts=impacts,
                earliest_executable=cluster.earliest_executable,
                stale=cluster.stale,
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return _apply_caps(out, settings)


def _logic(cluster: ClusteredNews, direction: Direction) -> str:
    tags = "、".join(cluster.tags)
    react = cluster.reaction or "价量缺数据"
    return (
        f"{direction}：催化={cluster.catalyst}，标签={tags}，新鲜度={cluster.freshness}，价量={react}。"
        "仅为机会与逻辑，不构成投资建议。"
    )


def _risk(cluster: ClusteredNews, tier: Tier) -> str:
    bits = []
    if "未经证实" in cluster.tags:
        bits.append("信息未交叉验证")
    if "未经官方证实" in cluster.tags:
        bits.append("据知情人士/未具名来源，跟踪官方确认中")
    if cluster.source_grade == 3:
        bits.append("三级机构观点，不可单独当事实")
    if cluster.reaction == "过度反应":
        bits.append("价量显示或已过度反应")
    if cluster.reaction_missing:
        bits.append(cluster.reaction_missing)
    if tier == "只观察":
        bits.append("不满足质量档/交易档硬条件")
    return "；".join(bits) if bits else "常规波动与政策风险"


def _apply_caps(rows: list[Opportunity], settings: NewsSettings) -> list[Opportunity]:
    kept: list[Opportunity] = []
    counts = {"质量档": 0, "交易档": 0, "只观察": 0}
    limits = {
        "质量档": settings.max_quality,
        "交易档": settings.max_trade,
        "只观察": settings.max_watch,
    }
    for row in rows:
        if row.lot_notional is not None and row.lot_notional > settings.max_notional_cny:
            continue
        if counts[row.tier] >= limits[row.tier]:
            continue
        counts[row.tier] += 1
        kept.append(row)
    return kept
