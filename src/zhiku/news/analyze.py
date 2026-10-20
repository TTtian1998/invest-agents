"""新闻分析层：入选重要新闻结构化输出；低价值只计数。"""

from __future__ import annotations

from datetime import date, timedelta

from zhiku.analyze import (
    ANALYSIS_FIELDS,
    DEGRADED_MARK,
    AnalysisCard,
    Claim,
    call_llm_json,
    card_from_llm,
    ensure_claim_kinds,
    llm_enabled,
)
from zhiku.news.models import ClusteredNews

CATALYST_CN = {
    "earnings_preview": "业绩预告",
    "earnings_beat": "业绩超预期/快报",
    "guidance": "指引修正",
    "buyback": "回购",
    "insider": "增减持",
    "contract": "大额合同",
    "mna": "并购重组",
    "dividend": "分红",
    "unlock": "解禁",
    "regulator": "监管函件",
    "rating_up": "评级上调",
    "rating_down": "评级下调",
    "lhb": "龙虎榜",
    "policy": "宏观/政策",
    "sentiment_hot": "情绪热度",
    "ai": "AI/算力主题",
}


def is_important(cluster: ClusteredNews) -> bool:
    groups = set(cluster.groups or [])
    tags = set(cluster.tags or [])
    x_cat = cluster.x_category or next((i.x_category for i in cluster.items if i.x_category), "")
    x_relevant = x_cat in {"institutional", "macro_fed", "newswire"} or cluster.resonance or cluster.newswire_matched
    if cluster.catalyst == "sentiment_hot" and not x_relevant:
        return False
    if (cluster.source_grade or 4) >= 4 and not x_relevant:
        return False
    if "持仓相关" in groups or "宏观" in groups or "地缘" in groups or "硬日历" in groups:
        return True
    if "事实" in tags and cluster.catalyst not in {"sentiment_hot", "other"}:
        return True
    if cluster.catalyst in {
        "earnings_beat",
        "earnings_preview",
        "guidance",
        "buyback",
        "mna",
        "regulator",
        "policy",
        "unlock",
        "rating_down",
        "rating_up",
    }:
        return True
    if cluster.chain_ids:
        return True
    return False


def _sources(cluster: ClusteredNews) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in cluster.items:
        url = item.url or ""
        key = url or item.headline
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": item.source, "headline": item.headline, "url": url})
    return out


def _priced_in_text(cluster: ClusteredNews) -> str:
    if cluster.reaction == "过度反应":
        return "【解读】价量显示过度反应，增量信息大概率已被透支，交易价值下降。"
    if cluster.reaction == "已反应":
        return "【解读】价量已有反应，继续交易需证明仍有未计价部分（例如指引/政策细则）。"
    if cluster.reaction == "未反应":
        return "【解读】价量尚未明显反应，信息可能仍未被计入；也可能是市场判定为噪音。"
    missing = cluster.reaction_missing or "缺数据：无价量确认"
    return f"【事实】{missing}。无法判断是否 priced-in，不编造。"


def _impact_text(
    cluster: ClusteredNews,
    *,
    holdings: set[str] | None,
    hist_winrate: float | None,
    chain_notes: list[str],
    script_note: str,
) -> str:
    bits: list[str] = []
    who = cluster.symbol or "相关板块（无单一代码）"
    direction = "偏空" if cluster.catalyst in {"rating_down", "regulator", "unlock"} else "偏多"
    if cluster.catalyst == "policy" and "降" not in cluster.headline:
        direction = "路径依赖（利率/风险偏好）"
    bits.append(f"【解读】首要影响对象：{who}，方向{direction}。")
    if hist_winrate is not None:
        bits.append(f"【事实】同类催化历史 T+5 方向胜率 {hist_winrate:.0%}（样本来自离线事件研究，非保证）。")
    else:
        bits.append("【事实】缺数据：无该类催化的历史胜率。")
    if chain_notes:
        bits.append("【解读】传导链：" + "；".join(chain_notes))
    if script_note:
        bits.append(script_note)
    if holdings is None:
        bits.append("【事实】未读取私有持仓，不做暴露映射。")
    elif cluster.symbol and cluster.symbol in holdings:
        bits.append(f"【事实】与持仓代码 {cluster.symbol} 相关（不展示数量/成本）。")
    elif holdings:
        bits.append("【事实】未直接命中当前持仓代码。")
    return "".join(f"{b} " for b in bits).strip()


def _counter_view(cluster: ClusteredNews) -> str:
    mapping = {
        "earnings_beat": "快报口径或一次性损益，正式财报可能下修；或已被提前抢跑。",
        "buyback": "回购额度不等于实际执行，对冲减持/解禁后净供给仍可能增加。",
        "policy": "决议符合预期但点阵图/记者会扭转交易；或数据修订改变路径。",
        "sentiment_hot": "热度是拥挤交易，反转风险高于增量信息。",
        "regulator": "函件只是流程动作，最终处罚/整改力度可能低于标题 implied。",
        "rating_down": "卖方下调常滞后价格，可能是已反应后的叙事。",
    }
    extra = mapping.get(cluster.catalyst, "标题叙事可能被证伪，或宏观窗口盖过个股逻辑。")
    return f"【推测】最强反方：{extra}"


def _next_check(cluster: ClusteredNews, on: date, upcoming: list[dict] | None) -> str:
    if cluster.catalyst in {"policy"} and upcoming:
        nxt = upcoming[0]
        when = nxt.get("scheduled_shanghai") or nxt.get("scheduled_at") or "缺数据"
        return f"【事实】下一验证节点：{nxt.get('name')}（{when}，Asia/Shanghai）。"
    if cluster.catalyst in {"earnings_beat", "earnings_preview", "guidance"}:
        return f"【事实】下一验证节点：正式财报/业绩说明会；若无日期则看 {(on + timedelta(days=1)).isoformat()} 收盘超额。"
    return f"【事实】下一验证节点：{(on + timedelta(days=1)).isoformat()} 交易日价量是否补反应或回吐。"


def _confidence(cluster: ClusteredNews) -> float:
    score = 0.35
    if "事实" in cluster.tags:
        score += 0.25
    if "未经证实" in cluster.tags:
        score -= 0.2
    if cluster.reaction in {"未反应", "已反应", "过度反应"}:
        score += 0.1
    if cluster.freshness == "NEW":
        score += 0.1
    if cluster.freshness == "旧闻翻炒":
        score -= 0.15
    if len(cluster.items) >= 2:
        score += 0.1
    return max(0.05, min(0.9, score))


def _so_what(cluster: ClusteredNews) -> str:
    cat = CATALYST_CN.get(cluster.catalyst, cluster.catalyst)
    tags = "、".join(cluster.tags) or "未标注"
    name = cluster.name or cluster.symbol or "宏观/主题"
    return (
        f"【解读】{name} 出现「{cat}」催化（标签 {tags}，新鲜度 {cluster.freshness}）。"
        "这意味着定价锚可能从旧叙事切换；是否值得交易取决于价量是否已消化、以及反方路径是否更强。"
        "不是对标题的复述，也不构成投资建议。"
    )


def rule_card(
    cluster: ClusteredNews,
    *,
    on: date,
    holdings: set[str] | None,
    hist_winrate: float | None,
    chain_notes: list[str],
    script_note: str,
    upcoming: list[dict] | None,
) -> AnalysisCard:
    card = AnalysisCard(
        subject=cluster.headline,
        important=True,
        so_what=_so_what(cluster),
        priced_in=_priced_in_text(cluster),
        impact=_impact_text(
            cluster,
            holdings=holdings,
            hist_winrate=hist_winrate,
            chain_notes=chain_notes,
            script_note=script_note,
        ),
        counter_view=_counter_view(cluster),
        next_check=_next_check(cluster, on, upcoming),
        confidence=_confidence(cluster),
        claims=[
            Claim(
                text=f"标题与来源可核对：{cluster.headline}（来源 {len(cluster.items)} 条）。",
                kind="事实",
            ),
            Claim(text=_so_what(cluster), kind="解读"),
            Claim(text=_counter_view(cluster), kind="推测"),
        ],
        deep=False,
        engine="rules",
        mark=DEGRADED_MARK,
        sources=_sources(cluster),
        symbol=cluster.symbol,
        catalyst=cluster.catalyst,
        source_grade=cluster.source_grade,
        source_grade_label=cluster.source_grade_label,
        institution_review=(
            cluster.institution_review.to_dict() if cluster.institution_review else None
        ),
    )
    if cluster.institution_review and cluster.source_grade == 3:
        card.counter_view = cluster.institution_review.counter_view or card.counter_view
        card.so_what = (
            f"{card.so_what} 【三级审查】立场：{cluster.institution_review.institution_stance}"
            f" 往绩：{cluster.institution_review.track_record}"
            f" 已知持仓：{cluster.institution_review.known_holdings}"
            f" 利益关系：{cluster.institution_review.conflicts}"
        )
    if cluster.resonance:
        card.so_what = f"{card.so_what} 【共振】多个关注账号观点同向。"
    if cluster.reverse_risk:
        card.counter_view = f"{card.counter_view} 【反向风险】情绪极度一致，拥挤交易可能反转。"
    if cluster.fomc_boost:
        card.impact = f"{card.impact} 【窗口】macro_fed 帖子处于 FOMC 事件窗口，权重上调。"
    x_cat = cluster.x_category or next((i.x_category for i in cluster.items if i.x_category), "")
    if cluster.newswire_matched:
        card.claims.append(Claim(text="快讯已匹配到一级或二级信源，可标事实。", kind="事实"))
    elif x_cat == "newswire":
        card.claims.append(Claim(text="快讯尚未匹配一级/二级信源，不能标事实。", kind="事实"))
    return ensure_claim_kinds(card)


def _llm_prompt(cluster: ClusteredNews, fallback: AnalysisCard) -> str:
    return (
        "对入选新闻做投研分析，不要复述标题。只输出 JSON，字段："
        f"{list(ANALYSIS_FIELDS)} 以及 claims（每项 text+kind，kind 必须覆盖 事实/解读/推测）。\n"
        f"headline: {cluster.headline}\n"
        f"catalyst: {cluster.catalyst}\n"
        f"tags: {cluster.tags}\n"
        f"reaction: {cluster.reaction}\n"
        f"freshness: {cluster.freshness}\n"
        f"规则草稿（可改写，不可编造数字）: {fallback.model_dump(include=set(ANALYSIS_FIELDS))}\n"
    )


def analyze_cluster(
    cluster: ClusteredNews,
    *,
    on: date,
    holdings: set[str] | None,
    hist_winrate: float | None = None,
    chain_notes: list[str] | None = None,
    script_note: str = "",
    upcoming: list[dict] | None = None,
) -> AnalysisCard:
    fallback = rule_card(
        cluster,
        on=on,
        holdings=holdings,
        hist_winrate=hist_winrate,
        chain_notes=chain_notes or [],
        script_note=script_note,
        upcoming=upcoming,
    )
    if not llm_enabled():
        return fallback
    payload = call_llm_json(_llm_prompt(cluster, fallback))
    if not payload:
        return fallback
    merged = card_from_llm(payload, fallback=fallback)
    return ensure_claim_kinds(merged or fallback)


def analyze_clusters(
    clusters: list[ClusteredNews],
    *,
    on: date,
    holdings: set[str] | None,
    hist: dict[str, float] | None = None,
    chain_notes_by_id: dict[str, str] | None = None,
    upcoming: list[dict] | None = None,
) -> tuple[list[AnalysisCard], int]:
    """返回（入选分析卡片, 低价值条数）。"""
    hist = hist or {}
    chain_notes_by_id = chain_notes_by_id or {}
    cards: list[AnalysisCard] = []
    low = 0
    for cluster in clusters:
        if not is_important(cluster):
            low += 1
            continue
        notes = [chain_notes_by_id[i] for i in cluster.chain_ids if i in chain_notes_by_id]
        cards.append(
            analyze_cluster(
                cluster,
                on=on,
                holdings=holdings,
                hist_winrate=hist.get(cluster.catalyst),
                chain_notes=notes,
                upcoming=upcoming,
            )
        )
    return cards, low
