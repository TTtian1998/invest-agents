"""价量确认：超额收益 / 量比 / 主力资金流 → 未反应 / 已反应 / 过度反应。"""

from __future__ import annotations

from datetime import date

from zhiku.news.models import ClusteredNews, MarketReaction, Reaction


def label_reaction(
    excess: float | None,
    volume_ratio: float | None,
    *,
    unused_thr: float = 0.01,
    over_thr: float = 0.04,
) -> Reaction | None:
    if excess is None and volume_ratio is None:
        return None
    ex = abs(excess) if excess is not None else 0.0
    vr = volume_ratio if volume_ratio is not None else 0.0
    if ex >= over_thr:
        return "过度反应"
    if ex >= unused_thr or vr >= 1.2:
        return "已反应"
    return "未反应"


def attach_reactions(
    clusters: list[ClusteredNews],
    reactions: dict[tuple[str, date], MarketReaction],
    on: date,
) -> None:
    for c in clusters:
        if not c.symbol:
            c.reaction_missing = "无标的代码，无法做价量确认"
            continue
        rx = reactions.get((c.symbol, on))
        if rx is None:
            c.reaction_missing = "缺数据：无当日超额收益/量比/资金流"
            continue
        c.excess_return = rx.excess_return
        c.volume_ratio = rx.volume_ratio
        c.main_net = rx.main_net
        if rx.missing:
            c.reaction_missing = rx.missing
        c.reaction = label_reaction(rx.excess_return, rx.volume_ratio)
        if c.reaction is None:
            c.reaction_missing = rx.missing or "缺数据：价量字段为空，不估算"
