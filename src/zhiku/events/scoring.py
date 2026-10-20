"""综合打分：预期差、新颖度、时效衰减、历史胜率、拥挤度、信源、价量、质量、持仓相关。"""

from __future__ import annotations

from zhiku.events.surprise import surprise_component
from zhiku.news.models import ClusteredNews
from zhiku.pipeline.settings import NewsSettings


def _price_confirm(cluster: ClusteredNews) -> float:
    if cluster.reaction == "未反应":
        return 1.15
    if cluster.reaction == "已反应":
        return 0.85
    if cluster.reaction == "过度反应":
        return 0.4
    return 0.7


def composite_score(
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
    w = settings.score_weights
    surprise = surprise_component(cluster)
    novelty = cluster.novelty if cluster.novelty is not None else 0.7
    decay = cluster.decay if cluster.decay is not None else 1.0
    if cluster.stale:
        decay = min(decay, 0.25)
    hw = hist_winrate if hist_winrate is not None else 0.5
    crowd = 0.5
    if cluster.crowding is not None:
        crowd = 1.0 - cluster.crowding  # 越拥挤分数越低；冷清抬分
        factor = getattr(cluster, "crowding_factor", 1.0) or 1.0
        crowd = max(0.0, min(1.0, crowd * factor))
    q = quality_score if quality_score is not None else 0.5
    if screen_score is not None:
        q = 0.5 * q + 0.5 * (screen_score / 100.0)
    if qlib_score is not None:
        q = 0.7 * q + 0.3 * (qlib_score / 100.0)
    corr = 1.0
    if holding_related is False:
        corr = 1.0 + settings.low_corr_bonus
    px = _price_confirm(cluster)
    cat = 0.5
    if "事实" in cluster.tags:
        cat = 0.7
    if cluster.freshness == "旧闻翻炒":
        cat *= 0.6
    denom = w.total() if hasattr(w, "total") else 1.0
    denom = max(0.1, denom)
    raw = (
        w.surprise * surprise
        + w.novelty * novelty
        + w.decay * decay
        + w.hist_winrate * hw
        + w.crowding * crowd
        + w.source * source_weight
        + w.price_confirm * px
        + w.quality * q
        + w.correlation * corr
        + w.catalyst * cat * catalyst_weight
    )
    return max(0.0, 100.0 * raw / denom)
