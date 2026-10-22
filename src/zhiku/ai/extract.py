"""公告/新闻原文 → 可回测文本因子。缺原文不编。"""

from __future__ import annotations

from typing import Any

from zhiku.ai.backend import LLMBackend
from zhiku.ai.schema import Extraction

EXTRACT_SYS = (
    "任务=抽取。你是投研信息抽取器。只输出一个 JSON 对象，"
    "字段：event_type, direction (-1/0/1), surprise (0-1 或 null), novelty (0-1 或 null), cite。"
    "cite 必须是所给原文的标题或摘录短句。不要编造未出现的数字。"
)


def extract_text_factor(
    backend: LLMBackend,
    *,
    symbol: str | None,
    headline: str,
    body: str = "",
    published: str = "",
) -> Extraction:
    if not (headline or body).strip():
        return Extraction(symbol=symbol, event_type="缺数据", cite="缺数据", headline=headline)
    user = (
        f"symbol={symbol or ''}\npublished={published}\nheadline={headline}\n原文摘录:\n{body[:2000]}"
    )
    raw = backend.complete_json(EXTRACT_SYS, user)
    return Extraction(
        symbol=symbol,
        event_type=str(raw.get("event_type") or "其它"),
        direction=raw.get("direction") or 0,
        surprise=raw.get("surprise"),
        novelty=raw.get("novelty"),
        cite=str(raw.get("cite") or headline or "缺数据"),
        headline=headline,
    )


def extract_news_bundle(backend: LLMBackend, news: dict[str, Any], *, limit: int = 8) -> list[Extraction]:
    items = []
    for cluster in (news or {}).get("clusters") or (news or {}).get("items") or []:
        if isinstance(cluster, dict):
            items.append(cluster)
        if len(items) >= limit:
            break
    out: list[Extraction] = []
    for item in items:
        headline = str(item.get("headline") or "")
        body = str(item.get("summary") or item.get("raw_body") or "")
        symbol = item.get("symbol")
        published = str(item.get("published") or "")
        try:
            out.append(
                extract_text_factor(
                    backend, symbol=symbol, headline=headline, body=body, published=published
                )
            )
        except Exception:  # noqa: BLE001
            out.append(Extraction(symbol=symbol, headline=headline, event_type="抽取失败", cite="缺数据"))
    return out


def factor_value(ext: Extraction) -> float | None:
    """方向 × 预期差 × 新颖度，供日后做面板。缺 surprise/novelty 则只用方向。"""
    if ext.cite == "缺数据" and ext.event_type == "缺数据":
        return None
    surprise = 1.0 if ext.surprise is None else float(ext.surprise)
    novelty = 1.0 if ext.novelty is None else float(ext.novelty)
    return float(ext.direction) * surprise * novelty
