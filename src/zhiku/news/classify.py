"""规则+关键词标签。LLM 只做可选摘要/分类，可关。"""

from __future__ import annotations

import hashlib
import os
import re

from zhiku.news.grades import tag_with_grades
from zhiku.news.models import ClusteredNews, GroupName, NewsItem, NewsTag

CATALYST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings_preview": ("业绩预告", "预增", "预减", "预盈", "预亏"),
    "earnings_beat": ("超预期", "好于预期", "beat", "业绩快报"),
    "guidance": ("指引", "guidance", "下调预期", "上调预期"),
    "buyback": ("回购", "buyback"),
    "insider": ("增持", "减持", "增减持"),
    "contract": ("大合同", "中标", "重大合同"),
    "mna": ("并购", "收购", "重组"),
    "dividend": ("分红", "派息", "股息"),
    "unlock": ("解禁",),
    "regulator": ("监管函", "问询函", "关注函", "立案"),
    "rating_up": ("上调评级", "买入评级", "目标价上调", "upgrade"),
    "rating_down": ("下调评级", "降级", "downgrade"),
    "lhb": ("龙虎榜", "机构席位"),
    "policy": ("央行", "降准", "议息", "FOMC", "CPI", "政策"),
    "sentiment_hot": ("人气榜", "热搜", "Reddit", "热度"),
    "ai": ("人工智能", "大模型", "算力", "GPU", "AI芯片", "英伟达"),
}

AI_KW = CATALYST_KEYWORDS["ai"]
MACRO_KW = ("央行", "降准", "加息", "议息", "FOMC", "CPI", "PPI", "PCE", "财政部", "宏观", "LPR")
CALENDAR_KW = ("财报", "业绩预告", "业绩快报", "解禁", "议息", "FOMC")
GEO_KW = ("中东", "霍尔木兹", "俄乌", "关税", "出口管制", "稀土", "避险", "地缘")

OFFICIAL = frozenset({"cninfo", "exchange", "company", "sse", "szse"})
SOCIAL = frozenset({"apewisdom", "twitter", "x", "eastmoney_hot", "reddit"})


def fingerprint(symbol: str | None, catalyst: str, headline: str) -> str:
    norm = re.sub(r"\s+", "", headline.lower())[:80]
    raw = f"{symbol or '-'}|{catalyst}|{norm}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def infer_catalyst(text: str) -> str:
    blob = text or ""
    for name, keys in CATALYST_KEYWORDS.items():
        if any(k.lower() in blob.lower() for k in keys):
            return name
    return "other"


def _has_any(text: str, keys: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keys)


def tag_cluster(items: list[NewsItem]) -> list[NewsTag]:
    """标签必须尊重信源等级：三级/社交不能单独构成「事实」。"""
    tags = tag_with_grades(items)
    blob = " ".join(i.headline + i.summary for i in items)
    if _has_any(blob, ("解读", "认为", "或将", "分析师")) and "市场解读" not in tags:
        tags.append("市场解读")
    if not tags:
        tags.append("未经证实")
    seen: list[NewsTag] = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen


def groups_for(cluster: ClusteredNews, holding_symbols: set[str] | None) -> list[GroupName]:
    blob = cluster.headline + " " + " ".join(i.summary for i in cluster.items)
    out: list[GroupName] = []
    if holding_symbols is not None and cluster.symbol and cluster.symbol in holding_symbols:
        out.append("持仓相关")
    if _has_any(blob, AI_KW) or cluster.catalyst == "ai":
        out.append("AI科技")
    if _has_any(blob, MACRO_KW) or cluster.catalyst == "policy":
        out.append("宏观")
    if _has_any(blob, GEO_KW):
        out.append("地缘")
    if _has_any(blob, CALENDAR_KW) or cluster.catalyst in {
        "earnings_preview",
        "earnings_beat",
        "unlock",
        "policy",
    }:
        out.append("硬日历")
    if not out:
        out.append("其它")
    return out


def llm_enabled() -> bool:
    flag = os.environ.get("ZHIKU_NEWS_LLM", "").lower() in {"1", "true", "yes"}
    key = os.environ.get("ZHIKU_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
    return bool(flag and key)


def maybe_llm_summary(text: str) -> str | None:
    if not llm_enabled():
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    key = os.environ.get("ZHIKU_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    model = os.environ.get("ZHIKU_LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "用不超过 40 字中文摘要下面新闻，不要编造数字：\n" + text[:2000],
            }
        ],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()
