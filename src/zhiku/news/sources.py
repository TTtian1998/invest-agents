"""可开关的新闻源。每个源独立 try/except，失败记 SourceError，不阻塞其它源。"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import yaml

from zhiku.data.ashare import _call_first_available
from zhiku.data.cache import SQLiteCache
from zhiku.models import Market, MissingData
from zhiku.news.classify import infer_catalyst
from zhiku.news.grades import annotate_items
from zhiku.news.httputil import cached_json, http_get, throttle
from zhiku.news.models import MarketReaction, NewsItem, SourceError
from zhiku.pipeline.settings import NewsSettings


def _coerce_yaml_on(row: dict) -> dict:
    """PyYAML 1.1 会把键 on 当成布尔 True。"""
    row = dict(row)
    if True in row and "on" not in row:
        row["on"] = row.pop(True)
    return row


def load_fixture_news(path: Path) -> tuple[list[NewsItem], list[MarketReaction]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [NewsItem.model_validate(row) for row in (raw.get("items") or [])]
    reactions: list[MarketReaction] = []
    for row in raw.get("reactions") or []:
        row = _coerce_yaml_on(row)
        if "on" in row and not isinstance(row["on"], date):
            row["on"] = date.fromisoformat(str(row["on"])[:10])
        reactions.append(MarketReaction.model_validate(row))
    annotate_items(items)
    return items, reactions


def _safe(source: str, fn) -> tuple[list[NewsItem], SourceError | None]:
    try:
        items = fn()
        return items or [], None
    except Exception as exc:  # noqa: BLE001
        return [], SourceError(source=source, reason=str(exc))


def fetch_live(
    on: date,
    settings: NewsSettings,
    cache: SQLiteCache | None = None,
) -> tuple[list[NewsItem], list[SourceError]]:
    """全市场扫描入口。未开启或未配 key 的源直接跳过。"""
    items: list[NewsItem] = []
    errors: list[SourceError] = []
    flags = settings.sources

    def add(name: str, enabled: bool, fn) -> None:
        if not enabled:
            return
        got, err = _safe(name, fn)
        items.extend(got)
        if err:
            errors.append(err)

    # 一级官方优先
    add(
        "official_rss",
        flags.official_rss,
        lambda: _official_rss(settings.official_feeds, on),
    )
    add("edgar", flags.edgar, lambda: _edgar(on, cache))
    add("cninfo", flags.cninfo, lambda: _ak_news("stock_notice_report", "cninfo", on, cache, kind="official"))
    # 二级媒体（付费墙源只保留标题/摘要/链接）
    add("eastmoney_news", flags.eastmoney_news, lambda: _ak_news("stock_news_em", "eastmoney", on, cache))
    add("cls", flags.cls, lambda: _ak_news("stock_info_global_cls", "cls", on, cache))
    add("yfinance_news", flags.yfinance_news, lambda: _yf_news(["AAPL", "0700.HK"], on))
    add("finnhub", flags.finnhub, lambda: _finnhub(on, cache))
    add("rss", flags.rss, lambda: _rss(settings.rss_feeds, on, paywall_ok=True))
    # 三级机构
    add("ratings", flags.ratings, lambda: _ak_news("stock_institute_recommend", "ratings", on, cache, kind="rating"))
    # 情绪信号最后
    add("eastmoney_hot", flags.eastmoney_hot, lambda: _ak_hot(on))
    add("lhb", flags.lhb, lambda: _ak_lhb(on))
    add("apewisdom", flags.apewisdom, lambda: _apewisdom(on, cache))
    add("twitter", flags.twitter, lambda: _twitter(on, settings, cache))
    annotate_items(items)
    return items, errors


def _ak_news(fn_name: str, source: str, on: date, cache: SQLiteCache | None, kind: str = "media") -> list[NewsItem]:
    throttle("akshare")
    raw = _call_first_available([fn_name])
    if isinstance(raw, MissingData):
        raise RuntimeError(raw.reason)
    items: list[NewsItem] = []
    if raw is None or getattr(raw, "empty", True):
        return items
    for i, row in raw.head(80).iterrows():
        headline = str(row.get("标题") or row.get("title") or row.get("新闻标题") or "")
        if not headline or headline == "nan":
            continue
        symbol = row.get("代码") or row.get("symbol")
        symbol_s = None if symbol is None or str(symbol) == "nan" else str(symbol)
        url = str(row.get("链接") or row.get("url") or row.get("新闻链接") or "")
        items.append(
            NewsItem(
                id=f"{source}-{on.isoformat()}-{i}",
                source=source,
                source_kind="official" if source == "cninfo" else kind,
                symbol=symbol_s,
                headline=headline[:200],
                url=url if url != "nan" else "",
                published=datetime.combine(on, datetime.min.time()),
                catalyst=infer_catalyst(headline),
            )
        )
    return items


def _ak_hot(on: date) -> list[NewsItem]:
    throttle("akshare")
    raw = _call_first_available(["stock_hot_rank_em", "stock_hot_rank_detail_em"])
    if isinstance(raw, MissingData):
        raise RuntimeError(raw.reason)
    items: list[NewsItem] = []
    if raw is None or getattr(raw, "empty", True):
        return items
    for i, row in raw.head(30).iterrows():
        code = str(row.get("代码") or row.get("code") or "")
        name = str(row.get("股票名称") or row.get("名称") or "")
        items.append(
            NewsItem(
                id=f"hot-{on}-{i}",
                source="eastmoney_hot",
                source_kind="social",
                symbol=code or None,
                name=name,
                headline=f"东财人气榜：{name or code}",
                catalyst="sentiment_hot",
                published=datetime.combine(on, datetime.min.time()),
            )
        )
    return items


def _ak_lhb(on: date) -> list[NewsItem]:
    throttle("akshare")
    raw = _call_first_available(
        ["stock_lhb_detail_em", "stock_lhb_stock_statistic_em"],
        start_date=on.strftime("%Y%m%d"),
        end_date=on.strftime("%Y%m%d"),
    )
    if isinstance(raw, MissingData):
        raise RuntimeError(raw.reason)
    items: list[NewsItem] = []
    if raw is None or getattr(raw, "empty", True):
        return items
    for i, row in raw.head(40).iterrows():
        code = str(row.get("代码") or "")
        name = str(row.get("名称") or "")
        items.append(
            NewsItem(
                id=f"lhb-{on}-{i}",
                source="lhb",
                source_kind="lhb",
                symbol=code or None,
                name=name,
                headline=f"龙虎榜：{name or code}",
                catalyst="lhb",
                published=datetime.combine(on, datetime.min.time()),
            )
        )
    return items


def _yf_news(symbols: list[str], on: date) -> list[NewsItem]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(str(exc)) from exc
    items: list[NewsItem] = []
    for sym in symbols:
        try:
            news = yf.Ticker(sym).news or []
        except Exception:
            continue
        for i, art in enumerate(news[:10]):
            title = str(art.get("title") or "")
            if not title:
                continue
            items.append(
                NewsItem(
                    id=f"yf-{sym}-{i}",
                    source="yfinance",
                    source_kind="media",
                    symbol=sym,
                    market=Market.US if not sym.endswith(".HK") else Market.HK,
                    headline=title,
                    url=str(art.get("link") or ""),
                    catalyst=infer_catalyst(title),
                    published=datetime.combine(on, datetime.min.time()),
                )
            )
    return items


def _finnhub(on: date, cache: SQLiteCache | None) -> list[NewsItem]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    data = cached_json(cache, "finnhub", "general", on, lambda: http_get(url))
    if isinstance(data, MissingData):
        raise RuntimeError(data.reason)
    items: list[NewsItem] = []
    for i, art in enumerate((data or [])[:30]):
        title = str(art.get("headline") or "")
        if not title:
            continue
        items.append(
            NewsItem(
                id=f"fh-{i}",
                source="finnhub",
                source_kind="media",
                headline=title,
                url=str(art.get("url") or ""),
                catalyst=infer_catalyst(title),
            )
        )
    return items


def _source_from_feed(feed: str) -> str:
    low = feed.lower()
    mapping = (
        ("federalreserve", "fed"),
        ("bls.gov", "bls"),
        ("ecb.europa", "ecb"),
        ("reuters", "reuters"),
        ("bloomberg", "bloomberg"),
        ("cnbc", "cnbc"),
        ("wsj", "wsj"),
        ("ft.com", "ft"),
        ("caixin", "caixin"),
        ("yicai", "yicai"),
        ("stcn", "stcn"),
    )
    for needle, name in mapping:
        if needle in low:
            return name
    return "rss"


def _rss_items(feed: str, on: date, *, source: str, kind: str, limit: int = 15) -> list[NewsItem]:
    body = http_get(feed)
    if isinstance(body, MissingData):
        raise RuntimeError(body.reason)
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise RuntimeError(f"RSS 无法解析: {exc}") from exc
    items: list[NewsItem] = []
    for node in root.findall(".//item")[:limit]:
        title_el = node.find("title")
        link_el = node.find("link")
        desc_el = node.find("description")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue
        summary = (desc_el.text or "").strip() if desc_el is not None else ""
        items.append(
            NewsItem(
                id=f"{source}-{on}-{len(items)}",
                source=source,
                source_kind=kind,
                headline=title[:200],
                url=(link_el.text or "") if link_el is not None else feed,
                summary=summary[:400],
                catalyst=infer_catalyst(title + " " + summary),
            )
        )
    return items


def _rss(feeds: list[str], on: date, paywall_ok: bool = True) -> list[NewsItem]:
    del paywall_ok  # 付费墙源本来就只取标题/摘要/链接
    items: list[NewsItem] = []
    for feed in feeds:
        try:
            items.extend(_rss_items(feed, on, source=_source_from_feed(feed), kind="media"))
        except Exception:
            continue
    return items


def _official_rss(feeds: dict[str, str], on: date) -> list[NewsItem]:
    items: list[NewsItem] = []
    errors: list[str] = []
    for name, url in (feeds or {}).items():
        try:
            items.extend(_rss_items(url, on, source=name, kind="official"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}:{exc}")
    if not items and errors:
        raise RuntimeError("；".join(errors))
    return items


def _edgar(on: date, cache: SQLiteCache | None) -> list[NewsItem]:
    # 公开公司检索接口；失败由 _safe 记错。不抓全文。
    url = "https://efts.sec.gov/LATEST/search-index?q=%22form%20type%22%3A%228-K%22&dateRange=custom"
    data = cached_json(cache, "edgar", "8k", on, lambda: http_get(url))
    if isinstance(data, MissingData):
        raise RuntimeError(data.reason)
    hits: list = []
    if isinstance(data, dict):
        hits = ((data.get("hits") or {}).get("hits") or []) if isinstance(data.get("hits"), dict) else data.get("hits") or []
    items: list[NewsItem] = []
    for i, row in enumerate(hits[:20]):
        src = row.get("_source") if isinstance(row, dict) else {}
        src = src if isinstance(src, dict) else {}
        title = str(src.get("file_description") or src.get("display_names") or "SEC filing")
        entity = ""
        names = src.get("display_names") or []
        if isinstance(names, list) and names:
            entity = str(names[0])
        items.append(
            NewsItem(
                id=f"edgar-{on}-{i}",
                source="edgar",
                source_kind="official",
                headline=f"EDGAR：{entity or title}"[:200],
                url=str(src.get("file_url") or ""),
                catalyst=infer_catalyst(title),
            )
        )
    return items


def _apewisdom(on: date, cache: SQLiteCache | None) -> list[NewsItem]:
    url = "https://apewisdom.io/api/v1.0/filter/all-stocks"
    data = cached_json(cache, "apewisdom", "all", on, lambda: http_get(url))
    if isinstance(data, MissingData):
        raise RuntimeError(data.reason)
    results = (data or {}).get("results") if isinstance(data, dict) else data
    items: list[NewsItem] = []
    for i, row in enumerate((results or [])[:20]):
        ticker = str(row.get("ticker") or "")
        items.append(
            NewsItem(
                id=f"ape-{ticker}-{i}",
                source="apewisdom",
                source_kind="social",
                symbol=ticker or None,
                market=Market.US,
                headline=f"Reddit 热度：{ticker}",
                catalyst="sentiment_hot",
            )
        )
    return items


def _twitter(on: date, settings: NewsSettings, cache: SQLiteCache | None) -> list[NewsItem]:
    from zhiku.news.xwatch import XSkipped, fetch_live_posts

    try:
        return fetch_live_posts(on, settings, cache)
    except XSkipped as exc:
        raise RuntimeError(str(exc)) from exc
