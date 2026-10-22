"""X 财经大V：名单、采集跳过、分析规则、命中率。全部离线虚构帖子。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from zhiku.config import Paths
from zhiku.news.aggregate import cluster_items
from zhiku.news.grades import REVIEW_FIELDS
from zhiku.news.models import ClusteredNews, NewsItem
from zhiku.news.opportunities import build_opportunities
from zhiku.news.runner import run_news
from zhiku.news.sources import fetch_live
from zhiku.news.xwatch import (
    SKIP_NO_TOKEN,
    XAccount,
    XCall,
    XSkipped,
    annotate_resonance,
    apply_fomc_boost,
    bearer_token,
    classify_call,
    example_watchlist_path,
    extract_tickers,
    fetch_live_posts,
    infer_direction,
    items_from_api_payload,
    load_fixture_posts,
    load_watchlist,
    load_x_calls,
    match_newswire,
    summarize_x_hits,
    update_x_weights,
)
from zhiku.notify.render import render
from zhiku.pipeline.settings import NewsSettings, NewsSourceFlags
from zhiku.screens.fundamentals import Fundamentals


def _twitter_only() -> NewsSettings:
    flags = {name: False for name in NewsSourceFlags.model_fields}
    flags["twitter"] = True
    return NewsSettings(sources=NewsSourceFlags(**flags), x_throttle_seconds=0.0)


def test_example_watchlist_has_required_public_handles() -> None:
    path = example_watchlist_path()
    assert path is not None and path.is_file()
    wl = load_watchlist(path)
    by_cat: dict[str, set[str]] = {}
    for acct in wl.accounts:
        by_cat.setdefault(acct.category, set()).add(acct.handle)
    assert by_cat["newswire"] == {"DeItaone", "FirstSquawk", "SquawkCNBC", "zerohedge"}
    assert {
        "NickTimiraos",
        "elerianm",
        "biancoresearch",
        "MacroAlf",
        "LynAldenContact",
        "RayDalio",
        "LizAnnSonders",
        "lisaabramowicz1",
        "TheStalwart",
        "KobeissiLetter",
        "charliebilello",
    } <= by_cat["macro_fed"]
    assert {
        "michaeljburry",
        "BillAckman",
        "CathieDWood",
        "chamath",
        "Jkylebass",
        "GoldmanSachs",
    } <= by_cat["institutional"]
    assert {"unusual_whales", "markminervini"} <= by_cat["flow_trader"]
    assert "jimcramer" in by_cat["contrarian"]
    for acct in wl.accounts:
        assert acct.category in {
            "newswire",
            "macro_fed",
            "institutional",
            "flow_trader",
            "contrarian",
        }


def test_skip_without_bearer_annotates(monkeypatch) -> None:
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    assert bearer_token() is None
    settings = _twitter_only()
    try:
        fetch_live_posts(date(2026, 10, 20), settings)
        raise AssertionError("应跳过")
    except XSkipped as exc:
        assert "Bearer" in str(exc)
        assert "跳过" in str(exc)
    items, errors = fetch_live(date(2026, 10, 20), settings)
    assert items == []
    assert errors
    assert any("Bearer" in e.reason and "跳过" in e.reason for e in errors)
    assert SKIP_NO_TOKEN.split("（")[0] in errors[0].reason


def test_adapter_uses_bearer_cache_and_throttle(monkeypatch, tmp_path: Path) -> None:
    from zhiku.data.cache import SQLiteCache
    from zhiku.news import xwatch as xmod

    calls: list[tuple[str, dict | None]] = []

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs.get("headers")))
        if "/users/by" in url:
            return json.dumps(
                {"data": [{"id": "1", "username": "DeItaone", "name": "Walter"}]}
            )
        return json.dumps(
            {
                "data": [
                    {
                        "id": "t1",
                        "text": "FOMC fictional $AAPL hold (not a real tweet).",
                        "created_at": "2026-10-20T00:00:00Z",
                        "entities": {"cashtags": [{"tag": "AAPL"}]},
                    }
                ]
            }
        )

    monkeypatch.setenv("X_BEARER_TOKEN", "test-token-not-real")
    monkeypatch.setattr(xmod, "http_get", fake_get)
    monkeypatch.setattr(xmod, "_x_sleep", lambda _s: None)
    wl = load_watchlist(example_watchlist_path())
    wl.accounts = [a for a in wl.accounts if a.handle == "DeItaone"]
    cache = SQLiteCache(tmp_path / "c.sqlite")
    settings = _twitter_only()
    items = fetch_live_posts(
        date(2026, 10, 20), settings, cache, watchlist=wl
    )
    assert items
    assert items[0].x_handle == "DeItaone"
    assert "AAPL" in items[0].tickers
    assert calls and all(
        (h or {}).get("Authorization") == "Bearer test-token-not-real" for _, h in calls
    )
    n_first = len(calls)
    again = fetch_live_posts(date(2026, 10, 20), settings, cache, watchlist=wl)
    assert again
    assert len(calls) == n_first  # 缓存命中，不再请求


def test_newswire_needs_grade12_match_for_fact(examples_dir: Path) -> None:
    from zhiku.news.sources import load_fixture_news

    wl = load_watchlist(example_watchlist_path())
    news_items, _ = load_fixture_news(examples_dir / "fixtures" / "news.yaml")
    x_items = load_fixture_posts(examples_dir / "fixtures" / "x_posts.yaml", wl)
    clusters = cluster_items(news_items + x_items, None)
    match_newswire(clusters)
    by_handle = {
        i.x_handle: c
        for c in clusters
        for i in c.items
        if i.x_handle in {"DeItaone", "FirstSquawk"}
    }
    delta = by_handle["DeItaone"]
    squawk = by_handle["FirstSquawk"]
    assert delta.newswire_matched is True
    assert "事实" in delta.tags
    assert squawk.newswire_matched is False
    assert "事实" not in squawk.tags
    assert "未经证实" in squawk.tags


def test_institutional_review_writes_holdings_and_conflicts() -> None:
    from zhiku.news.grades import review_institution

    acct = XAccount(
        handle="michaeljburry",
        category="institutional",
        known_holdings=["AAPL"],
        conflicts="对冲基金经理（测试占位，非真实账户）。",
    )
    cluster = ClusteredNews(
        fingerprint="b",
        symbol="AAPL",
        headline="Fictional: reducing exposure to $AAPL",
        catalyst="other",
        source_grade=3,
        items=[
            NewsItem(
                id="x",
                source="x:michaeljburry",
                source_kind="rating",
                headline="Fictional: reducing exposure to $AAPL",
                source_grade=3,
                x_handle="michaeljburry",
                x_category="institutional",
                x_direction="看空",
                tickers=["AAPL"],
            )
        ],
    )
    review = review_institution(cluster, account=acct)
    for field in REVIEW_FIELDS:
        assert getattr(review, field), field
    assert "AAPL" in review.known_holdings
    assert "数量" not in review.known_holdings or "无数量" in review.known_holdings
    assert "对冲基金" in review.conflicts


def test_extract_tickers_and_opportunity_link(examples_dir: Path, tmp_kb: Path) -> None:
    assert extract_tickers("watching $AAPL and 000333 plus 0700.HK") == [
        "AAPL",
        "000333",
        "0700.HK",
    ]
    assert infer_direction("reducing exposure to $AAPL") == "看空"
    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    bundle = run_news(paths, date(2026, 10, 20), provider_kind="fixture")
    aapl_opps = [o for o in bundle.opportunities if o.get("symbol") == "AAPL"]
    assert aapl_opps
    handles = {p["handle"] for p in bundle.kol_feed}
    assert {"michaeljburry", "CathieDWood", "jimcramer"} <= handles


def test_resonance_and_reverse_risk(examples_dir: Path) -> None:
    wl = load_watchlist(example_watchlist_path())
    items = load_fixture_posts(examples_dir / "fixtures" / "x_posts.yaml", wl)
    clusters = cluster_items(items, None)
    reports = annotate_resonance(clusters, min_accounts=3, extreme_share=0.80)
    assert reports
    aapl = next((r for r in reports if r["key"] == "AAPL"), None)
    assert aapl is not None
    assert aapl["resonance"] is True
    assert aapl["reverse_risk"] is True
    assert any(c.reverse_risk for c in clusters)


def test_macro_fed_fomc_window_boost() -> None:
    cluster = ClusteredNews(
        fingerprint="m",
        headline="FOMC commentary",
        catalyst="policy",
        x_category="macro_fed",
        items=[
            NewsItem(
                id="n",
                source="x:NickTimiraos",
                headline="FOMC commentary",
                x_handle="NickTimiraos",
                x_category="macro_fed",
            )
        ],
    )
    apply_fomc_boost([cluster], active=False)
    assert cluster.fomc_boost is False
    apply_fomc_boost([cluster], active=True)
    assert cluster.fomc_boost is True
    settings = NewsSettings()
    from zhiku.news.xwatch import fomc_weight

    assert fomc_weight(cluster, settings) > 1.0


def test_weekly_hitrate_reweights(examples_dir: Path) -> None:
    calls = load_x_calls(examples_dir / "fixtures" / "x_calls.yaml")
    assert calls
    assert classify_call(XCall(handle="michaeljburry", date=date(2026, 10, 13), direction="看空", t1=-0.01)) == "hit"
    hits = summarize_x_hits(calls)
    by = {a["handle"]: a for a in hits["accounts"]}
    assert by["michaeljburry"]["hit_rate"] == 1.0
    assert by["jimcramer"]["hit_rate"] == 0.0
    weights = update_x_weights({"x_handle": {}}, hits)
    table = weights["x_handle"]
    assert table["michaeljburry"] > 1.0
    assert table["jimcramer"] < 1.0


def test_report_section_and_fictional_posts(tmp_kb: Path, examples_dir: Path, monkeypatch) -> None:
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("ZHIKU_ANALYSIS_LLM", "0")
    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    bundle = run_news(paths, date(2026, 10, 20), provider_kind="fixture")
    assert bundle.kol_feed
    assert any("虚构" in (p.get("headline") or "") or "Fictional" in (p.get("headline") or "") for p in bundle.kol_feed)
    inst = next(p for p in bundle.kol_feed if p["handle"] == "michaeljburry")
    assert inst["institution_review"]
    for field in ("known_holdings", "conflicts"):
        assert inst["institution_review"].get(field)
    md = render(
        "premarket_ah.md.j2",
        news=bundle,
        disclaimer="仅为机会与逻辑，不构成投资建议",
    )
    assert "大V动态" in md
    assert "@michaeljburry" in md
    assert "共振" in md or "反向" in md
    daily_md = render(
        "daily.md.j2",
        daily=type("D", (), {"asof": "2026-10-20", "news": bundle, "macro": None, "notes": [], "candidates": [], "screens": {}, "gated_out": [], "backtests": {}, "audits": [], "quality_rejected": []})(),
    )
    assert "大V动态" in daily_md
    weekly = render(
        "weekly.md.j2",
        review=type("R", (), {"start": date(2026, 10, 19), "end": date(2026, 10, 23), "groups": [], "records": [], "lessons": []})(),
        provider="fixture",
        kb=str(tmp_kb),
        script_hits=[],
        source_hits={},
        x_hits=bundle.x_hits,
    )
    assert "大V命中率" in weekly
    assert "michaeljburry" in weekly
    blob = md + daily_md
    assert "qty" not in blob.lower()
    assert "持仓成本" not in blob


def test_api_payload_parser_no_network() -> None:
    tweets = [
        {
            "id": "99",
            "text": "Fictional upgrade $AAPL",
            "created_at": "2026-10-20T01:00:00Z",
            "entities": {"cashtags": [{"tag": "AAPL"}]},
        }
    ]
    acct = XAccount(handle="GoldmanSachs", category="institutional", conflicts="投行测试占位")
    items = items_from_api_payload(tweets, handle="GoldmanSachs", account=acct)
    assert items[0].source_grade == 3
    assert items[0].x_category == "institutional"
    assert items[0].tickers == ["AAPL"]


def test_x_tickers_enter_opportunity_radar() -> None:
    cluster = ClusteredNews(
        fingerprint="t",
        headline="Fictional flow $AAPL",
        catalyst="other",
        tags=["情绪", "未经证实"],
        source_grade=4,
        tickers=["AAPL"],
        items=[
            NewsItem(
                id="f",
                source="x:unusual_whales",
                source_kind="social",
                headline="Fictional flow $AAPL",
                source_grade=4,
                x_handle="unusual_whales",
                x_category="flow_trader",
                tickers=["AAPL"],
                x_direction="看多",
            )
        ],
        resonance=True,
        reverse_risk=True,
    )
    fund = Fundamentals(symbol="AAPL", name="示例美股", industry="科技", industry_leader=True)
    opps = build_opportunities(
        [cluster],
        funds={"AAPL": fund},
        quality_passed={"AAPL"},
        screen_scores={},
        qlib_scores={},
        holding_symbols=None,
        prices={"AAPL": 100.0},
        hist={},
        weights={"source": {}, "catalyst": {}, "x_handle": {"unusual_whales": 1.2}},
        settings=NewsSettings(),
    )
    assert opps
    assert opps[0].symbol == "AAPL"
    assert opps[0].tier == "只观察"
    assert "反向" in (opps[0].risk or "")
