"""事件驱动层：预期差、新颖度、拥挤、传导、止损、应急、PIT。全部离线虚构。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import json
import pandas as pd
import pytest

from zhiku.calendar import SHANGHAI
from zhiku.config import Paths
from zhiku.events.blackswan import detect_emergency
from zhiku.events.crowding import crowding_adjust, summarize_by_crowding
from zhiku.events.graph import impacts_for_cluster, load_supply_chains
from zhiku.events.invalidation import attach_invalidation, expiry_alerts, time_stop_days
from zhiku.events.log import DecisionLog
from zhiku.events.novelty import earliest_executable, jaccard, novelty_against, session_of
from zhiku.events.pit import (
    LookAheadError,
    asof_snapshot,
    assert_no_lookahead,
    load_pit_rows,
    universe_with_delist,
)
from zhiku.events.surprise import implied_move_from_straddle, surprise_component, vs_implied
from zhiku.macro.chains import load_chains
from zhiku.macro.surprise import surprise_z
from zhiku.models import Market, MissingData
from zhiku.news.models import ClusteredNews, NewsItem, Opportunity
from zhiku.news.runner import run_news
from zhiku.news.study import EventSample
from zhiku.pipeline.settings import EventSettings, NewsSettings, load_pipeline_settings
from zhiku.screens.fundamentals import Fundamentals


def test_surprise_missing_and_implied_straddle() -> None:
    z, missing = surprise_z(1.35, None, 0.1)
    assert z is None and missing and "缺数据" in missing
    z, _ = surprise_z(1.35, 1.20, 0.10)
    assert z == pytest.approx(1.5)
    move = implied_move_from_straddle(100.0, 3.0, 3.0)
    assert move == pytest.approx(0.06)
    assert isinstance(implied_move_from_straddle(None, 1, 1), MissingData)
    assert "缺数据" in vs_implied(None, 0.03)


def test_score_driven_by_surprise_not_goodness() -> None:
    beat = ClusteredNews(fingerprint="a", headline="快报", catalyst="earnings_beat", surprise_z=1.6)
    miss = ClusteredNews(
        fingerprint="b", headline="快报", catalyst="earnings_beat", surprise_z=-1.6
    )
    none = ClusteredNews(fingerprint="c", headline="快报", catalyst="earnings_beat")
    assert surprise_component(beat) == surprise_component(miss)
    assert surprise_component(none) == 0.5
    priced = ClusteredNews(
        fingerprint="d",
        headline="快报",
        catalyst="earnings_beat",
        surprise_z=1.6,
        implied_move=0.03,
        actual_move=0.04,
    )
    assert surprise_component(priced) < surprise_component(beat)


def test_novelty_session_t1_and_decay() -> None:
    assert novelty_against("示例家电发布业绩快报", ["示例家电发布业绩快报：利润好于预期"]) < 0.5
    assert jaccard("hello world", "hello world") == 1.0
    pub = datetime(2026, 10, 20, 16, 0, tzinfo=SHANGHAI)
    session, mkt = session_of(pub, Market.CN)
    assert session == "盘后" and mkt == "CN"
    text = earliest_executable("盘后", Market.CN, date(2026, 10, 20))
    assert "T+1" in text
    none_s, _ = session_of(None, Market.CN)
    assert none_s == "缺数据"


def test_crowding_good_news_penalty() -> None:
    settings = EventSettings()
    factor, note = crowding_adjust(0.85, good_news=True, settings=settings)
    assert factor < 1 and "利好出尽" in note
    factor_low, note_low = crowding_adjust(0.1, good_news=True, settings=settings)
    assert factor_low > 1
    factor_miss, note_miss = crowding_adjust(None, good_news=True, settings=settings)
    assert factor_miss == 1.0 and "缺数据" in note_miss
    samples = [
        EventSample(catalyst="earnings_beat", t1=0.02, crowding_bucket="low"),
        EventSample(catalyst="earnings_beat", t1=-0.02, crowding_bucket="high"),
    ]
    grouped = summarize_by_crowding(samples)
    assert grouped["low"]["winrate_t1"] == 1.0
    assert grouped["high"]["winrate_t1"] == 0.0


def test_supply_chain_first_and_second_hop() -> None:
    chains = load_supply_chains()
    geos = load_chains()
    cluster = ClusteredNews(
        fingerprint="g",
        symbol="AAPL",
        headline="苹果供应链关注 iPhone 需求（虚构）",
        catalyst="other",
        items=[NewsItem(id="1", source="fixture", headline="苹果供应链关注 iPhone")],
    )
    hits = impacts_for_cluster(cluster, chains, geos)
    hops = {(h.hop, h.symbol) for h in hits if h.kind == "company"}
    assert (1, "000333") in hops
    assert (2, "0700.HK") in hops
    geo = ClusteredNews(
        fingerprint="h",
        headline="虚构：霍尔木兹海峡过境扰动引发原油关注",
        catalyst="policy",
        items=[NewsItem(id="2", source="rss", headline="霍尔木兹")],
    )
    geo_hits = impacts_for_cluster(geo, chains, geos)
    assert any(h.kind == "geo" and h.hop in {1, 2} for h in geo_hits)


def test_invalidation_and_time_stop_expiry() -> None:
    cluster = ClusteredNews(
        fingerprint="i",
        symbol="000333",
        headline="示例快报",
        catalyst="earnings_beat",
        market=Market.CN,
        items=[],
    )
    opp = Opportunity(
        symbol="000333",
        name="示例",
        direction="做多",
        tier="交易档",
        score=10,
        catalyst="earnings_beat",
        trigger_headline="示例快报",
        entry_logic="x",
        risk="r",
        price=10.0,
    )
    attach_invalidation(
        [opp],
        [cluster],
        on=date(2026, 10, 20),
        event_settings=EventSettings(),
        news_settings=NewsSettings(),
    )
    assert opp.invalidation and "失效" in opp.invalidation
    assert opp.time_stop == time_stop_days("earnings_beat", EventSettings())
    assert opp.time_stop_on is not None
    p = Path("/tmp/zhiku-opps-test.jsonl")
    p.write_text(
        json.dumps(
            {
                "symbol": "000333",
                "trigger_headline": "示例快报",
                "time_stop_on": "2026-10-20",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    alerts = expiry_alerts(p, date(2026, 10, 21))
    assert alerts and alerts[0].reason == "未兑现，撤"


def test_blackswan_emergency_and_playbook() -> None:
    settings = EventSettings(vix_spike=0.2, index_intraday_drop=0.03, emergency_min_sources=2)
    clusters = [
        ClusteredNews(
            fingerprint="s1",
            headline="制裁升级（虚构）",
            items=[
                NewsItem(
                    id="a",
                    source="DeItaone",
                    headline="制裁升级",
                    x_handle="DeItaone",
                    x_category="newswire",
                )
            ],
        ),
        ClusteredNews(
            fingerprint="s2",
            headline="冲突扩大（虚构）",
            items=[
                NewsItem(
                    id="b",
                    source="FirstSquawk",
                    headline="冲突扩大",
                    x_handle="FirstSquawk",
                    x_category="newswire",
                )
            ],
        ),
    ]
    em = detect_emergency(
        clusters=clusters,
        settings=settings,
        shock={"vix_change": 0.25, "index_intraday": -0.04},
        holdings={"510300"},
        playbook=["先降风险，不扩大暴露", "不抄底"],
    )
    assert em.active is True
    assert em.freeze_adds is True
    assert "510300" in em.affected
    assert any("不抄底" in p for p in em.playbook)
    quiet = detect_emergency(clusters=[], settings=settings, shock={"vix_change": 0.01})
    assert quiet.active is False


def test_decision_log_append_only(tmp_path) -> None:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append("opportunity", {"symbol": "000333", "score": 1})
    log.append("opportunity", {"symbol": "AAPL", "score": 2})
    rows = log.read()
    assert len(rows) == 2
    assert all("ts" in r for r in rows)
    text = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    assert text.count("\n") == 2
    # 没有改写接口：再读仍是两行
    assert len(log.read()) == 2


def test_pit_no_lookahead_and_delist(examples_dir) -> None:
    rows = load_pit_rows(examples_dir / "fixtures" / "pit_fundamentals.yaml")
    snap = asof_snapshot(rows, date(2026, 10, 20))
    assert snap["000333"].roe == 0.21  # 10-25 公告不可用
    future = asof_snapshot(rows, date(2026, 10, 26))
    assert future["000333"].roe == 0.99
    with pytest.raises(LookAheadError):
        fake = Fundamentals(symbol="000333", name="x", roe=0.99)
        assert_no_lookahead(rows, date(2026, 10, 20), [fake])
    assert "GONE" not in universe_with_delist(
        ["000333", "GONE"], {"GONE": date(2026, 6, 1)}, date(2026, 10, 20)
    )
    assert "GONE" in universe_with_delist(["GONE"], {"GONE": date(2026, 6, 1)}, date(2026, 5, 1))


def test_event_backtest_uses_time_stop() -> None:
    from zhiku.backtest.engine import backtest_event_trades

    idx = pd.bdate_range("2026-10-01", periods=12)
    px = pd.DataFrame({"AAPL": [100 + i for i in range(12)]}, index=idx)
    result = backtest_event_trades(
        px,
        [
            {
                "symbol": "AAPL",
                "entry": "2026-10-02",
                "direction": "做多",
                "time_stop": 3,
                "stop_pct": 0.5,
            }
        ],
    )
    assert result.notes
    assert result.equity


def test_half_life_uses_event_study() -> None:
    from zhiku.events.novelty import attach_novelty, estimate_half_life_hours

    hl = estimate_half_life_hours(0.04, 0.005, 48.0)
    assert 6.0 <= hl < 40.0
    cluster = ClusteredNews(
        fingerprint="hl",
        headline="半衰期样本",
        catalyst="earnings_beat",
        items=[NewsItem(id="hl1", source="fixture", headline="半衰期样本")],
    )
    attach_novelty(
        [cluster],
        on=date(2026, 10, 20),
        settings=EventSettings(),
        journal=None,
        hist_avgs={"earnings_beat": (0.04, 0.005)},
        now=datetime(2026, 10, 20, 12, 0, tzinfo=SHANGHAI),
    )
    assert cluster.half_life_hours is not None
    assert cluster.half_life_hours < 40.0


def test_macro_surprise_missing_on_fomc(examples_dir) -> None:
    from zhiku.events.surprise import attach_surprise, load_consensus

    cards = load_consensus(examples_dir / "fixtures" / "earnings_consensus.yaml")
    cluster = ClusteredNews(
        fingerprint="fomc",
        headline="FOMC 将于本周议息，市场关注点阵图",
        catalyst="policy",
        items=[NewsItem(id="f", source="cls", headline="FOMC")],
    )
    attach_surprise([cluster], cards, {})
    assert cluster.surprise_z is None
    assert cluster.surprise_missing and "缺数据" in cluster.surprise_missing


def test_csv_supply_chain(tmp_path) -> None:
    csv_path = tmp_path / "supply_chains.csv"
    csv_path.write_text(
        "chain_id,chain_name,keywords,symbol,name,role,hop,direction\n"
        "demo_csv,CSV示例链,CSVFOO,AAPL,品牌,brand,0,同向\n"
        "demo_csv,CSV示例链,CSVFOO,000333,一阶,supplier,1,同向\n"
        "demo_csv,CSV示例链,CSVFOO,0700.HK,二阶,component,2,同向\n",
        encoding="utf-8",
    )
    chains = load_supply_chains(csv_path)
    cluster = ClusteredNews(
        fingerprint="csv",
        symbol="AAPL",
        headline="CSVFOO 传导测试",
        catalyst="other",
        items=[NewsItem(id="c", source="fixture", headline="CSVFOO")],
    )
    hits = impacts_for_cluster(cluster, chains, [])
    hops = {(h.hop, h.symbol) for h in hits if h.kind == "company"}
    assert (1, "000333") in hops
    assert (2, "0700.HK") in hops


def test_emergency_english_keyword_word_boundary() -> None:
    from zhiku.events.blackswan import detect_emergency

    settings = EventSettings(emergency_min_sources=2)
    clusters = [
        ClusteredNews(
            fingerprint="w1",
            headline="software warning went forward by default",
            items=[NewsItem(id="a", source="DeItaone", headline="software warning went forward")],
        ),
        ClusteredNews(
            fingerprint="w2",
            headline="another forward warning",
            items=[NewsItem(id="b", source="FirstSquawk", headline="another forward warning")],
        ),
    ]
    quiet = detect_emergency(clusters=clusters, settings=settings, shock={})
    assert quiet.active is False


def test_freeze_adds_on_evaluations() -> None:
    from types import SimpleNamespace

    from zhiku.events.invalidation import apply_event_discipline
    from zhiku.events.models import Emergency, ExpiryAlert

    ev = SimpleNamespace(warnings=[], freeze_adds=False)
    em = Emergency(active=True, freeze_adds=True, reasons=["VIX 冲击"], playbook=["不抄底"])
    banners = apply_event_discipline(
        [ev], emergency=em, expiries=[ExpiryAlert(symbol="AAPL", time_stop_on=date(2026, 10, 20))]
    )
    assert ev.freeze_adds is True
    assert any("禁加" in b for b in banners)
    assert any("未兑现，撤" in b for b in banners)


def test_decision_log_strips_qty(tmp_path) -> None:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append("opportunity", {"symbol": "000333", "qty": 100, "cost": 12.3, "score": 1})
    rows = log.read()
    assert "qty" not in rows[0]["payload"]
    assert "cost" not in rows[0]["payload"]
    assert rows[0]["payload"]["score"] == 1
    assert not hasattr(log, "update")
    assert not hasattr(log, "delete")


def test_pit_asof_universe_no_lookahead(examples_dir) -> None:
    from zhiku.events.pit import load_asof_universe

    rows = load_asof_universe(
        fundamentals_path=examples_dir / "fixtures" / "fundamentals.yaml",
        pit_path=examples_dir / "fixtures" / "pit_fundamentals.yaml",
        delist_path=examples_dir / "fixtures" / "delisted.yaml",
        asof=date(2026, 10, 20),
    )
    by = {r.symbol: r for r in rows}
    assert by["000333"].roe == 0.21
    assert "GONE" not in by
    later = load_asof_universe(
        fundamentals_path=examples_dir / "fixtures" / "fundamentals.yaml",
        pit_path=examples_dir / "fixtures" / "pit_fundamentals.yaml",
        delist_path=examples_dir / "fixtures" / "delisted.yaml",
        asof=date(2026, 10, 26),
    )
    assert {r.symbol: r for r in later}["000333"].roe == 0.99


def test_event_study_groups_crowding() -> None:
    from zhiku.news.study import event_study_from_prices

    idx = pd.bdate_range("2026-10-01", periods=8)
    px = pd.DataFrame(
        {"AAPL": [100, 101, 102, 103, 104, 105, 106, 107], "510300": [4] * 8}, index=idx
    )
    samples = event_study_from_prices(
        px,
        [("AAPL", pd.Timestamp("2026-10-01"), "buyback")],
        crowding={"AAPL": "high"},
    )
    assert samples[0].crowding_bucket == "high"
    assert samples[0].t1 is not None


def test_run_news_wires_event_layer(tmp_kb, examples_dir, monkeypatch) -> None:
    monkeypatch.setenv("ZHIKU_ANALYSIS_LLM", "0")
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    bundle = run_news(paths, date(2026, 10, 20), provider_kind="fixture")
    c333 = next(
        c
        for c in bundle.clusters
        if c.get("symbol") == "000333" and c.get("surprise_z") is not None
    )
    assert c333["surprise_z"] is not None
    aapl = next(
        (c for c in bundle.clusters if c.get("symbol") == "AAPL" and c.get("crowding") is not None),
        None,
    )
    assert aapl is not None
    assert aapl.get("crowding_note")
    opps = bundle.opportunities
    if opps:
        assert any(o.get("invalidation") for o in opps)
        assert any(o.get("time_stop") for o in opps)
    log_path = tmp_kb / "journal" / "decisions.jsonl"
    assert log_path.is_file()
    blob = log_path.read_text(encoding="utf-8")
    assert "opportunity" in blob
    assert "judgment" in blob
    opp_log = tmp_kb / "journal" / "opportunities.jsonl"
    assert opp_log.is_file()
    fomc = next((c for c in bundle.clusters if "FOMC" in (c.get("headline") or "")), None)
    assert fomc is not None
    assert fomc.get("surprise_missing") and "缺数据" in str(fomc.get("surprise_missing"))
    from zhiku.notify.render import render

    md = render("premarket_ah.md.j2", news=bundle, disclaimer="仅为机会与逻辑，不构成投资建议")
    assert "大V动态" in md
    settings = load_pipeline_settings()
    assert settings.events.enabled is True
    assert settings.news.score_weights.surprise > 0
