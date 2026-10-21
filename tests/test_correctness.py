"""正确性校验：复权混入、闸门口径、标签前视、应急价格确认、日历白名单、机会字段。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from zhiku.backtest.engine import compute_benchmarks, sanitize_mixed_adjust
from zhiku.data.store import MarketStore
from zhiku.data.sync import _upsert_spot_as_bars, repair_mixed_qfq
from zhiku.events.blackswan import detect_emergency
from zhiku.factors.features import label_forward
from zhiku.factors.qlib_adapter import pick_better
from zhiku.macro.models import MacroEvent
from zhiku.macro.sources import official_core_events, parse_ws_calendar_row
from zhiku.macro.windows import build_windows
from zhiku.news.models import ClusteredNews, NewsItem
from zhiku.news.opportunities import build_opportunities
from zhiku.news.tickers import attach_cn_tickers
from zhiku.pipeline.daily import DailyResult
from zhiku.pipeline.settings import BacktestSettings, EventSettings, NewsSettings, load_pipeline_settings
from zhiku.screens.fundamentals import Fundamentals


def test_spot_does_not_write_raw_into_qfq(tmp_path) -> None:
    store = MarketStore(tmp_path / "data")
    hist = pd.DataFrame(
        [
            {
                "symbol": "600519",
                "market": "CN",
                "date": date(2026, 10, 20),
                "adjust": "qfq",
                "open": 10000.0,
                "high": 10100.0,
                "low": 9900.0,
                "close": 10000.0,
                "volume": 1e6,
            },
            {
                "symbol": "600519",
                "market": "CN",
                "date": date(2026, 10, 20),
                "adjust": "none",
                "open": 1000.0,
                "high": 1010.0,
                "low": 990.0,
                "close": 1000.0,
                "volume": 1e6,
            },
        ]
    )
    store.upsert_bars(hist)
    store.set_state("bars:qfq", "600519", last_ok=date(2026, 10, 20), status="ok")
    store.set_state("bars:none", "600519", last_ok=date(2026, 10, 20), status="ok")
    n = _upsert_spot_as_bars(
        store,
        [{"symbol": "600519", "close": 1100.0, "open": 1080.0, "high": 1110.0, "low": 1070.0}],
        date(2026, 10, 21),
        symbols=["600519"],
        only_existing=True,
    )
    assert n == 1
    qfq = store.bars(["600519"], adjust="qfq")
    last = qfq[qfq["date"] == pd.Timestamp("2026-10-21")].iloc[0]
    assert abs(float(last["close"]) - 11000.0) < 1e-6
    none = store.bars(["600519"], adjust="none")
    raw = none[none["date"] == pd.Timestamp("2026-10-21")].iloc[0]
    assert abs(float(raw["close"]) - 1100.0) < 1e-6


def test_repair_and_sanitize_mixed_qfq(tmp_path) -> None:
    store = MarketStore(tmp_path / "data")
    rows = []
    for i, close in enumerate([100.0, 101.0, 30.0]):
        d = date(2026, 10, 19 + i)
        rows.append(
            {
                "symbol": "000333",
                "market": "CN",
                "date": d,
                "adjust": "qfq",
                "close": close,
                "open": close,
                "high": close,
                "low": close,
            }
        )
        rows.append(
            {
                "symbol": "000333",
                "market": "CN",
                "date": d,
                "adjust": "none",
                "close": 30.0 if i == 2 else 20.0 + i,
                "open": 30.0,
                "high": 30.0,
                "low": 30.0,
            }
        )
    store.upsert_bars(pd.DataFrame(rows))
    store.set_state("bars:qfq", "000333", last_ok=date(2026, 10, 21), status="ok")
    n = repair_mixed_qfq(store, jump=0.20)
    assert n == 1
    left = store.bars(["000333"], adjust="qfq")
    assert pd.Timestamp("2026-10-21") not in set(pd.to_datetime(left["date"]))
    qfq = pd.DataFrame(
        {"000333": [100.0, 101.0, 30.0]},
        index=pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"]),
    )
    none = pd.DataFrame(
        {"000333": [20.0, 21.0, 30.0]},
        index=qfq.index,
    )
    clean = sanitize_mixed_adjust(qfq, none, jump=0.20)
    last_ts = pd.Timestamp("2026-10-21")
    if last_ts in clean.index:
        assert pd.isna(clean.loc[last_ts, "000333"])
    else:
        assert last_ts not in clean.index


def test_benchmark_sanity_gap(tmp_path) -> None:
    idx = pd.to_datetime(["2024-10-31", "2025-10-31", "2026-10-21"])
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 1.1, 1.2],
            "BBB": [1.0, 1.05, 1.1],
            "510300": [1.0, 1.08, 1.15],
        },
        index=idx,
    )
    benches = compute_benchmarks(prices, BacktestSettings())
    assert "universe_ew" in benches
    assert "csi300" in benches
    assert benches["universe_ew"].annualized is not None
    assert benches["csi300"].annualized is not None
    gap = abs(benches["universe_ew"].annualized - benches["csi300"].annualized)
    assert gap < 0.25


def test_gated_formula_excluded_from_total() -> None:
    from zhiku.factors.merge import merge_scores
    from zhiku.pipeline.settings import MergeSettings
    from zhiku.screens.runner import ScreenBundle, formula_scores
    from zhiku.screens.greenblatt import ScreenRow

    all_screens = {
        "greenblatt": [ScreenRow(symbol="600519", name="示例", score=90, breakdown={})],
        "multifactor": [ScreenRow(symbol="000333", name="家电", score=80, breakdown={})],
    }
    active = {"multifactor": all_screens["multifactor"]}
    formula_all = formula_scores(ScreenBundle(2, ["000333"], [], all_screens))
    merge_formula = formula_scores(ScreenBundle(2, ["000333"], [], active))
    merged = merge_scores(
        {"600519": "示例", "000333": "家电"},
        merge_formula,
        {"600519": 70.0, "000333": 40.0},
        MergeSettings(formula=0.6, qlib=0.4),
    )
    by = {c.symbol: c for c in merged}
    assert "greenblatt" not in by["600519"].screens
    assert by["600519"].formula_score == 0.0
    assert by["600519"].source == "qlib"
    assert "multifactor" in by["000333"].screens
    assert formula_all["600519"]["greenblatt"] == 90


def test_label_forward_uses_next_open() -> None:
    bars = pd.DataFrame(
        [
            {"date": "2026-10-19", "symbol": "000333", "open": 10.0, "close": 10.0},
            {"date": "2026-10-20", "symbol": "000333", "open": 11.0, "close": 12.0},
            {"date": "2026-10-21", "symbol": "000333", "open": 12.0, "close": 13.0},
        ]
    )
    lab = label_forward(bars, horizon=2)
    val = float(lab.xs(pd.Timestamp("2026-10-19"), level="date").iloc[0])
    # T+1 open 11 → T+2 close 13
    assert abs(val - (13.0 / 11.0 - 1.0)) < 1e-9


def test_pick_better_prefers_rank_icir() -> None:
    winner = pick_better(
        {"kind": "lgb_lite", "rank_ic": 0.02588, "rank_icir": 0.004, "ic": 0.051},
        {"kind": "qlib_alpha158", "rank_ic": 0.02468, "rank_icir": 0.183, "ic": 0.007},
    )
    assert winner["kind"] == "qlib_alpha158"


def test_emergency_keywords_need_price() -> None:
    settings = EventSettings(vix_spike=0.2, index_intraday_drop=0.03, emergency_min_sources=2)
    clusters = [
        ClusteredNews(
            fingerprint="s1",
            headline="制裁升级（虚构）",
            items=[NewsItem(id="a", source="DeItaone", headline="制裁升级", x_handle="DeItaone")],
        ),
        ClusteredNews(
            fingerprint="s2",
            headline="冲突扩大（虚构）",
            items=[NewsItem(id="b", source="FirstSquawk", headline="冲突扩大", x_handle="FirstSquawk")],
        ),
    ]
    quiet = detect_emergency(clusters=clusters, settings=settings, shock={})
    assert quiet.active is False
    assert quiet.price_confirmed is False
    assert any("价格未确认" in r for r in quiet.reasons)
    hot = detect_emergency(
        clusters=clusters,
        settings=settings,
        shock={"vix_change": 0.25, "index_intraday": -0.04},
    )
    assert hot.active is True
    assert hot.price_confirmed is True
    assert any("VIX" in r for r in hot.reasons)


def test_conference_not_window_and_fomc_core() -> None:
    row = {
        "时间": "2026-10-23 09:00",
        "地区": "中国",
        "事件": "2026世界智能网联汽车大会",
        "重要性": "高",
    }
    ev = parse_ws_calendar_row(row, asof=date(2026, 10, 23), index=0)
    assert ev is not None
    assert ev.importance <= 2
    assert ev.event_type in {"CONFERENCE", "OTHER"}
    settings = load_pipeline_settings()
    windows = build_windows([ev], settings.macro)
    assert windows == []
    core = official_core_events(date(2026, 10, 21), 7)
    names = " ".join(e.name for e in core)
    assert "FOMC" in names
    assert any(e.scheduled_at and e.scheduled_at.date().isoformat() >= "2026-10-27" for e in core)
    fomc = MacroEvent(
        id="f",
        event_type="FOMC",
        name="FOMC利率决议",
        importance=5,
        scheduled_at=core[-1].scheduled_at,
    )
    assert build_windows([fomc], settings.macro)


def test_opportunity_has_reason_and_cn_ticker() -> None:
    cluster = ClusteredNews(
        fingerprint="cn",
        headline="格力电器回购进展（虚构）",
        catalyst="buyback",
        tags=["事实"],
        source_grade=2,
        source_grade_label="二级·媒体",
        items=[
            NewsItem(
                id="n",
                source="eastmoney",
                source_kind="media",
                headline="格力电器回购进展（虚构）",
                summary="000333 公告回购",
                source_grade=2,
            )
        ],
    )
    n = attach_cn_tickers([cluster], {"格力电器": "000333"})
    assert n >= 1
    assert cluster.symbol == "000333"
    fund = Fundamentals(symbol="000333", name="格力电器", industry="家电")
    opps = build_opportunities(
        [cluster],
        funds={"000333": fund},
        quality_passed=set(),
        screen_scores={},
        qlib_scores={},
        holding_symbols=None,
        prices={"000333": 80.0},
        hist={},
        weights={"source": {}, "catalyst": {}},
        settings=NewsSettings(),
    )
    assert opps
    from zhiku.events.invalidation import attach_invalidation

    attach_invalidation(
        opps, [cluster], on=date(2026, 10, 21), event_settings=EventSettings(), news_settings=NewsSettings()
    )
    assert opps[0].invalidation
    assert opps[0].time_stop
    assert opps[0].trigger_headline
    assert opps[0].source_grade == 2


def test_brief_shows_emergency_reason_and_under_150() -> None:
    from zhiku.notify.render import render

    daily = DailyResult(
        asof="2026-10-21",
        notes=["未过闸公式分不计入总分，只在单票详情（screens_all）展示。"],
        candidates=[
            {
                "symbol": "600519",
                "name": "示例",
                "total": 80,
                "formula_score": 0,
                "qlib_score": 80,
                "screens": {},
            }
        ],
        backtests={
            "greenblatt": {
                "annualized": 0.1,
                "max_drawdown": -0.1,
                "sharpe": 0.5,
                "excess_annualized": 0.02,
                "passed_gate": True,
            },
            "universe_ew": {"annualized": 0.08, "max_drawdown": -0.2, "sharpe": 0.4, "passed_gate": True},
        },
        qlib={
            "kind": "qlib_alpha158",
            "rank_ic": 0.02,
            "rank_icir": 0.18,
            "comparison": {
                "lgb_lite": {"rank_ic": 0.02, "rank_icir": 0.004, "quintiles": {"Q5": 0.01}},
                "qlib_alpha158": {"rank_ic": 0.02, "rank_icir": 0.18, "quintiles": {"Q5": 0.02}},
            },
        },
        news={
            "clusters": [{"symbol": "000333"}],
            "opportunities": [
                {
                    "tier": "交易档",
                    "direction": "做多",
                    "symbol": "000333",
                    "name": "示例家电",
                    "entry_logic": "回购",
                    "invalidation": "跌破止损",
                    "time_stop_on": "2026-10-28",
                    "trigger_headline": "回购（虚构）",
                    "source_grade": 2,
                    "source_grade_label": "二级·媒体",
                }
            ],
            "emergency": {
                "active": False,
                "reasons": ["2 个快讯源同时出现冲突/制裁/监管关键词（仅关键词，价格未确认，不进入应急）"],
                "metrics": {"vix_change": 0.01, "vix_threshold": 0.2, "index_intraday": -0.01, "index_threshold": -0.03},
            },
        },
        macro={
            "geo": {"temperature": "缺数据"},
            "rate_expectation": {"source": "缺数据", "note": "无真实利率"},
            "released": [],
            "calendar_7d": [
                {
                    "event_type": "FOMC",
                    "name": "FOMC 利率决议",
                    "scheduled_shanghai": "2026-10-28 02:00",
                    "importance": 5,
                }
            ],
        },
        universe_size=3,
        missing_ratio=0.1,
    )
    md = render("daily_brief.md.j2", daily=daily)
    assert len(md.splitlines()) <= 150
    assert "价格未确认" in md
    assert "FOMC" in md
    assert "缺数据" in md
    assert "000333" in md
    assert "跌破止损" in md
