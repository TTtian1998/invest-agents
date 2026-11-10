"""Optional context sources fail closed, never substitute demo data for live."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from zhiku.calendar import SHANGHAI
from zhiku.config import Paths
from zhiku.macro import runner as macro_runner
from zhiku.news import runner as news_runner
from zhiku.pipeline.settings import PipelineSettings
from zhiku.pipeline.sources import resolve_source_mode

ON = date(2026, 10, 20)


def _unexpected(*args, **kwargs):
    raise AssertionError("disabled or deterministic execution reached a forbidden operation")


def _clock_at(monkeypatch, module, day: date) -> None:
    original = module.now_shanghai
    monkeypatch.setattr(
        module, "now_shanghai",
        lambda now=None: original(now) if now is not None else datetime.combine(
            day, datetime.min.time(), tzinfo=SHANGHAI,
        ),
    )


def _forbid_fixture_reads(monkeypatch, fixtures: Path) -> None:
    read_text = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path.is_relative_to(fixtures):
            raise AssertionError(f"live must not read fixture {path.name}")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)


@pytest.mark.parametrize(
    ("provider", "enabled", "expected"),
    [
        ("fixture", True, "demo"),
        ("demo", True, "demo"),
        ("live", True, "live"),
        ("disabled", True, "disabled"),
        ("fixture", False, "disabled"),
        ("demo", False, "disabled"),
        ("live", False, "disabled"),
    ],
)
def test_resolve_source_mode(provider, enabled, expected) -> None:
    assert resolve_source_mode(provider, enabled=enabled) == expected


@pytest.mark.parametrize("enabled", [True, False])
def test_invalid_source_never_selects_demo(enabled) -> None:
    with pytest.raises(ValueError, match="Unknown source mode"):
        resolve_source_mode("misspelled-live", enabled=enabled)


@pytest.mark.parametrize(
    ("provider", "enabled"),
    [("live", False), ("fixture", False), ("demo", False), ("disabled", True)],
)
def test_news_disabled_before_sources_history_or_analysis(tmp_path, monkeypatch, provider, enabled):
    from zhiku.events import layer
    from zhiku.news import analyze, xwatch

    paths = Paths.load(kb=tmp_path / "kb", fixtures=tmp_path / "fixtures")
    settings = PipelineSettings()
    settings.news.enabled = enabled
    for name in (
        "load_fixture_news", "fetch_live", "_holdings", "cluster_items",
        "load_event_history", "load_weights", "summarize_samples", "update_weights",
    ):
        monkeypatch.setattr(news_runner, name, _unexpected)
    monkeypatch.setattr(xwatch, "load_watchlist", _unexpected)
    monkeypatch.setattr(layer, "apply_event_layer", _unexpected)
    monkeypatch.setattr(analyze, "rule_card", _unexpected)
    monkeypatch.setattr(analyze, "call_llm_json", _unexpected)

    result = news_runner.run_news(
        paths, ON, provider_kind=provider, settings=settings, include_research=True,
    )

    assert result.source_mode == "disabled"
    assert result.analysis_engine == "disabled"
    assert result.clusters == result.analyses == result.opportunities == []
    assert result.x_hits == result.catalyst_stats == {}
    assert json.loads(result.to_json())["source_mode"] == "disabled"
    assert not paths.kb.exists()


@pytest.mark.parametrize(
    ("provider", "enabled"),
    [("live", False), ("fixture", False), ("demo", False), ("disabled", True)],
)
def test_macro_disabled_before_sources_history_or_analysis(tmp_path, monkeypatch, provider, enabled):
    from zhiku.macro import analyze, sources

    paths = Paths.load(kb=tmp_path / "kb", fixtures=tmp_path / "fixtures")
    settings = PipelineSettings()
    settings.macro.enabled = enabled
    for name in (
        "load_calendar", "live_calendar", "load_history", "load_yaml", "load_geo",
        "_holdings_fundamentals", "build_scripts", "live_fedwatch_approx",
    ):
        monkeypatch.setattr(macro_runner, name, _unexpected)
    monkeypatch.setattr(sources, "live_anchors", _unexpected)
    monkeypatch.setattr(analyze, "rule_card", _unexpected)
    monkeypatch.setattr(analyze, "call_llm_json", _unexpected)

    result = macro_runner.run_macro(paths, ON, provider_kind=provider, settings=settings)

    assert result.source_mode == "disabled"
    assert result.analysis_engine == "disabled"
    assert result.calendar_7d == result.anchors == result.analyses == []
    assert result.rate_expectation == result.geo == {}
    assert not result.freeze_adds
    assert json.loads(result.to_json())["source_mode"] == "disabled"
    assert not paths.kb.exists()


@pytest.mark.parametrize("provider", ["fixture", "demo"])
def test_demo_and_fixture_alias_are_offline_and_deterministic(
    tmp_kb, examples_dir, monkeypatch, provider,
):
    from zhiku.macro import analyze as macro_analyze
    from zhiku.news import analyze as news_analyze

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    monkeypatch.setattr(news_runner, "fetch_live", _unexpected)
    monkeypatch.setattr(macro_runner, "live_calendar", _unexpected)
    # A configured paid LLM must not be called by either context collector.
    for module in (news_analyze, macro_analyze):
        monkeypatch.setattr(module, "llm_enabled", lambda: True)
        monkeypatch.setattr(module, "call_llm_json", _unexpected)

    news = news_runner.run_news(paths, ON, provider_kind=provider)
    macro = macro_runner.run_macro(paths, ON, provider_kind=provider)

    assert news.source_mode == macro.source_mode == "demo"
    assert news.clusters and macro.calendar_7d
    assert news.analyses and macro.analyses
    assert news.analysis_engine == macro.analysis_engine == "rules"
    assert all(not card["deep"] for card in news.analyses + macro.analyses)


def test_news_default_does_not_fit_or_write_history(tmp_kb, examples_dir, monkeypatch):
    from zhiku.events import layer
    from zhiku.news import xwatch

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    settings = PipelineSettings()
    settings.macro.enabled = False
    monkeypatch.setattr(layer, "summarize_samples", _unexpected)
    monkeypatch.setattr(layer, "summarize_by_crowding", _unexpected)
    for name in ("load_event_history", "summarize_samples", "update_weights", "save_weights"):
        monkeypatch.setattr(news_runner, name, _unexpected)
    for name in (
        "load_x_calls", "key_calls_from_clusters", "summarize_x_hits",
        "update_x_weights", "append_x_calls",
    ):
        monkeypatch.setattr(xwatch, name, _unexpected)

    result = news_runner.run_news(paths, ON, provider_kind="demo", settings=settings)

    assert result.clusters
    assert result.catalyst_stats == result.x_hits == {}
    assert not (paths.journal / "news_weights.json").exists()
    assert not (paths.journal / "x_calls.jsonl").exists()
    assert not (paths.journal / "demo" / "news_weights.json").exists()


def test_demo_research_is_explicit_and_cannot_train_live_weights(tmp_kb, examples_dir):
    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    settings = PipelineSettings()
    settings.macro.enabled = False
    live_weights = paths.journal / "news_weights.json"
    original = '{"catalyst": {"policy": 0.75}, "source": {}, "x_handle": {}}\n'
    live_weights.write_text(original, encoding="utf-8")

    result = news_runner.run_news(
        paths, ON, provider_kind="demo", settings=settings, include_research=True,
    )

    assert result.catalyst_stats
    assert result.x_hits["accounts"]
    assert (paths.journal / "demo" / "news_weights.json").is_file()
    assert (paths.journal / "demo" / "x_calls.jsonl").is_file()
    assert live_weights.read_text(encoding="utf-8") == original
    assert not (paths.journal / "x_calls.jsonl").exists()


@pytest.mark.parametrize("failure", ["empty", "cache", "provider"])
def test_live_news_failure_does_not_load_demo(tmp_kb, examples_dir, monkeypatch, failure):
    from zhiku.data import cache
    from zhiku.events import layer

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    settings = PipelineSettings()
    settings.macro.enabled = False
    _forbid_fixture_reads(monkeypatch, paths.fixtures)
    _clock_at(monkeypatch, news_runner, ON)
    monkeypatch.setattr(news_runner, "load_fixture_news", _unexpected)
    monkeypatch.setattr(layer, "live_shock", lambda **kwargs: {})
    calls = []

    def live_fetch(*args, **kwargs):
        calls.append(kwargs["cache"])
        if failure == "provider":
            raise RuntimeError("provider offline")
        return [], []

    def broken_cache(*args, **kwargs):
        raise sqlite3.OperationalError("cache is unavailable")

    monkeypatch.setattr(news_runner, "fetch_live", live_fetch)
    if failure == "cache":
        monkeypatch.setattr(cache, "SQLiteCache", broken_cache)

    result = news_runner.run_news(paths, ON, provider_kind="live", settings=settings)

    assert result.source_mode == "live"
    assert result.clusters == result.opportunities == []
    assert len(calls) == 1
    assert any("不回退" in note for note in result.notes)
    if failure == "cache":
        assert calls == [None]
        assert any(err["source"] == "cache" for err in result.source_errors)
    if failure == "provider":
        assert any(err["source"] == "news" for err in result.source_errors)


@pytest.mark.parametrize("fail", [False, True])
def test_live_macro_failure_does_not_load_demo(tmp_kb, examples_dir, monkeypatch, fail):
    from zhiku.macro import sources
    from zhiku.models import MissingData

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    _forbid_fixture_reads(monkeypatch, paths.fixtures)
    _clock_at(monkeypatch, macro_runner, ON)

    def live_empty(*args, **kwargs):
        if fail:
            raise RuntimeError("source offline")
        return [], []

    def missing_rate(*args, **kwargs):
        if fail:
            raise RuntimeError("rates offline")
        return MissingData("rates unavailable")

    monkeypatch.setattr(macro_runner, "live_calendar", live_empty)
    monkeypatch.setattr(sources, "live_anchors", live_empty)
    monkeypatch.setattr(macro_runner, "live_fedwatch_approx", missing_rate)

    result = macro_runner.run_macro(paths, ON, provider_kind="live")

    assert result.source_mode == "live"
    assert result.calendar_7d == result.anchors == result.scripts == []
    assert result.rate_expectation["source"] == "缺数据"
    assert result.geo["temperature"] == "缺数据"
    assert any("不回退" in note for note in result.notes)


def test_macro_history_excludes_future_or_undated_rows(tmp_kb, examples_dir, monkeypatch):
    from zhiku.macro.models import HistorySample

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    known = HistorySample(event_type="FOMC", scenario="hawkish", released=date(2026, 1, 1))
    future = known.model_copy(update={"released": date(2027, 1, 1)})
    undated = known.model_copy(update={"released": None})
    monkeypatch.setattr(macro_runner, "load_history", lambda path: [known, future, undated])

    result = macro_runner.run_macro(paths, ON, provider_kind="demo")

    fomc = next(script for script in result.scripts if script["event_type"] == "FOMC")
    assert next(pb for pb in fomc["playbooks"] if pb["scenario"] == "hawkish")["n"] == 1
    assert any("发布日期" in note for note in result.notes)


def test_historical_live_news_never_labels_current_feed_as_history(tmp_path, monkeypatch):
    from zhiku.events import layer

    paths = Paths.load(kb=tmp_path / "kb")
    _clock_at(monkeypatch, news_runner, ON + timedelta(days=1))
    monkeypatch.setattr(news_runner, "fetch_live", _unexpected)
    monkeypatch.setattr(news_runner, "load_fixture_news", _unexpected)
    monkeypatch.setattr(layer, "apply_event_layer", _unexpected)

    result = news_runner.run_news(paths, ON, provider_kind="live")

    assert result.source_mode == "live"
    assert result.clusters == []
    assert "point-in-time" in result.source_errors[0]["reason"]
    assert not paths.journal.exists()


def test_live_news_drops_future_publication_dates(tmp_kb, examples_dir, monkeypatch):
    from zhiku.news.models import NewsItem

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    settings = PipelineSettings()
    settings.macro.enabled = False
    settings.events.enabled = False
    _clock_at(monkeypatch, news_runner, ON)
    _forbid_fixture_reads(monkeypatch, paths.fixtures)
    known = NewsItem(
        id="known", source="test", headline="Known live item",
        published=datetime(2026, 10, 20, 10, tzinfo=SHANGHAI),
    )
    future = known.model_copy(update={
        "id": "future", "headline": "Future item",
        "published": datetime(2026, 10, 21, 10, tzinfo=SHANGHAI),
    })
    monkeypatch.setattr(news_runner, "fetch_live", lambda *args, **kwargs: ([known, future], []))

    result = news_runner.run_news(paths, ON, provider_kind="live", settings=settings)

    assert [row["headline"] for row in result.clusters] == ["Known live item"]
    assert any("之后发布" in note for note in result.notes)


def test_historical_macro_skips_current_snapshots_and_future_actuals(tmp_kb, monkeypatch):
    from zhiku.macro import sources
    from zhiku.macro.models import MacroEvent

    paths = Paths.load(kb=tmp_kb)
    _clock_at(monkeypatch, macro_runner, ON + timedelta(days=1))
    future = MacroEvent(
        id="future", event_type="US_CPI", importance=5,
        scheduled_at=datetime(2026, 10, 21, 20, tzinfo=SHANGHAI),
        consensus=0.1, actual=0.9, sigma=0.1,
    )
    monkeypatch.setattr(macro_runner, "live_calendar", lambda *args, **kwargs: ([future], []))
    monkeypatch.setattr(macro_runner, "live_fedwatch_approx", _unexpected)
    monkeypatch.setattr(sources, "live_anchors", _unexpected)

    result = macro_runner.run_macro(paths, ON, provider_kind="live")

    assert result.source_mode == "live"
    assert result.anchors == []
    assert result.rate_expectation["source"] == "缺数据"
    assert result.calendar_7d[0]["actual"] is None
    assert result.calendar_7d[0]["surprise_z"] is None
    assert result.calendar_7d[0]["consensus"] == 0.1
    assert any("当前快照" in note for note in result.notes)


@pytest.fixture
def ambient_live_calendar(monkeypatch):
    """An opted-in process with a counted calendar endpoint, never a real one."""
    import sys
    from types import SimpleNamespace

    import pandas as pd

    from zhiku import calendar

    calls = []

    def trade_dates():
        calls.append("calendar")
        return pd.DataFrame({"trade_date": [ON]})

    monkeypatch.setenv("ZHIKU_LIVE_CALENDAR", "1")
    monkeypatch.setattr(calendar, "_CN_CALENDAR_CACHE", {})
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(tool_trade_date_hist_sina=trade_dates))
    return calls


@pytest.mark.parametrize("provider", ["fixture", "demo", "disabled"])
def test_collectors_block_ambient_live_calendar(
    tmp_kb, examples_dir, provider, ambient_live_calendar,
):
    import os

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    news = news_runner.run_news(paths, ON, provider_kind=provider)
    macro = macro_runner.run_macro(paths, ON, provider_kind=provider)

    assert ambient_live_calendar == []
    assert os.environ["ZHIKU_LIVE_CALENDAR"] == "1"
    if provider != "disabled":
        assert news.clusters and macro.calendar_7d


def test_calendar_scopes_isolate_live_and_offline_cache(ambient_live_calendar):
    from zhiku.calendar import calendar_network, load_cn_trade_dates

    with calendar_network(enabled=False):
        offline = load_cn_trade_dates(ON.year)
        assert ON + timedelta(days=1) in offline
    assert ambient_live_calendar == []

    with calendar_network(enabled=True):
        live = load_cn_trade_dates(ON.year)
    assert live == {ON}
    assert ambient_live_calendar == ["calendar"]

    with calendar_network(enabled=False):
        assert load_cn_trade_dates(ON.year) == offline
    assert ambient_live_calendar == ["calendar"]


def test_calendar_scope_restores_after_error_and_is_context_local(ambient_live_calendar):
    from contextvars import copy_context

    from zhiku.calendar import _want_live_calendar, calendar_network, load_cn_trade_dates

    unscoped = copy_context()
    with pytest.raises(RuntimeError, match="test exit"):
        with calendar_network(enabled=False):
            assert not _want_live_calendar()
            assert unscoped.run(_want_live_calendar)
            with calendar_network(enabled=True):
                assert not _want_live_calendar()
                load_cn_trade_dates(ON.year)
            raise RuntimeError("test exit")

    assert _want_live_calendar()
    assert ambient_live_calendar == []
    assert load_cn_trade_dates(ON.year) == {ON}
    assert ambient_live_calendar == ["calendar"]


def test_live_news_preserves_calendar_opt_in(
    tmp_kb, examples_dir, monkeypatch, ambient_live_calendar,
):
    from zhiku.calendar import is_trading_day
    from zhiku.models import Market

    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    settings = PipelineSettings()
    settings.events.enabled = False
    settings.macro.enabled = False
    _clock_at(monkeypatch, news_runner, ON)

    def live_news(*args, **kwargs):
        assert is_trading_day(Market.CN, ON)
        return [], []

    monkeypatch.setattr(news_runner, "fetch_live", live_news)
    result = news_runner.run_news(paths, ON, provider_kind="live", settings=settings)

    assert result.source_mode == "live"
    assert ambient_live_calendar == ["calendar"]
