"""宏观锚点：预期差缺数据不编造、事件窗口、剧本三情景、分析字段。"""

from __future__ import annotations

from datetime import date, datetime

from zhiku.analyze import ANALYSIS_FIELDS, DEGRADED_MARK
from zhiku.calendar import SHANGHAI
from zhiku.config import Paths
from zhiku.macro.models import MacroEvent
from zhiku.macro.runner import run_macro
from zhiku.macro.surprise import attach_surprise, surprise_z
from zhiku.macro.windows import build_windows
from zhiku.pipeline.settings import load_pipeline_settings


def test_surprise_missing_consensus_not_fabricated() -> None:
    z, missing = surprise_z(0.2, None, 0.1)
    assert z is None
    assert missing and "缺数据" in missing
    ev = MacroEvent(
        id="x",
        event_type="CN_CPI",
        previous=0.1,
        consensus=None,
        actual=0.2,
        sigma=0.15,
    )
    attach_surprise([ev])
    assert ev.surprise_z is None
    assert ev.surprise_missing
    assert any("不计算预期差" in n or "无市场一致预期" in (ev.surprise_missing or "") for n in ev.notes + [ev.surprise_missing])


def test_run_macro_scripts_windows_analysis(tmp_kb, examples_dir, monkeypatch) -> None:
    monkeypatch.delenv("ZHIKU_LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ZHIKU_ANALYSIS_LLM", "0")
    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    bundle = run_macro(paths, date(2026, 10, 20), provider_kind="fixture")
    assert bundle.calendar_7d
    assert any(e["name"] for e in bundle.calendar_7d)
    from zhiku.macro.sources import load_calendar

    events = load_calendar(paths.fixtures / "macro_calendar.yaml")
    attach_surprise(events)
    no_cons = next(e for e in events if e.id == "CN_CPI_NOCONSENSUS")
    assert no_cons.surprise_z is None
    assert no_cons.surprise_missing and "缺数据" in no_cons.surprise_missing
    released = bundle.released
    assert released
    retail = next(e for e in released if e["event_type"] == "US_RETAIL")
    assert retail["surprise_z"] is not None
    assert retail["scenario"] in {"hawkish", "inline", "dovish"}
    assert bundle.scripts
    fomc = next(s for s in bundle.scripts if s["event_type"] == "FOMC")
    scenarios = {p["scenario"] for p in fomc["playbooks"]}
    assert scenarios == {"hawkish", "inline", "dovish"}
    assert bundle.anchors
    copper = next(a for a in bundle.anchors if a["asset"] == "COPPER")
    assert copper["missing"]
    assert bundle.geo["temperature"] in {"低", "中", "高", "缺数据"}
    assert bundle.chains_fired
    assert bundle.analyses
    for card in bundle.analyses:
        for field in ANALYSIS_FIELDS:
            assert card.get(field) not in (None, ""), field
        kinds = {c["kind"] for c in card.get("claims") or []}
        assert {"事实", "解读", "推测"} <= kinds
        assert card.get("deep") is False
        assert DEGRADED_MARK in (card.get("mark") or "")
    assert bundle.active_windows or bundle.windows
    assert any("事件窗口" in b or "降仓位" in b for b in bundle.window_banners) or bundle.windows
    assert "联邦基金" in (bundle.rate_expectation.get("source") or "") or "fixture" in (
        bundle.rate_expectation.get("source") or ""
    )


def test_event_window_covers_fomc() -> None:
    settings = load_pipeline_settings()
    ev = MacroEvent(
        id="FOMC_x",
        event_type="FOMC",
        name="FOMC利率决议",
        importance=5,
        scheduled_at=datetime(2026, 10, 22, 2, 0, tzinfo=SHANGHAI),
    )
    windows = build_windows([ev], settings.macro)
    assert windows
    w = windows[0]
    assert w.start < date(2026, 10, 22) < w.end or w.covers(date(2026, 10, 22))
    assert "降仓位" in w.hint or "事件窗口" in w.hint
