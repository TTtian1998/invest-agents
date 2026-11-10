"""Research/publication/daily boundaries, without network or model extras."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from zhiku.cli import app
from zhiku.config import Paths
from zhiku.pipeline.artifacts import load_approved_gates, publish_gate_research, save_gate_research, validated_gate_status
from zhiku.pipeline.daily import run_daily, write_daily_reports
from zhiku.pipeline.settings import PipelineSettings, load_pipeline_settings

ON = date(2026, 10, 20)


def _evidence(**overrides):
    result = {
        "screen": "greenblatt", "engine": "pandas-pit", "fill": "next_open", "close_fallback_fills": 0,
        "annualized": .20, "benchmark_annualized": .10, "sharpe": 1.2,
        "excess_annualized": .10, "excess_first_half": .03, "excess_second_half": .04,
        "coverage_ratio": 1., "n_months": 12, "n_covered": 12,
        "benchmarks": {"universe_ew": {"sharpe": .5, "engine": "pandas-pit", "fill": "next_open", "close_fallback_fills": 0}},
        "equity": [["2026-07-31", 1.0], ["2026-08-31", 1.02], ["2026-09-30", 1.05], ["2026-10-19", 1.08]],
    }
    return {**result, **overrides}


def _publish(paths, settings, evidence=None, provider="demo", on=ON):
    if evidence is None:
        evidence = _evidence()
        evidence["equity"] = [[(date.fromisoformat(day) + (on - ON)).isoformat(), value] for day, value in evidence["equity"]]
    artifact = save_gate_research(paths, on, provider, settings,
        backtests={"greenblatt": _evidence() if evidence is None else evidence},
        provenance={"method": "pandas-pit", "data_asof": on.isoformat(), "input_hash": "a" * 64}, notes=[])
    return artifact, publish_gate_research(paths, artifact, on=on, provider_kind=provider, settings=settings)


def test_formula_gate_requires_complete_finite_evidence():
    settings = PipelineSettings()
    assert validated_gate_status({"passed_gate": True}, settings) == "insufficient"
    for key in ["sharpe", "excess_annualized", "excess_first_half", "coverage_ratio"]:
        assert validated_gate_status(_evidence(**{key: None}), settings) == "insufficient"
        assert validated_gate_status(_evidence(**{key: float("nan")}), settings) == "insufficient"
    assert validated_gate_status(_evidence(), settings) == "passed"
    assert validated_gate_status(_evidence(excess_second_half=-.01, passed_gate=True), settings) == "failed"
    assert validated_gate_status(_evidence(n_covered=2), settings) == "insufficient"


def test_gate_publication_and_context_boundaries(tmp_kb, examples_dir):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = PipelineSettings()
    _publish(paths, settings)
    _, info = load_approved_gates(paths, on=ON, provider_kind="fixture", settings=settings)
    assert info["approved_screens"] == ["greenblatt"]
    for day, provider, cfg, reason in [
        (date(2026, 10, 19), "demo", settings, "future_artifact"),
        (date(2026, 12, 20), "demo", settings, "expired"),
        (ON, "live", settings, "provider_mismatch"),
        (ON, "demo", settings.model_copy(update={"merge": settings.merge.model_copy(update={"formula": .5})}), "config_mismatch"),
    ]:
        evidence, refused = load_approved_gates(paths, on=day, provider_kind=provider, settings=cfg)
        assert evidence == {} and refused["approved_screens"] == []
        assert refused["reason"] == reason


def test_invalid_gate_does_not_clobber_and_hash_checked(tmp_kb, examples_dir):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = PipelineSettings()
    _, approved = _publish(paths, settings)
    with pytest.raises(ValueError, match="No complete"):
        _publish(paths, settings, {"passed_gate": True})
    _, loaded = load_approved_gates(paths, on=ON, provider_kind="demo", settings=settings)
    assert loaded["version"] == approved["version"]
    path = paths.models_dir / "approved-gates" / f"{approved['version']}.json"
    raw = json.loads(path.read_text())
    raw["backtests"]["greenblatt"]["sharpe"] = 99
    path.write_text(json.dumps(raw))
    _, loaded = load_approved_gates(paths, on=ON, provider_kind="demo", settings=settings)
    assert loaded["reason"] == "invalid_artifact"


def test_negative_gate_can_explicitly_close_pool(tmp_kb, examples_dir):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = PipelineSettings()
    _publish(paths, settings)
    _publish(paths, settings, _evidence(excess_second_half=-.01))
    _, info = load_approved_gates(paths, on=ON, provider_kind="demo", settings=settings)
    assert info["status"] == "approved" and info["approved_screens"] == []
    assert info["statuses"]["greenblatt"] == "failed"


def test_daily_missing_gates_keeps_only_observations(tmp_kb, examples_dir, monkeypatch):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    def forbidden(*args, **kwargs):
        pytest.fail("Daily must not call historical evaluation, training, or live collection")
    monkeypatch.setattr("zhiku.backtest.pit_gate.evaluate_screens", forbidden)
    monkeypatch.setattr("zhiku.factors.qlib_adapter.train_and_select", forbidden)
    monkeypatch.setattr("zhiku.news.sources.fetch_live", forbidden)
    # Even a valid-looking quant score cannot bypass missing formula approval.
    monkeypatch.setattr("zhiku.pipeline.daily.predict_published_scores", lambda *a, **k: ({"600519": 99.}, {"status": "approved"}))
    daily = run_daily(paths, ON, provider_kind="demo")
    assert daily.candidates == []
    assert daily.observations and all(x["status"] == "research_only" for x in daily.observations)
    assert daily.gate_artifact["reason"] == "not_published"
    assert daily.source_mode == "demo" and daily.fictional
    assert daily.review_pack
    assert not daily.backtests


def test_approved_daily_happy_path_and_immutable_snapshots(tmp_kb, examples_dir, monkeypatch):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = load_pipeline_settings()
    _publish(paths, settings)
    monkeypatch.setattr("zhiku.factors.qlib_adapter.train_and_select", lambda *a, **k: pytest.fail("no train"))
    daily = run_daily(paths, ON, provider_kind="demo")
    assert daily.candidates
    assert all(set(c["screens"]) == {"greenblatt"} for c in daily.candidates)
    assert all(c["source"] == "formula" for c in daily.candidates)
    write_daily_reports(paths, ON, daily)
    original = list((paths.reports / "daily-snapshots").glob("*.json"))
    assert len(original) == 1
    before = original[0].read_text()
    daily.notes.append("new note")
    write_daily_reports(paths, ON, daily)
    assert original[0].read_text() == before
    assert len(list((paths.reports / "daily-snapshots").glob("*.json"))) == 2


def test_run_daily_demo_never_syncs_or_trains(tmp_kb, examples_dir, monkeypatch):
    import sys
    from types import SimpleNamespace
    def forbidden(*args, **kwargs):
        pytest.fail("offline command crossed live/research boundary")
    monkeypatch.setattr("zhiku.calendar._CN_CALENDAR_CACHE", {})
    monkeypatch.setenv("ZHIKU_LIVE_CALENDAR", "1")
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(tool_trade_date_hist_sina=forbidden))
    monkeypatch.setattr("zhiku.data.sync.sync_market", forbidden)
    monkeypatch.setattr("zhiku.factors.qlib_adapter.train_and_select", forbidden)
    result = CliRunner().invoke(app, ["--provider", "demo", "run-daily", "--date", ON.isoformat(), "--skip-train"], env={"ZHIKU_KB": str(tmp_kb), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")})
    assert result.exit_code == 0, result.output
    assert "已弃用" in result.output
    raw = json.loads((tmp_kb / "reports" / "latest-daily.json").read_text())
    assert raw["source_mode"] == "demo" and raw["candidates"] == []
    assert any("耗时 total" in s for s in raw["notes"])


def test_demo_research_is_honestly_insufficient(tmp_kb, examples_dir):
    runner = CliRunner()
    env = {"ZHIKU_KB": str(tmp_kb), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")}
    research = runner.invoke(app, ["--provider", "demo", "research", "gates", "--date", ON.isoformat()], env=env)
    assert research.exit_code == 0, research.output
    path = research.output.strip()
    result = runner.invoke(app, ["--provider", "demo", "research", "publish-gates", path, "--date", ON.isoformat()], env=env)
    assert result.exit_code == 1 and "No complete" in result.output
    assert not (tmp_kb / "models" / "approved-gates" / "approved.json").exists()


def test_export_does_not_run_daily(tmp_kb, examples_dir, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("export-review must not rerun daily")
    monkeypatch.setattr("zhiku.pipeline.daily.run_daily", forbidden)
    result = CliRunner().invoke(app, ["--provider", "demo", "export-review", "--date", ON.isoformat()], env={"ZHIKU_KB": str(tmp_kb), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")})
    assert result.exit_code == 1
    assert "先运行 daily" in result.output


@pytest.mark.parametrize("bad", [[], None, {"n_months": 0}, {"n_months": "12"}, {"benchmarks": []}, {"equity": "invalid"}])
def test_malformed_gate_evidence_is_not_accepted(bad):
    value = {**_evidence(), **bad} if isinstance(bad, dict) else bad
    assert validated_gate_status(value, PipelineSettings()) == "insufficient"


@pytest.mark.parametrize("payload", [[], None, {"version": "../escape"}])
def test_malformed_manifest_fails_closed(tmp_kb, examples_dir, payload):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    folder = paths.models_dir / "approved-gates"
    folder.mkdir(parents=True)
    (folder / "approved.json").write_text(json.dumps(payload))
    data, info = load_approved_gates(paths, on=ON, provider_kind="demo", settings=PipelineSettings())
    assert data == {} and info["reason"] == "invalid_artifact"


def test_strategy_version_invalidates_artifacts(tmp_kb, examples_dir, monkeypatch):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = PipelineSettings()
    _publish(paths, settings)
    monkeypatch.setattr("zhiku.pipeline.artifacts.STRATEGY_VERSION", "different")
    data, info = load_approved_gates(paths, on=ON, provider_kind="demo", settings=settings)
    assert data == {} and info["reason"] == "config_mismatch"


def test_ingest_renders_same_snapshot_without_daily(tmp_kb, examples_dir, monkeypatch):
    from zhiku.ai.pack import load_review_snapshot
    from zhiku.ai.schema import CitedClaim, Judgment, ReviewResult
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    _publish(paths, load_pipeline_settings())
    daily = run_daily(paths, ON, provider_kind="demo")
    write_daily_reports(paths, ON, daily)
    pack = load_review_snapshot(paths.reports, asof=ON)
    result = ReviewResult(asof=ON.isoformat(), snapshot_id=pack["snapshot_id"], backend="external",
        judgments=[Judgment(symbol=item["candidate"]["symbol"], stance="观察", confidence=.4,
            long_case=[CitedClaim(claim="查看 ROE", cite="roe")], invalidation="ROE 转负")
            for item in pack["candidates"]])
    result_path = tmp_kb / "result.json"
    result_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False))
    monkeypatch.setattr("zhiku.pipeline.daily.run_daily", lambda *a, **k: pytest.fail("must render saved evidence"))
    env = {"ZHIKU_KB": str(tmp_kb), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")}
    runner = CliRunner()
    exported = runner.invoke(app, ["--provider", "demo", "export-review", "--date", ON.isoformat()], env=env)
    assert exported.exit_code == 0 and pack["snapshot_id"] in exported.output
    ingested = runner.invoke(app, ["--provider", "demo", "ingest-review", str(result_path)], env=env)
    assert ingested.exit_code == 0, ingested.output
    assert "已更新匹配快照日报" in ingested.output
    saved = json.loads((paths.reports / "latest-daily.json").read_text())
    assert saved["ai_layer"]["snapshot_id"] == pack["snapshot_id"]
    again = runner.invoke(app, ["--provider", "demo", "ingest-review", str(result_path)], env=env)
    assert again.exit_code == 0 and "写入 0 条" in again.output
    result.judgments[0].confidence = .8
    result_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False))
    conflict = runner.invoke(app, ["--provider", "demo", "ingest-review", str(result_path)], env=env)
    assert conflict.exit_code == 1 and "研判写入被拒绝" in conflict.output


def test_factor_train_cli_binds_producer_context(tmp_kb, examples_dir, monkeypatch):
    from zhiku.pipeline.artifacts import config_fingerprint
    calls = []
    def train(**kwargs):
        calls.append(kwargs)
        return {"kind": "lgb_lite", "path": "candidate.pkl"}
    monkeypatch.setattr("zhiku.factors.qlib_adapter.train_and_select", train)
    result = CliRunner().invoke(app, ["--provider", "demo", "factor", "train"], env={"ZHIKU_KB": str(tmp_kb), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")})
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0]["provider_kind"] == "fixture"
    assert calls[0]["config_fingerprint"] == config_fingerprint(load_pipeline_settings())
    assert "candidate.pkl" in result.output and "尚未发布" in result.output


def test_live_prepare_does_not_fall_back_to_fixtures(tmp_kb, examples_dir, monkeypatch):
    monkeypatch.setattr("zhiku.factors.qlib_adapter.prepare_from_csv", lambda *a, **k: pytest.fail("live must not read fixtures"))
    result = CliRunner().invoke(app, ["--provider", "live", "factor", "prepare"], env={"ZHIKU_KB": str(tmp_kb), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")})
    assert result.exit_code == 1
    assert "不回退 fixture" in result.output


@pytest.mark.parametrize("close,offset,formal", [(1, 0, True), (1, -1, False), (0, 0, False), (-1, 0, False), (float("inf"), 0, False), (float("nan"), 0, False), (None, 0, False), ("bad", 0, False)])
def test_live_current_inputs_are_required(tmp_kb, examples_dir, monkeypatch, close, offset, formal):
    import pandas as pd
    from zhiku.calendar import now_shanghai
    from zhiku.pipeline.universe import load_pipeline_universe
    from zhiku.macro.runner import MacroBundle
    from zhiku.news.runner import NewsBundle
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = load_pipeline_settings()
    on = now_shanghai().date() + timedelta(days=offset)
    _publish(paths, settings, provider="live", on=on)
    rows, delist, notes = load_pipeline_universe(paths, ON, provider_kind="demo", settings=settings)
    class Store:
        def __init__(self, path):
            pass
        def latest_spot(self, asof):
            return pd.DataFrame({"symbol": [row.symbol for row in rows], "asof": [on] * len(rows)})
        def bars(self, *args, **kwargs):
            return pd.DataFrame({"symbol": [row.symbol for row in rows], "date": [on] * len(rows), "close": [close] * len(rows)})
    monkeypatch.setattr("zhiku.data.store.MarketStore", Store)
    monkeypatch.setattr("zhiku.pipeline.daily.load_pipeline_universe", lambda *a, **k: (rows, delist, notes))
    monkeypatch.setattr("zhiku.macro.runner.run_macro", lambda *a, **k: MacroBundle(asof=on.isoformat()))
    monkeypatch.setattr("zhiku.news.runner.run_news", lambda *a, **k: NewsBundle(asof=on.isoformat(), kind="all"))
    monkeypatch.setenv("ZHIKU_LLM_BACKEND", "external")
    daily = run_daily(paths, on, provider_kind="live")
    assert daily.gate_artifact["approved_screens"] == ([] if offset else ["greenblatt"])
    assert bool(daily.candidates) is formal and daily.observations
    if offset:
        assert any("PIT 来源尚未验证" in note for note in daily.notes)
    elif not formal:
        assert any("有效正数价格" in note for note in daily.notes)



def test_relabelled_old_gate_evidence_cannot_be_renewed(tmp_kb, examples_dir):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = load_pipeline_settings()
    old = _evidence(equity=[["2026-01-31", 1.], ["2026-02-28", 1.02], ["2026-03-31", 1.05], ["2026-04-30", 1.08]])
    for cutoff, match in [("2026-04-30", "expired"), (ON.isoformat(), "No complete")]:
        path = save_gate_research(paths, ON, "demo", settings, backtests={"greenblatt": old},
            provenance={"method": "pandas-pit", "data_asof": cutoff, "input_hash": "a" * 64}, notes=[])
        with pytest.raises(ValueError, match=match):
            publish_gate_research(paths, path, on=ON, provider_kind="demo", settings=settings)


@pytest.mark.parametrize("fallback", [None, True, "0", 1, -1])
def test_next_open_gate_requires_proven_zero_close_substitutions(fallback):
    assert validated_gate_status(_evidence(close_fallback_fills=fallback), PipelineSettings()) == "insufficient"


def test_shared_kb_mode_switch_is_rejected(tmp_kb, examples_dir):
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    run_daily(paths, ON, provider_kind="demo")
    with pytest.raises(ValueError, match="独立 --kb"):
        run_daily(paths, ON, provider_kind="live")
    result = CliRunner().invoke(app, ["--provider", "live", "--kb", str(tmp_kb), "factor", "predict"])
    assert result.exit_code != 0 and "独立 --kb" in result.output



@pytest.mark.parametrize("fallback", [None, True, "0", 1, -1])
def test_benchmark_next_open_evidence_cannot_substitute_closes(fallback):
    evidence = _evidence()
    evidence["benchmarks"]["universe_ew"]["close_fallback_fills"] = fallback
    assert validated_gate_status(evidence, PipelineSettings()) == "insufficient"



def test_gate_actual_publication_uses_shanghai_date(tmp_kb, examples_dir, monkeypatch):
    from datetime import datetime, timezone
    from zhiku.pipeline import artifacts
    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 22, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(artifacts, "datetime", LaterDatetime)
    paths = Paths.load(tmp_kb, examples_dir / "fixtures")
    settings = load_pipeline_settings()
    _publish(paths, settings, provider="live", on=date(2026, 10, 22))
    _, info = load_approved_gates(paths, on=date(2026, 10, 22), provider_kind="live", settings=settings)
    assert info["reason"] == "future_artifact"
    _, info = load_approved_gates(paths, on=date(2026, 10, 23), provider_kind="live", settings=settings)
    assert info["status"] == "approved"


def test_demo_cli_refuses_explicit_live_sync(tmp_kb):
    result = CliRunner().invoke(app, ["--provider", "demo", "--kb", str(tmp_kb), "data", "sync"])
    assert result.exit_code != 0 and "data sync 仅支持 live" in result.output
