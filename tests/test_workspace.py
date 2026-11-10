"""Read-only HTTP boundary and CLI integration, without network or fixture fallback."""
from __future__ import annotations

import builtins
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from click import unstyle
from typer.testing import CliRunner

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from zhiku.app import workspace
from zhiku.cli import app as cli
from zhiku.config import Paths


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths.load(tmp_path / "kb")


@pytest.fixture
def report() -> dict:
    return {
        "asof": "2026-10-22", "source_mode": "demo", "fictional": True,
        "snapshot_id": "a" * 64, "candidates": [],
        "observations": [{"symbol": "EXAMPLE", "name": "示例", "formula_score": 68.0,
            "qlib_score": None, "screens": {"greenblatt": 68.0}, "source": "formula",
            "eligible": False, "status": "research_only"}],
        "gate_artifact": {"status": "unavailable", "reason": "not_published", "approved_screens": []},
        "universe_size": 3, "quality_passed": ["EXAMPLE"], "missing_ratio": 0.1,
        "qlib": {"status": "未训练", "available": False, "reason": "not_published"},
        "news": {"asof": "2026-10-22", "source_mode": "demo", "analysis_engine": "rules",
            "clusters": [{"headline": "虚构新闻", "symbol": "EXAMPLE", "sources": [{"source": "example", "url": "https://example.com/story?token=secret#key"}]}]},
        "macro": {"asof": "2026-10-22", "source_mode": "demo", "anchors": [{"asset": "VIX", "label": "VIX", "price": 18.0, "chg": 0.02}]},
    }


@pytest.fixture
def client(paths: Paths, monkeypatch) -> TestClient:
    monkeypatch.setattr(workspace, "now_shanghai", lambda: datetime(2026, 10, 22, 21, tzinfo=ZoneInfo("Asia/Shanghai")))
    return TestClient(workspace.create_app(paths), base_url="http://localhost")


def write_report(paths: Paths, report: dict, snapshot: str = "latest") -> Path:
    path = paths.reports / "latest-daily.json" if snapshot == "latest" else paths.reports / snapshot[:7] / f"{snapshot}-zk-daily.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_missing_is_useful_unknown_and_does_not_create_kb(client, paths):
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "missing"
    assert data["service"]["mode"] == "unknown"
    assert data["news"]["emergency"]["active"] is None
    assert not data["candidates"] and not data["research_observations"]
    assert client.get("/api/export").status_code == 409
    assert not paths.kb.exists()


def test_health_means_process_only(client):
    assert client.get("/healthz").json() == {"status": "ok", "service": "zhiku-workspace", "read_only": True, "scope": "process_only", "market_data_checked": False}


@pytest.mark.parametrize("raw", ["{", "[]", '{"asof":"bad"}', '{"asof":"2026-10-22", "asof":"2026-10-21"}', '{"x":NaN}'])
def test_corrupt_report_is_not_500_or_green(client, paths, raw):
    paths.reports.mkdir(parents=True)
    (paths.reports / "latest-daily.json").write_text(raw)
    response = client.get("/api/workspace")
    assert response.status_code == 200
    assert response.json()["report"]["state"] == "invalid"
    assert response.json()["service"]["mode"] == "unknown"
    assert str(paths.kb) not in response.text


def test_demo_provenance_and_evidence_projection(client, paths, report):
    report["screens"] = {"greenblatt": [{"symbol": "EXAMPLE", "score": 68.0, "breakdown": {"roc": 0.3, "holding_quantity": 5000}}]}
    report["audits"] = [{"symbol": "EXAMPLE", "conclusion": "观察", "opinions": [{"master": "graham", "stance": "观察", "rationale": "等待证据"}]}]
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "ready"
    assert data["service"]["demo"] is True
    assert data["report"]["freshness"] == "current"
    assert data["summary"]["formal_candidates"] == 0
    assert data["summary"]["research_observations"] == 1
    assert data["research_observations"][0]["status"] == "research_only"
    assert data["screen_evidence"]["greenblatt"][0]["breakdown"] == {"roc": 0.3}
    assert data["audits"][0]["opinions"][0]["rationale"] == "等待证据"
    assert data["news"]["clusters"][0]["sources"][0]["url"] == "https://example.com/story"
    assert data["sources"][0]["status"] == "demo"
    assert data["model"]["available"] is False


def test_export_and_api_omit_private_arbitrary_fields(client, paths, report):
    report.update(review_pack="/home/user/private/review.json", portfolio={"secret": "not-public"}, positions=[{"symbol": "PRIVATE", "quantity": 77}])
    report["news"]["emergency"] = {"active": True, "freeze_adds": True, "reasons": ["样例风险"], "affected": ["PRIVATE"]}
    report["notes"] = ["日志 /tmp/private/file.json", "api_key=secret-value"]
    write_report(paths, report)
    for endpoint in ("/api/workspace", "/api/export"):
        response = client.get(endpoint)
        assert response.status_code == 200
        for forbidden in ("review_pack", "/home/user", "not-public", "PRIVATE", "quantity", "secret-value", "/tmp/private"):
            assert forbidden not in response.text
    assert 'attachment; filename="zhiku-demo-2026-10-22-workspace.json"' == response.headers["content-disposition"]


@pytest.mark.parametrize("change", ["legacy", "mixed", "fictional", "nested_demo", "unapproved", "nan"])
def test_invalid_mode_and_formal_boundaries_fail_closed(client, paths, report, change):
    if change == "legacy":
        report.pop("observations")
    elif change == "mixed":
        report["candidates"] = report["observations"]
    elif change == "fictional":
        report["fictional"] = False
    elif change == "nested_demo":
        report.update(source_mode="live", fictional=False)
    elif change == "unapproved":
        report["candidates"] = [{"symbol": "EXAMPLE", "total": 90.0, "formula_score": 90.0, "screens": {"greenblatt": 90.0}}]
    else:
        report["observations"][0]["formula_score"] = float("nan")
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "invalid"
    assert data["candidates"] == []
    assert client.get("/api/export").status_code == 409


def test_source_marker_mismatch_blocks_display(client, paths, report):
    write_report(paths, report)
    (paths.kb / ".zhiku-source-mode").write_text("live\n")
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "blocked"
    assert data["service"]["mode"] == "unknown"
    assert not data["research_observations"]


def test_valid_formal_snapshot_retains_historical_label(client, paths, report):
    report["gate_artifact"] = {"status": "approved", "reason": "validated", "approved_screens": ["greenblatt"], "statuses": {"greenblatt": "passed"}, "version": "b" * 64, "evaluated_asof": "2026-10-20"}
    report["candidates"] = [{"symbol": "EXAMPLE", "formula_score": 68.0, "total": 61.2, "screens": {"greenblatt": 68.0}, "source": "formula"}]
    report["asof"] = report["news"]["asof"] = report["macro"]["asof"] = "2026-10-21"
    write_report(paths, report, "2026-10-21")
    response = client.get("/api/workspace?snapshot=2026-10-21")
    data = response.json()
    assert data["report"]["state"] == "ready"
    assert data["report"]["historical"] is True
    assert data["report"]["freshness"] == "stale"
    assert data["report"]["age_days"] == 1
    assert len(data["candidates"]) == 1
    assert data["selected_snapshot"] == "2026-10-21"
    assert client.get("/api/export?snapshot=2026-10-21").json()["selected_snapshot"] == "2026-10-21"
    assert client.get("/api/workspace").json()["report"]["state"] == "missing"  # no silent latest fallback


def test_snapshot_registry_rejects_paths_and_mismatched_dates(client, paths, report):
    write_report(paths, report, "2026-10-21")
    assert client.get("/api/workspace?snapshot=2026-10-21").json()["report"]["state"] == "invalid"
    for key in ("../secret", "other.json", "2026-10-20"):
        assert client.get("/api/workspace", params={"snapshot": key}).status_code == 404
    assert client.get("/api/workspace", params={"snapshot": "/etc/passwd"}).status_code in {404, 422}


@pytest.mark.parametrize("component", ["file", "reports", "marker", "month"])
def test_symlink_components_are_never_followed(client, paths, report, tmp_path, component):
    secret = tmp_path / "outside"
    secret.mkdir()
    (secret / "daily.json").write_text(json.dumps(report))
    paths.kb.mkdir(parents=True)
    snapshot = "latest"
    if component == "reports":
        paths.reports.symlink_to(secret, target_is_directory=True)
    elif component == "file":
        paths.reports.mkdir()
        (paths.reports / "latest-daily.json").symlink_to(secret / "daily.json")
    elif component == "marker":
        write_report(paths, report)
        (paths.kb / ".zhiku-source-mode").symlink_to(secret / "daily.json")
    else:
        paths.reports.mkdir()
        (paths.reports / "2026-10").symlink_to(secret, target_is_directory=True)
        snapshot = "2026-10-22"
    response = client.get("/api/workspace", params={"snapshot": snapshot})
    if component == "month":
        assert response.status_code == 404
    else:
        assert response.json()["report"]["state"] == "blocked"


def test_oversized_and_nonregular_reports_block(client, paths, monkeypatch):
    paths.reports.mkdir(parents=True)
    file = paths.reports / "latest-daily.json"
    file.write_text("x" * 100)
    # Use the reader directly because its bound limit is deliberate.
    with pytest.raises(workspace.ReportUnavailable, match="blocked"):
        workspace._read_beneath(paths.kb, ("reports", "latest-daily.json"), max_bytes=10)
    file.unlink()
    file.mkdir()
    assert client.get("/api/workspace").json()["report"]["state"] == "blocked"


def test_static_security_and_read_only_methods(client, paths, report, monkeypatch):
    write_report(paths, report)
    before = {p.relative_to(paths.kb): p.read_bytes() for p in paths.kb.rglob("*") if p.is_file()}
    for _ in range(2):
        assert client.get("/api/workspace").status_code == 200
        assert client.get("/api/export").status_code == 200
    assert client.get("/").status_code == 200
    asset = client.get("/static/favicon.svg")
    assert asset.status_code == 200
    assert "script-src 'self'" in asset.headers["content-security-policy"]
    assert asset.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in asset.headers
    assert client.get("/api/workspace", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/workspace", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/api/workspace", json={"run": True}).status_code == 405
    assert client.get("/static/../../reports/latest-daily.json").status_code == 404
    assert client.get("/docs").status_code == 404
    after = {p.relative_to(paths.kb): p.read_bytes() for p in paths.kb.rglob("*") if p.is_file()}
    assert before == after


def test_cli_web_uses_loopback_and_does_not_claim_mode(paths, report, monkeypatch):
    import uvicorn
    called = {}
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: called.update(kwargs))
    result = CliRunner().invoke(cli, ["--kb", str(paths.kb), "ui", "--port", "8765"])
    assert result.exit_code == 0, result.output
    assert called == {"host": "127.0.0.1", "port": 8765, "proxy_headers": False}
    assert not paths.kb.exists()
    write_report(paths, report)
    (paths.kb / ".zhiku-source-mode").write_text("demo\n")
    assert CliRunner().invoke(cli, ["--kb", str(paths.kb), "ui"]).exit_code == 0
    assert (paths.kb / ".zhiku-source-mode").read_text() == "demo\n"


def test_cli_legacy_preserves_paths(paths, monkeypatch):
    import subprocess
    import sys
    import types
    monkeypatch.setitem(sys.modules, "streamlit", types.ModuleType("streamlit"))
    called = {}
    def call(cmd, env):
        called.update(cmd=cmd, env=env)
        return 0
    monkeypatch.setattr(subprocess, "call", call)
    result = CliRunner().invoke(cli, ["--kb", str(paths.kb), "ui", "--legacy"])
    assert result.exit_code == 0, result.output
    assert "streamlit" in called["cmd"]
    assert "--server.address" in called["cmd"]
    assert called["env"]["ZHIKU_KB"] == str(paths.kb)


def test_cli_missing_web_dependency_help(paths, monkeypatch):
    original_import = builtins.__import__
    def without_uvicorn(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("absent")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_uvicorn)
    result = CliRunner().invoke(cli, ["--kb", str(paths.kb), "ui"])
    assert result.exit_code == 1
    assert '.[ui]' in result.output
    assert "Traceback" not in result.output


def test_live_and_disabled_sources_never_claim_healthy(client, paths, report):
    report.update(source_mode="live", fictional=False)
    report["news"]["source_mode"] = "live"
    report["news"]["source_errors"] = ["feed timeout"]
    report["macro"].update(source_mode="disabled", anchors=[])
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["service"]["mode"] == "live" and data["service"]["demo"] is False
    assert [source["status"] for source in data["sources"]] == ["snapshot", "partial", "disabled"]
    assert data["news"]["emergency"]["active"] is None


def test_future_snapshot_freshness_is_explicit(client, paths, report):
    report["asof"] = report["news"]["asof"] = report["macro"]["asof"] = "2026-10-23"
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["freshness"] == "future"
    assert data["report"]["age_days"] == -1
    assert data["report"]["issues"]


def test_get_does_not_start_pipeline_network_or_fixture(client, paths, report, monkeypatch):
    import socket
    from zhiku.data.fixture import FixtureProvider
    from zhiku.pipeline import daily
    def denied(*args, **kwargs):
        raise AssertionError("A read-only GET must not invoke this operation")
    monkeypatch.setattr(daily, "run_daily", denied)
    monkeypatch.setattr(FixtureProvider, "__init__", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    write_report(paths, report)
    assert client.get("/api/workspace").json()["report"]["state"] == "ready"
    assert client.get("/api/export").status_code == 200


def test_allowlist_rejects_unsafe_urls_and_redacts_prefixed_keys(client, paths, report):
    report["notes"] = ["OPENAI_API_KEY=secret-value", "/private/new/path.json", "key sk-1234567890abcdef"]
    report["news"]["clusters"][0]["sources"][0]["url"] = "javascript:alert(1)"
    write_report(paths, report)
    response = client.get("/api/workspace")
    assert response.json()["news"]["clusters"][0]["sources"][0]["url"] == ""
    for secret in ("secret-value", "/private", "sk-1234567890"):
        assert secret not in response.text


@pytest.mark.parametrize("force_color", [False, True], ids=["default", "forced-color"])
def test_cli_rejects_wildcard_allowed_host(paths, force_color):
    result = CliRunner().invoke(
        cli, ["--kb", str(paths.kb), "ui", "--allow-host", "*"],
        color=force_color,
        env={"FORCE_COLOR": "1", "TERM": "xterm-256color"} if force_color else {},
    )
    assert result.exit_code == 2
    if force_color:
        assert "\x1b[" in result.output  # Exercise Rich's split ANSI-styled option.
    assert "--allow-host" in unstyle(result.output)


@pytest.mark.parametrize("broken", ["status", "reason", "version", "evaluated_asof", "statuses"])
def test_empty_formal_pool_cannot_bypass_gate_provenance(client, paths, report, broken):
    gate = {"status": "approved", "reason": "validated", "approved_screens": ["greenblatt"],
            "version": "b" * 64, "evaluated_asof": "2026-10-21", "statuses": {"greenblatt": "passed"}}
    gate[broken] = {"status": "unavailable", "reason": "expired", "version": None,
                    "evaluated_asof": None, "statuses": {"greenblatt": "failed"}}[broken]
    report["gate_artifact"] = gate
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "invalid"
    assert data["gates"]["approved_screens"] == []
    assert data["gates"]["status"] == "unavailable"


def test_nested_source_without_mode_does_not_inherit_demo(client, paths, report):
    report["news"].pop("source_mode")
    report["macro"].pop("source_mode")
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "ready"
    assert data["service"]["demo"] is True
    assert data["news"]["source_mode"] is None
    assert data["macro"]["source_mode"] is None
    assert [source["status"] for source in data["sources"]] == ["demo", "unknown", "unknown"]


def test_future_formal_snapshot_keeps_explicit_context(client, paths, report):
    report["asof"] = report["news"]["asof"] = report["macro"]["asof"] = "2026-10-23"
    report["gate_artifact"] = {"status": "approved", "reason": "validated", "approved_screens": ["greenblatt"],
        "version": "b" * 64, "evaluated_asof": "2026-10-21", "statuses": {"greenblatt": "passed"}}
    report["candidates"] = [{"symbol": "EXAMPLE", "formula_score": 68.0, "total": 61.2,
                              "screens": {"greenblatt": 68.0}, "source": "formula"}]
    write_report(paths, report, "2026-10-23")
    data = client.get("/api/workspace?snapshot=2026-10-23").json()
    assert data["report"]["state"] == "ready"
    assert data["report"]["historical"] is True
    assert data["report"]["freshness"] == "future"
    assert data["report"]["issues"]
    assert len(data["candidates"]) == 1


def test_actual_news_bundle_errors_remain_partial_and_redacted(client, paths, report):
    from zhiku.news.runner import NewsBundle
    report["news"] = json.loads(NewsBundle(asof=report["asof"], kind="all", source_mode="demo",
        source_errors=[{"source": "rss", "reason": "feed timeout OPENAI_API_KEY=secret-value /private/feed/config.json",
                        "authorization": "SECRET-EXTRA-FIELD"}]).to_json())
    write_report(paths, report)
    for endpoint in ("/api/workspace", "/api/export"):
        response = client.get(endpoint)
        data = response.json()
        assert response.status_code == 200
        assert data["report"]["state"] == "ready"
        assert data["sources"][1]["status"] == "partial"
        assert data["news"]["source_errors"][0].startswith("rss: feed timeout")
        for secret in ("secret-value", "/private/feed", "SECRET-EXTRA-FIELD", "authorization"):
            assert secret not in response.text


def test_actual_optional_bundle_defaults_remain_unknown(client, paths, report):
    from zhiku.news.runner import NewsBundle
    from zhiku.macro.runner import MacroBundle
    report["news"] = json.loads(NewsBundle(asof=report["asof"], kind="all").to_json())
    report["macro"] = json.loads(MacroBundle(asof=report["asof"]).to_json())
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "ready"
    assert [source["status"] for source in data["sources"]] == ["demo", "unknown", "unknown"]


def test_published_model_missing_inference_preserves_actual_reason(client, paths, report):
    report["qlib"] = {"status": "trained", "kind": "lgb_panel", "available": True,
        "inference_available": False, "inference_reason": "no_current_features_or_scores",
        "trained_on": "2026-10-21", "publication_status": "approved",
        "validation_scope": "artifact_integrity_and_inference_only", "path": "/private/models/model.pkl"}
    write_report(paths, report)
    data = client.get("/api/workspace").json()
    assert data["report"]["state"] == "ready"
    assert data["model"]["implementation"] == "lgb_panel"
    assert data["model"]["reason"] == "no_current_features_or_scores"
    assert data["model"]["inference_available"] is False
    assert data["model"]["validation_scope"] == "artifact_integrity_and_inference_only"
    assert "path" not in data["model"]


def test_embedded_error_urls_never_expose_credentials_or_signatures(client, paths, report):
    reasons = [
        "源 rss 失败：401 https://alice:my-password@example.com/feed",
        "signed source https://example.com/feed?X-Amz-Credential=private-credential&X-Amz-Signature=private-signature#private-fragment",
    ]
    report["notes"] = reasons
    report["news"]["source_errors"] = [{"source": "rss", "reason": reason} for reason in reasons]
    write_report(paths, report)
    for endpoint in ("/api/workspace", "/api/export"):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json()["report"]["state"] == "ready"
        assert response.json()["sources"][1]["status"] == "partial"
        for secret in ("alice", "my-password", "X-Amz-", "private-credential", "private-signature", "private-fragment"):
            assert secret not in response.text
        assert "https://example.com/feed" in response.text
