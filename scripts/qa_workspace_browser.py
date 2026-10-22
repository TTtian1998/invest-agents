#!/usr/bin/env python3
"""Exercise the real read-only HTTP app in Chromium using fictional reports only.

Run: python scripts/qa_workspace_browser.py --output-dir browser-artifacts
The optional --http-only preflight does NOT count as browser or visual validation.
No private KB, external market data, public listener, broker or secrets are used.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASOF = "2026-10-22"
HISTORY = "2026-10-21"
EMPTY = "2026-10-20"
INVALID = "2026-10-19"


def demo_report() -> dict[str, Any]:
    """An explicit synthetic schema fixture, never a real market recommendation."""
    return {
        "asof": ASOF, "source_mode": "demo", "fictional": True,
        "snapshot_id": "a" * 64, "universe_size": 128,
        "quality_passed": ["990001", "DEMO02.HK", "DEMO03.US", "990004"],
        "missing_ratio": 0.08,
        "candidates": [
            {"symbol": "990001", "name": "演示 · 星河科技", "formula_score": 86.4,
             "qlib_score": None, "total": 86.4, "screens": {"greenblatt": 86.4}, "source": "demo"},
            {"symbol": "DEMO02.HK", "name": "演示 · 远山消费", "formula_score": 74.2,
             "qlib_score": None, "total": 74.2, "screens": {"greenblatt": 74.2}, "source": "demo"},
        ],
        "observations": [
            {"symbol": "DEMO03.US", "name": "演示 · 海岚医疗", "formula_score": 91.5,
             "screens": {"piotroski": 91.5}, "source": "demo", "eligible": False,
             "status": "research_only"},
            {"symbol": "990004", "name": "演示 · 青禾制造", "formula_score": 63.8,
             "screens": {"piotroski": 63.8}, "source": "demo", "eligible": False,
             "status": "research_only"},
        ],
        "gate_artifact": {"status": "approved", "reason": "validated",
                          "approved_screens": ["greenblatt"], "version": "b" * 64,
                          "evaluated_asof": ASOF,
                          "statuses": {"greenblatt": "passed", "piotroski": "insufficient"}},
        "qlib": {"status": "unavailable", "reason": "not_published", "available": False,
                 "inference_available": False, "implementation": "未发布的演示模型"},
        "screens": {
            "greenblatt": [
                {"symbol": "990001", "name": "演示 · 星河科技", "score": 86.4,
                 "breakdown": {"earnings_yield": 0.08, "return_on_capital": 0.21}},
                {"symbol": "DEMO02.HK", "name": "演示 · 远山消费", "score": 74.2,
                 "breakdown": {"earnings_yield": 0.06, "return_on_capital": 0.17}},
            ],
            "piotroski": [{"symbol": "DEMO03.US", "name": "演示 · 海岚医疗", "score": 91.5,
                           "breakdown": {"f_score": 8.0}}],
        },
        "audits": [{"symbol": "990001", "name": "演示 · 星河科技",
                    "conclusion": "演示结论：仍需复核原始证据，不构成建议。",
                    "opinions": [{"master": "quality", "stance": "neutral",
                                  "rationale": "虚构研究记录，用于验证详情展示。"}]}],
        "news": {"asof": ASOF, "source_mode": "demo", "analysis_engine": "fixture",
                 "clusters": [{"fingerprint": "qa-news", "symbol": "990001",
                               "name": "演示 · 星河科技", "headline": "演示事件：季度研究资料更新",
                               "catalyst": "等待可核验的正式披露。", "tags": ["演示", "业绩"],
                               "freshness": "fresh", "source_grade": 1,
                               "source_grade_label": "虚构原始来源", "first_seen": ASOF,
                               "sources": [{"source": "演示资料", "headline": "合成测试证据",
                                            "url": "https://example.com/fictional-research",
                                            "grade": 1, "grade_label": "虚构原始来源"}]}],
                 "source_errors": [], "emergency": {"active": False, "freeze_adds": False}},
        "macro": {"asof": ASOF, "source_mode": "demo", "calendar_7d": [
            {"id": "qa-event", "event_type": "demo", "name": "演示宏观数据发布",
             "region": "示例", "importance": 3, "scheduled_at": "2026-10-23T12:30:00Z",
             "scheduled_shanghai": "2026-10-23 20:30", "previous": 2.1,
             "consensus": 2.0, "actual": None, "unit": "%"}],
            "anchors": [{"asset": "DEMO_INDEX", "label": "演示指数", "price": 100.0, "chg": 0.6},
                        {"asset": "DEMO_FX", "label": "演示汇率", "price": None,
                         "missing": "演示缺失状态"}]},
        "notes": ["所有名称、数据和验收均为自动化 QA 虚构样例，不是真实市场信号。"],
    }


def seed_workspace(kb: Path) -> None:
    from zhiku.app.workspace import DailyReport

    reports = kb / "reports"
    (reports / "2026-10").mkdir(parents=True)
    (kb / ".zhiku-source-mode").write_text("demo\n", encoding="utf-8")
    report = demo_report()
    DailyReport.model_validate(report)
    (reports / "latest-daily.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    for day in (HISTORY, EMPTY):
        historical = copy.deepcopy(report)
        historical["asof"] = day
        historical["snapshot_id"] = ("c" if day == HISTORY else "d") * 64
        historical["gate_artifact"]["evaluated_asof"] = day
        historical["news"]["asof"] = historical["macro"]["asof"] = day
        if day == EMPTY:
            historical.update(candidates=[], observations=[], screens={}, audits=[], quality_passed=[])
            historical["news"]["clusters"] = []
            historical["macro"].update(calendar_7d=[], anchors=[])
        DailyReport.model_validate(historical)
        (reports / "2026-10" / f"{day}-zk-daily.json").write_text(
            json.dumps(historical, ensure_ascii=False), encoding="utf-8")
    (reports / "2026-10" / f"{INVALID}-zk-daily.json").write_text("{invalid-json", encoding="utf-8")


def fetch(url: str, *, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    try:
        response = urlopen(Request(url, method=method), timeout=10)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, {key.lower(): value for key, value in response.headers.items()}, response.read()


@contextmanager
def serve(kb: Path, output: Path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {**os.environ, "ZHIKU_KB": str(kb), "PYTHONPATH": str(ROOT / "src")}
    base = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "zhiku.app.workspace:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(port), "--no-proxy-headers"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("HTTP server exited before readiness; inspect server.log")
                try:
                    if fetch(base + "/healthz")[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("HTTP server did not become ready within 30 seconds")
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def check_http(base: str) -> dict[str, Any]:
    status, headers, content = fetch(base + "/api/workspace")
    assert status == 200, status
    data = json.loads(content)
    assert data["report"]["state"] == "ready"
    assert data["service"]["demo"] is True
    assert data["service"]["read_only"] is True
    assert data["service"]["network_actions"] is False
    assert len(data["candidates"]) == len(data["research_observations"]) == 2
    assert all("total" not in row for row in data["research_observations"])
    assert "default-src 'self'" in headers["content-security-policy"]
    assert fetch(base + "/api/workspace", method="POST")[0] == 405
    assert fetch(base + "/api/workspace?snapshot=1999-01-01")[0] == 404
    assert json.loads(fetch(base + f"/api/workspace?snapshot={EMPTY}")[2])["candidates"] == []
    assert json.loads(fetch(base + f"/api/workspace?snapshot={INVALID}")[2])["report"]["state"] == "invalid"
    assert fetch(base + f"/api/export?snapshot={INVALID}")[0] == 409
    for resource in ("/", "/static/workspace.css", "/static/app.js", "/static/ui.js", "/static/views.js", "/static/favicon.svg"):
        assert fetch(base + resource)[0] == 200, f"Missing production asset: {resource}"
    return {"ready": True, "source_mode": "demo", "formal_count": 2, "observation_count": 2,
            "read_only_post_rejected": True, "invalid_snapshot_rejected": True}


def run_browser(base: str, kb: Path, output: Path, result: dict[str, Any]) -> None:
    """Implemented against accessible labels/production DOM, not mock page HTML."""
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        result["engine"] = {"name": "chromium", "version": browser.version}
        failures = []
        try:
            for name, width, height in (("desktop-1440", 1440, 1000), ("mobile-390", 390, 844)):
                errors: list[str] = []
                console: list[dict[str, str]] = []
                requests: list[dict[str, Any]] = []
                responses: list[dict[str, Any]] = []
                failed_requests: list[dict[str, Any]] = []
                scenario: dict[str, Any] = {"viewport": {"width": width, "height": height}, "screenshots": []}
                result["browser"][name] = scenario
                context = browser.new_context(viewport={"width": width, "height": height},
                                              device_scale_factor=1, locale="zh-CN", timezone_id="Asia/Shanghai",
                                              reduced_motion="reduce", accept_downloads=True)
                context.tracing.start(screenshots=True, snapshots=True, sources=True)
                page = context.new_page()
                page.set_default_timeout(10000)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: console.append({"type": message.type, "text": message.text}))
                page.on("request", lambda request: requests.append({"url": request.url, "method": request.method}))
                page.on("response", lambda response: responses.append({"url": response.url, "status": response.status}))
                page.on("requestfailed", lambda request: failed_requests.append({"url": request.url, "failure": request.failure}))
                directory = output / name
                directory.mkdir()

                def capture(state: str) -> None:
                    page.screenshot(path=directory / f"{state}.png", full_page=True, animations="disabled")
                    scenario["screenshots"].append(f"{name}/{state}.png")
                    overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
                    assert not overflow, f"Horizontal page overflow: {name}/{state}"

                def ready() -> None:
                    expect(page.locator("#content")).to_have_attribute("aria-busy", "false")

                def navigate(target: str) -> None:
                    if width < 700 and page.locator("#menu-toggle").get_attribute("aria-expanded") != "true":
                        page.get_by_role("button", name="展开导航").click()
                    page.locator(f'#navigation [data-page="{target}"]').click()
                    expect(page).to_have_url(base + "/#" + target)
                    expect(page.locator("#page-title")).to_have_text({"overview": "今日概览", "universe": "候选与观察",
                                                                   "evidence": "消息与证据", "validation": "研究验证"}[target])
                    ready()

                try:
                    page.goto(base, wait_until="networkidle")
                    ready()
                    expect(page.locator("#mode-badge")).to_contain_text("演示")
                    expect(page.locator("#trust-banner")).to_contain_text("演示")
                    # Numeric bars must survive the strict CSP and have an actual rendered width.
                    expect(page.locator(".chart-fill").first).to_be_visible()
                    assert page.locator(".chart-fill").first.evaluate("node => node.getBoundingClientRect().width") > 0
                    capture("01-overview")
                    navigate("universe")
                    capture("02-universe")
                    # Remaining controls are validated below after initial DOM readiness.
                    exercise_universe(page, expect, capture)
                    navigate("evidence")
                    expect(page.locator("#content")).to_contain_text("演示事件")
                    capture("06-evidence")
                    navigate("validation")
                    capture("07-validation")
                    page.go_back()
                    expect(page).to_have_url(base + "/#evidence")
                    page.go_forward()
                    expect(page).to_have_url(base + "/#validation")
                    navigate("overview")
                    for _ in range(2):
                        page.get_by_role("button", name="刷新", exact=True).click()
                        ready()
                    capture("08-refreshed")
                    with page.expect_download() as download_info:
                        page.locator("#export").click()
                    download = download_info.value
                    export_path = directory / download.suggested_filename
                    download.save_as(export_path)
                    exported = json.loads(export_path.read_text(encoding="utf-8"))
                    assert exported["service"]["demo"] is True
                    page.locator("#snapshot-select").select_option(HISTORY)
                    ready()
                    expect(page.locator("#footer-snapshot")).to_contain_text(HISTORY)
                    capture("09-history")
                    page.locator("#snapshot-select").select_option(EMPTY)
                    ready()
                    navigate("universe")
                    expect(page.locator("tbody tr")).to_have_count(0)
                    expect(page.locator("#content")).to_contain_text("没有")
                    capture("10-empty")
                    page.locator("#snapshot-select").select_option(INVALID)
                    ready()
                    expect(page.locator("#content")).to_contain_text("校验未通过")
                    expect(page.locator("#export")).to_have_attribute("aria-disabled", "true")
                    expect(page.locator("#content")).not_to_contain_text("990001")
                    capture("11-invalid-report")
                    page.locator("#snapshot-select").select_option("latest")
                    ready()
                    latest = kb / "reports" / "latest-daily.json"
                    saved = latest.read_bytes()
                    try:
                        latest.unlink()
                        page.get_by_role("button", name="刷新", exact=True).click()
                        ready()
                        expect(page.locator("#content")).to_contain_text("还没有研究快照")
                        expect(page.locator("#export")).to_have_attribute("aria-disabled", "true")
                        capture("12-missing-report")
                    finally:
                        latest.write_bytes(saved)
                    # Inject a transport failure only in the browser, leaving the real server unchanged.
                    page.route("**/api/workspace*", lambda route: route.abort("failed"))
                    page.get_by_role("button", name="刷新", exact=True).click()
                    ready()
                    expect(page.locator("#content")).to_contain_text("连接暂时不可用")
                    expect(page.locator("#content")).not_to_contain_text("DEMO03.US")
                    expect(page.locator("#export")).to_have_attribute("aria-disabled", "true")
                    capture("13-api-failure")
                    page.unroute("**/api/workspace*")
                    page.get_by_role("button", name="刷新", exact=True).click()
                    ready()
                    expect(page.locator("#mode-badge")).to_contain_text("演示")
                    expect(page.locator("tbody tr")).to_have_count(2)
                    expect(page.locator("#export")).to_have_attribute("aria-disabled", "false")
                    capture("14-recovered")
                    assert not errors, errors
                    assert all(item["method"] == "GET" for item in requests), requests
                    assert all(item["url"].startswith(base + "/") for item in requests), requests
                    assert all(item["status"] < 400 for item in responses), responses
                    unexpected_failures = [item for item in failed_requests if not (
                        ("/api/workspace?" in item["url"] and item["failure"] == "net::ERR_FAILED")
                        or ("/api/export?" in item["url"] and item["failure"] == "net::ERR_ABORTED"))]
                    assert not unexpected_failures, unexpected_failures
                    # One intentional aborted fetch is expected; unrelated JS/CSP errors are not.
                    unexpected_console = [item for item in console if item["type"] == "error"
                                          and "net::ERR_FAILED" not in item["text"]]
                    assert not unexpected_console, unexpected_console
                    scenario["status"] = "passed"
                except Exception:
                    scenario["status"] = "failed"
                    scenario["failure"] = traceback.format_exc()
                    page.screenshot(path=directory / "failure.png", full_page=True)
                    failures.append(f"{name}: {scenario['failure']}")
                finally:
                    scenario["page_errors"] = errors
                    (directory / "console.json").write_text(json.dumps(console, ensure_ascii=False, indent=2), encoding="utf-8")
                    (directory / "requests.json").write_text(json.dumps(
                        {"requests": requests, "responses": responses, "failed_requests": failed_requests},
                        ensure_ascii=False, indent=2), encoding="utf-8")
                    context.tracing.stop(path=directory / "trace.zip")
                    context.close()
            if failures:
                raise AssertionError("\n".join(failures))
        finally:
            browser.close()


def exercise_universe(page, expect, capture) -> None:
    """Production pool, search, market, sorting and dialog interactions."""
    page.get_by_role("tab", name="正式候选").click()
    expect(page.get_by_role("tab", name="正式候选")).to_have_attribute("aria-selected", "true")
    expect(page.locator("tbody tr")).to_have_count(2)
    search = page.get_by_role("searchbox", name="搜索名称或代码")
    search.fill("星河")
    expect(page.locator("tbody tr")).to_have_count(1)
    expect(page.locator("tbody")).to_contain_text("990001")
    capture("03-search")
    search.fill("NO_SUCH_DEMO_SYMBOL")
    expect(page.locator("#content")).to_contain_text("没有匹配的标的")
    expect(page.locator("#content")).not_to_contain_text("990001")
    capture("04-empty-search")
    page.get_by_role("button", name="清除筛选", exact=True).click()
    expect(search).to_have_value("")
    expect(page.locator("tbody tr")).to_have_count(2)
    page.get_by_label("筛选市场", exact=True).select_option("HK")
    expect(page.locator("tbody tr")).to_have_count(1)
    expect(page.locator("tbody")).to_contain_text("DEMO02.HK")
    capture("04b-market-filter")
    page.get_by_label("筛选市场", exact=True).select_option("all")
    page.get_by_label("排序方式", exact=True).select_option("low")
    expect(page.locator("tbody tr").first).to_contain_text("DEMO02.HK")
    page.get_by_label("排序方式", exact=True).select_option("high")
    expect(page.locator("tbody tr").first).to_contain_text("990001")
    for dismissal in ("escape", "close", "escape"):
        page.locator('[data-symbol="990001"]').first.click()
        expect(page.get_by_role("dialog")).to_be_visible()
        expect(page.get_by_role("dialog")).to_contain_text("990001")
        if dismissal == "close":
            capture("05-detail")
            page.get_by_role("button", name="关闭详情", exact=True).click()
        else:
            page.keyboard.press("Escape")
        expect(page.get_by_role("dialog")).not_to_be_visible()
    page.get_by_role("tab", name="研究观察").click()
    expect(page.get_by_role("tab", name="研究观察")).to_have_attribute("aria-selected", "true")
    expect(page.locator("tbody tr")).to_have_count(2)
    expect(page.locator("tbody")).not_to_contain_text("990001")
    page.locator('[data-symbol="DEMO03.US"]').first.click()
    expect(page.get_by_role("dialog")).to_contain_text("不证明验收资格")
    capture("05b-observation-detail")
    page.get_by_role("button", name="关闭详情", exact=True).click()
    expect(page.get_by_role("dialog")).not_to_be_visible()



def prepare_pipeline_demo(kb: Path, output: Path) -> dict[str, Any]:
    """Run the repository's actual fixture-backed daily pipeline, not a hand-built report."""
    for name in ("config", "rules", "portfolio"):
        shutil.copytree(ROOT / "examples" / "kb" / name, kb / name)
    env = {**os.environ, "ZHIKU_KB": str(kb), "ZHIKU_FIXTURES": str(ROOT / "examples" / "fixtures"),
           "PYTHONPATH": str(ROOT / "src"), "ZHIKU_LLM_BACKEND": "external",
           "ZHIKU_ANALYSIS_LLM": "0", "ZHIKU_LIVE_CALENDAR": "0"}
    command = [sys.executable, "-m", "zhiku.cli", "--provider", "demo", "run-daily", "--force",
               "--skip-sync", "--date", "2026-10-20"]
    with (output / "pipeline.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=120, check=True)
    from zhiku.app.workspace import DailyReport
    report = DailyReport.model_validate_json((kb / "reports" / "latest-daily.json").read_text(encoding="utf-8"))
    assert report.source_mode == "demo" and report.fictional
    assert len(report.candidates) == 0, "Unpublished formulas must not create formal candidates"
    assert len(report.observations) == 7, "Update QA expectations if the checked-in fixture universe changes"
    return {"data": "repository-fixture-pipeline", "asof": report.asof,
            "formal_count": len(report.candidates), "observation_count": len(report.observations),
            "gate_reason": report.gate_artifact.reason, "browser": {}}


def run_pipeline_browser(base: str, output: Path, result: dict[str, Any]) -> None:
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        result["engine"] = {"name": "chromium", "version": browser.version}
        failures = []
        try:
            for name, width, height in (("desktop-1440", 1440, 1000), ("mobile-390", 390, 844)):
                directory = output / name
                directory.mkdir()
                scenario: dict[str, Any] = {"screenshots": [], "viewport": {"width": width, "height": height}}
                result["browser"][name] = scenario
                errors: list[str] = []
                requests: list[str] = []
                context = browser.new_context(viewport={"width": width, "height": height}, locale="zh-CN",
                                              timezone_id="Asia/Shanghai", reduced_motion="reduce")
                context.tracing.start(screenshots=True, snapshots=True, sources=True)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                page.on("request", lambda request: requests.append(request.url))
                try:
                    page.goto(base, wait_until="networkidle")
                    expect(page.locator("#content")).to_have_attribute("aria-busy", "false")
                    expect(page.locator("#mode-badge")).to_contain_text("演示")
                    expect(page.locator("#nav-count")).to_have_text("7")
                    expect(page.locator(".summary-value").first).to_contain_text("0")
                    for route, shot in (("overview", "01-overview"), ("universe", "02-formal-empty"),
                                        ("universe", "03-research-observations"),
                                        ("evidence", "04-evidence"), ("validation", "05-validation")):
                        if route == "universe" and shot.startswith("03"):
                            page.get_by_role("tab", name="研究观察").click()
                            expect(page.locator("tbody tr")).to_have_count(7)
                        else:
                            if width < 700:
                                page.get_by_role("button", name="展开导航").click()
                            page.locator(f'#navigation [data-page="{route}"]').click()
                            expect(page).to_have_url(base + "/#" + route)
                            expect(page.locator("#page-title")).to_have_text({"overview": "今日概览", "universe": "候选与观察",
                                                                           "evidence": "消息与证据", "validation": "研究验证"}[route])
                            if route == "universe":
                                expect(page.locator("#content")).to_contain_text("当前没有正式候选")
                        expect(page.locator("#page-title")).to_be_visible()
                        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1")
                        page.screenshot(path=directory / f"{shot}.png", full_page=True, animations="disabled")
                        scenario["screenshots"].append(f"pipeline/{name}/{shot}.png")
                    assert not errors, errors
                    assert all(url.startswith(base + "/") for url in requests), requests
                    scenario["status"] = "passed"
                except Exception:
                    scenario["status"] = "failed"
                    scenario["failure"] = traceback.format_exc()
                    failures.append(scenario["failure"])
                    page.screenshot(path=directory / "failure.png", full_page=True)
                finally:
                    scenario["console_errors"] = errors
                    (directory / "requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
                    context.tracing.stop(path=directory / "trace.zip")
                    context.close()
            if failures:
                raise AssertionError("\n".join(failures))
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("browser-artifacts"))
    parser.add_argument("--http-only", action="store_true", help="HTTP preflight only; NOT browser QA")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "running", "scope": "http-only" if args.http_only else "real-chromium",
                              "data": "fictional-only", "listener": "127.0.0.1", "browser": {}}
    code = 0
    try:
        with tempfile.TemporaryDirectory(prefix="zhiku-browser-qa-") as temporary:
            pipeline_output = output / "pipeline"
            pipeline_output.mkdir()
            pipeline_kb = Path(temporary) / "pipeline-kb"
            result["pipeline"] = prepare_pipeline_demo(pipeline_kb, pipeline_output)
            browser_failures = []
            with serve(pipeline_kb, pipeline_output) as base:
                status, _, content = fetch(base + "/api/workspace")
                payload = json.loads(content)
                assert status == 200 and payload["report"]["state"] == "ready"
                assert payload["summary"]["formal_candidates"] == 0
                assert payload["summary"]["research_observations"] == 7
                result["pipeline"]["http_status"] = "passed"
                if not args.http_only:
                    try:
                        run_pipeline_browser(base, pipeline_output, result["pipeline"])
                    except Exception:
                        browser_failures.append(traceback.format_exc())
            kb = Path(temporary) / "kb"
            seed_workspace(kb)
            with serve(kb, output) as base:
                result["http"] = check_http(base)
                if not args.http_only:
                    try:
                        run_browser(base, kb, output, result)
                    except Exception:
                        browser_failures.append(traceback.format_exc())
            if browser_failures:
                raise AssertionError("\n".join(browser_failures))
            assert "Traceback (most recent call last)" not in (output / "server.log").read_text(encoding="utf-8")
            result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["failure"] = traceback.format_exc()
        traceback.print_exc()
        code = 1
    finally:
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
