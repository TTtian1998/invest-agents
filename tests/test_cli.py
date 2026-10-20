"""CliRunner + fixture：四个命令写出报告；事件日志幂等；debate 友好退出。"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from zhiku.cli import app
from zhiku.review.events import EventLog

RUNNER = CliRunner()


def _env(tmp_kb: Path, examples_dir: Path) -> dict[str, str]:
    return {
        "ZHIKU_KB": str(tmp_kb),
        "ZHIKU_FIXTURES": str(examples_dir / "fixtures"),
        "ZHIKU_ANALYSIS_LLM": "0",
    }


def test_help_lists_commands() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("check", "close", "review", "exposure", "debate", "news", "macro", "daily", "screen"):
        assert name in result.output


def test_four_commands_write_markdown(tmp_kb: Path, examples_dir: Path) -> None:
    env = _env(tmp_kb, examples_dir)
    check = RUNNER.invoke(
        app,
        ["--provider", "fixture", "check", "--at", "1455", "--date", "2026-10-19"],
        env=env,
    )
    assert check.exit_code == 0, check.output
    close = RUNNER.invoke(
        app,
        ["--provider", "fixture", "close", "--date", "2026-10-20"],
        env=env,
    )
    assert close.exit_code == 0, close.output
    expo = RUNNER.invoke(app, ["--provider", "fixture", "exposure"], env=env)
    assert expo.exit_code == 0, expo.output
    review = RUNNER.invoke(
        app,
        ["--provider", "fixture", "review", "--week", "--end", "2026-10-23"],
        env=env,
    )
    assert review.exit_code == 0, review.output

    reports = list((tmp_kb / "reports").rglob("*.md"))
    names = {p.name for p in reports}
    assert "2026-10-19-zk-check.md" in names
    assert "2026-10-20-zk-close.md" in names
    assert "2026-10-23-zk-weekly.md" in names
    assert any(p.name.endswith("zk-exposure.md") for p in reports)

    check_md = (tmp_kb / "reports" / "2026-10" / "2026-10-19-zk-check.md").read_text(encoding="utf-8")
    assert "510300" in check_md
    assert "预警" in check_md or "PREVIEW" in check_md or "已触发" in check_md
    assert "有效线" in check_md
    assert "600519" in check_md

    close_md = (tmp_kb / "reports" / "2026-10" / "2026-10-20-zk-close.md").read_text(encoding="utf-8")
    assert "AAPL" in close_md
    assert "有效线" in close_md
    assert "价源" in close_md

    expo_md = next(p for p in reports if p.name.endswith("zk-exposure.md")).read_text(encoding="utf-8")
    assert "禁加" in expo_md
    assert "510300" in expo_md

    weekly = (tmp_kb / "reports" / "2026-10" / "2026-10-23-zk-weekly.md").read_text(encoding="utf-8")
    assert "命中率" in weekly
    assert "候选教训" in weekly

    assert "禁加" in expo.output

    news = RUNNER.invoke(
        app,
        ["--provider", "fixture", "news", "--date", "2026-10-20"],
        env=env,
    )
    assert news.exit_code == 0, news.output
    assert "意味着什么" in news.output or "so_what" in news.output
    assert "一级" in news.output or "信源" in news.output
    assert "未做深度分析" in news.output
    assert "无" in news.output  # 空档打印「无」或分析正文含无

    macro = RUNNER.invoke(
        app,
        ["--provider", "fixture", "macro", "--date", "2026-10-20"],
        env=env,
    )
    assert macro.exit_code == 0, macro.output
    assert "宏观锚点" in macro.output or "核心日历" in macro.output
    assert "缺数据" in macro.output

    daily = RUNNER.invoke(
        app,
        ["--provider", "fixture", "daily", "--date", "2026-10-20"],
        env=env,
    )
    assert daily.exit_code == 0, daily.output
    assert "宏观锚点" in daily.output


def test_event_log_idempotent(tmp_kb: Path, examples_dir: Path) -> None:
    env = _env(tmp_kb, examples_dir)
    args = ["--provider", "fixture", "close", "--date", "2026-10-20"]
    first = RUNNER.invoke(app, args, env=env)
    second = RUNNER.invoke(app, args, env=env)
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    events = EventLog(tmp_kb / "journal" / "trigger_events.jsonl").read()
    keys = [(e.rule_id, e.date.isoformat()) for e in events]
    assert len(keys) == len(set(keys))
    assert keys  # 收盘日应至少记下一笔虚构示例触发


def test_debate_without_extra_exits_cleanly() -> None:
    result = RUNNER.invoke(app, ["debate", "600519"])
    assert result.exit_code == 1
    text = result.output + (result.stderr or "")
    assert "TradingAgents" in text
    assert "Traceback" not in text
    assert "uv pip install" in text or "pip install" in text
