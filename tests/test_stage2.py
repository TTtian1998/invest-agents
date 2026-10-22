"""第二阶段：明细财务映射、股息率、经典公式门槛、宏观日历、run-daily 跳过。"""

from __future__ import annotations

from datetime import date

import pandas as pd
from typer.testing import CliRunner

from zhiku.cli import app
from zhiku.data.compute import dividend_yield_from_rows, ebit_of, nfa_of, nwc_of
from zhiku.data.detail_fin import map_sina_statement, merge_payload
from zhiku.factors.qlib_adapter import needs_retrain, pick_better
from zhiku.macro.sources import parse_ws_calendar_row
from zhiku.pipeline.settings import PipelineSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.quality import apply_tradable
from zhiku.screens.runner import run_screens
from zhiku.models import Market


def test_sina_mapper_uses_announced_and_fields() -> None:
    df = pd.DataFrame(
        [
            {
                "报告日": "20251231",
                "公告日期": "2026-03-28",
                "营业利润": 80,
                "利息费用": 2,
                "流动资产合计": 200,
                "流动负债合计": 80,
                "固定资产净额": 90,
                "货币资金": 40,
                "经营活动产生的现金流量净额": 60,
            }
        ]
    )
    income = map_sina_statement("600519", "income", df)
    assert not isinstance(income, list) or income
    assert income[0]["announced"] == date(2026, 3, 28)
    assert income[0]["payload"]["营业利润"] == 80
    assert ebit_of(income[0]["payload"]) == 80
    bal = map_sina_statement("600519", "balance", df)
    assert nwc_of(bal[0]["payload"]) is not None
    assert nfa_of(bal[0]["payload"]) == 90


def test_sina_mapper_falls_back_to_lookup() -> None:
    df = pd.DataFrame([{"报告日": "20241231", "营业利润": 10}])
    rows = map_sina_statement(
        "000333", "income", df, announced_lookup={"2024-12-31": date(2025, 3, 31)}
    )
    assert rows[0]["announced"] == date(2025, 3, 31)
    assert rows[0]["announced_is_period"] is False


def test_merge_payload_does_not_overwrite_with_empty() -> None:
    out = merge_payload({"净利润": 10, "营业利润": None}, {"营业利润": 8, "净利润": None})
    assert out["净利润"] == 10
    assert out["营业利润"] == 8


def test_dividend_yield_from_per10_and_stored() -> None:
    df = pd.DataFrame(
        [
            {"ex_date": "2026-06-01", "amount": 5.0, "yield": None},
            {"ex_date": "2025-06-01", "amount": 4.0, "yield": None},
        ]
    )
    y = dividend_yield_from_rows(df, date(2026, 10, 21), raw_close=25.0)
    # (5+4)/10 / 25 = 0.036
    assert y is not None and abs(y - 0.036) < 1e-9
    stored = pd.DataFrame([{"ex_date": "2025-06-01", "amount": 5.0, "yield": 4.2}])
    y2 = dividend_yield_from_rows(stored, date(2026, 10, 21), raw_close=25.0)
    assert y2 is not None and abs(y2 - 0.042) < 1e-9


def test_classic_screens_ignore_quality_gate() -> None:
    cheap = Fundamentals(
        symbol="CHEAP",
        market=Market.CN,
        name="虚构高息",
        instrument="stock",
        mktcap=2e11,
        amount_avg=5e8,
        pe_percentile=0.2,
        pb_percentile=0.2,
        dividend_yield=0.05,
        dividend_years=5,
        ebit=20,
        nwc=5,
        nfa=5,
        ev=80,
        roe=0.05,  # 过不了质量门槛
        ocf=1,
        debt_asset=0.3,
        roa=0.08,
        roa_prev=0.05,
        net_income=8,
        lt_debt_asset=0.1,
        lt_debt_asset_prev=0.2,
        current_ratio=1.5,
        current_ratio_prev=1.1,
        shares_issued=False,
        gross_margin=0.4,
        gross_margin_prev=0.3,
        asset_turnover=0.8,
        asset_turnover_prev=0.6,
    )
    etf = Fundamentals(symbol="510300", instrument="etf", market=Market.CN, name="ETF")
    settings = PipelineSettings()
    bundle = run_screens([cheap, etf], settings)
    assert "CHEAP" not in bundle.quality_passed
    assert any(r.symbol == "CHEAP" for r in bundle.screens["greenblatt"])
    assert any(r.symbol == "CHEAP" for r in bundle.screens["piotroski"])
    assert any(r.symbol == "CHEAP" for r in bundle.screens["dividend"])
    assert "510300" not in {r.symbol for r in bundle.screens["greenblatt"]}


def test_tradable_does_not_fail_on_missing_pe() -> None:
    from zhiku.pipeline.settings import UniverseSettings

    row = Fundamentals(
        symbol="X",
        market=Market.CN,
        instrument="stock",
        mktcap=2e11,
        amount_avg=5e8,
        pe_percentile=None,
        pb_percentile=None,
    )
    q = apply_tradable(row, UniverseSettings())
    assert q.passed


def test_ws_calendar_parser_keeps_surprise_inputs() -> None:
    row = {
        "时间": "2026-10-21 20:30",
        "地区": "美国",
        "事件": "美国CPI月率",
        "重要性": "高",
        "今值": 0.2,
        "预期": 0.3,
        "前值": 0.1,
    }
    ev = parse_ws_calendar_row(row, asof=date(2026, 10, 21), index=0)
    assert ev is not None
    assert ev.event_type == "US_CPI"
    assert ev.previous == 0.1
    assert ev.consensus == 0.3
    assert ev.actual == 0.2
    assert ev.importance >= 4


def test_needs_retrain_monthly() -> None:
    assert needs_retrain(None, date(2026, 10, 21))
    assert needs_retrain({"status": "trained", "trained_on": "2026-09-30"}, date(2026, 10, 21))
    assert not needs_retrain({"status": "trained", "trained_on": "2026-10-01"}, date(2026, 10, 21))
    better = pick_better(
        {"rank_ic": 0.02, "rank_icir": 0.4, "ic": 0.05, "kind": "a"},
        {"rank_ic": 0.04, "rank_icir": 0.2, "ic": 0.03, "kind": "b"},
    )
    assert better["kind"] == "a"


def test_run_daily_skips_weekend(tmp_path, examples_dir, monkeypatch) -> None:
    monkeypatch.setenv("ZHIKU_KB", str(tmp_path))
    monkeypatch.setenv("ZHIKU_FIXTURES", str(examples_dir / "fixtures"))
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--provider", "fixture", "run-daily", "--date", "2026-10-18", "--skip-sync", "--skip-train"],
        env={"ZHIKU_KB": str(tmp_path), "ZHIKU_FIXTURES": str(examples_dir / "fixtures")},
    )
    assert result.exit_code == 0
    assert "非交易日" in (result.output + result.stdout + (result.stderr or ""))


def test_brief_template_under_150_lines() -> None:
    from zhiku.notify.render import render
    from zhiku.pipeline.daily import DailyResult

    daily = DailyResult(
        asof="2026-10-21",
        notes=["测试"],
        candidates=[{"symbol": "600519", "name": "示例", "total": 80, "formula_score": 50, "qlib_score": 30, "screens": {"greenblatt": 80}}],
        backtests={"greenblatt": {"annualized": 0.1, "max_drawdown": -0.1, "sharpe": 0.5, "passed_gate": True}},
        qlib={"kind": "lgb_lite", "ic": 0.05, "rank_ic": 0.02, "icir": 0.4},
        news={"clusters": [], "opportunities": [], "emergency": {"active": False}},
        macro={"geo": {"temperature": "低"}, "rate_expectation": {"source": "fixture"}, "released": [], "calendar_7d": []},
        universe_size=3,
        missing_ratio=0.1,
    )
    md = render("daily_brief.md.j2", daily=daily)
    assert len(md.splitlines()) <= 150
    assert "600519" in md
