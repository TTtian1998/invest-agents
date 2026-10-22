"""AI 研判层：模拟后端、任务包写回、scorecard。不访问网络。"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from zhiku.ai.backend import MockBackend, resolve_backend
from zhiku.ai.extract import extract_text_factor, factor_value
from zhiku.ai.journal import DecisionJournal
from zhiku.ai.judge import judge_symbol
from zhiku.ai.pack import RESULT_SCHEMA, build_review_pack, load_review_result, result_from_parts, save_review_snapshot, write_review_pack
from zhiku.ai.regime import advise_regime
from zhiku.ai.schema import CitedClaim, Judgment, ReviewResult
from zhiku.ai.scorecard import build_scorecard
from zhiku.cli import app
from tests.test_screens import _row

RUNNER = CliRunner()


def test_resolve_backend_auto_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ZHIKU_LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.delenv("ZHIKU_LLM_BACKEND", raising=False)
    assert resolve_backend("auto").name == "external"
    monkeypatch.setenv("ZHIKU_LLM_BACKEND", "mock")
    assert resolve_backend().name == "mock"


def test_mock_extract_and_factor() -> None:
    backend = MockBackend()
    ext = extract_text_factor(
        backend,
        symbol="600000",
        headline="公司首次中标超预期大单",
        body="公告称中标金额高于市场预期。",
    )
    assert ext.direction == 1
    assert ext.cite
    assert factor_value(ext) is not None


def test_mock_judge_requires_cite() -> None:
    row = _row(
        symbol="000333",
        name="虚构家电",
        pe=10,
        pb=1.2,
        roe=0.2,
        ocf=12,
        debt_asset=0.3,
        momentum_12_1=0.1,
        eps_growth=0.2,
        ncav=8e10,
    )
    judged = judge_symbol(MockBackend(), row, masters=["graham", "livermore"], formula_rank=1)
    assert judged.stance in {"研究跟进", "观察", "回避"}
    assert judged.long_case or judged.short_case
    assert all(c.cite for c in judged.long_case + judged.short_case)


def test_cited_claim_rejects_empty() -> None:
    try:
        CitedClaim(claim="空口", cite="  ")
    except Exception:
        return
    raise AssertionError("缺 cite 应失败")


def test_pack_roundtrip(tmp_path: Path) -> None:
    row = _row(symbol="000333", name="虚构家电", pe=8, pb=1, roe=0.18, ocf=10, momentum_12_1=0.2)
    pack = build_review_pack(
        asof=date(2026, 10, 21),
        candidates=[{"symbol": "000333", "name": "虚构家电", "formula_score": 80, "screens": {"graham": 80}}],
        rows=[row],
        masters=["graham"],
        news={"clusters": [{"symbol": "000333", "headline": "虚构中报增长", "summary": "收入增长"}]},
        macro={"geo": {"temperature": "低"}},
    )
    assert pack["schema"] == "zhiku.review_pack.v1"
    assert pack["candidates"][0]["checklists"]
    dest = tmp_path / "pack.json"
    write_review_pack(dest, pack)
    result = ReviewResult(
        schema_name=RESULT_SCHEMA,
        asof="2026-10-21",
        backend="external",
        judgments=[
            Judgment(
                symbol="000333",
                name="虚构家电",
                stance="研究跟进",
                confidence=0.6,
                invalidation="ROE 转负",
                long_case=[CitedClaim(claim="ROE 仍高", cite="roe")],
                short_case=[CitedClaim(claim="估值不便宜则放弃", cite="pe")],
                formula_rank=1,
            )
        ],
    )
    out = tmp_path / "result.json"
    out.write_text(
        '{"schema":"zhiku.review_result.v1","asof":"2026-10-21","backend":"external",'
        '"judgments":[{"symbol":"000333","stance":"研究跟进","confidence":0.6,'
        '"invalidation":"ROE 转负","long":[{"claim":"ROE 仍高","cite":"roe"}],'
        '"short":[{"claim":"估值不便宜则放弃","cite":"pe"}],"formula_rank":1}]}',
        encoding="utf-8",
    )
    loaded = load_review_result(out)
    assert loaded.judgments[0].long_case[0].cite == "roe"
    journal = DecisionJournal(tmp_path / "ai_decisions.jsonl")
    assert journal.append_result(result) >= 1
    assert any(r.get("kind") == "judgment" for r in journal.read())


def test_regime_mock() -> None:
    advice = advise_regime(MockBackend(), trailing={"universe_12_1": 0.12})
    assert advice.prefer_screens
    assert advice.cite


def test_pack_skips_missing_symbols_without_changing_ranks() -> None:
    pack = build_review_pack(
        asof=date(2026, 10, 21),
        candidates=[{}, {"symbol": None}, {"symbol": "000333"}],
        rows=[_row(symbol="000333")],
        masters=["graham"],
    )
    assert len(pack["candidates"]) == 1
    assert pack["candidates"][0]["rank"] == 3
    assert pack["candidates"][0]["candidate"]["symbol"] == "000333"


def test_result_from_parts_preserves_schema_alias() -> None:
    result = result_from_parts(
        asof="2026-10-21", backend="mock", extractions=[], judgments=[], regime=None
    )
    assert result.schema_name == RESULT_SCHEMA
    assert result.to_dict()["schema"] == RESULT_SCHEMA
    assert "schema_name" not in result.to_dict()


def test_scorecard_follow_vs_avoid(tmp_path: Path) -> None:
    idx = pd.bdate_range("2026-01-05", periods=40)
    prices = pd.DataFrame(
        {
            "WIN": [1.0 + 0.02 * i for i in range(len(idx))],
            "LAG": [1.0 - 0.01 * i for i in range(len(idx))],
            "MID": [1.0 + 0.001 * i for i in range(len(idx))],
        },
        index=idx,
    )
    journal = DecisionJournal(tmp_path / "ai_decisions.jsonl")
    asof = idx[0].date().isoformat()
    journal.append(
        {
            "kind": "judgment",
            "asof": asof,
            "symbol": "WIN",
            "stance": "研究跟进",
            "formula_rank": 1,
        }
    )
    journal.append(
        {
            "kind": "judgment",
            "asof": asof,
            "symbol": "LAG",
            "stance": "回避",
            "formula_rank": 2,
        }
    )
    card = build_scorecard(journal, prices, formula_top_n=10)
    follow = card.groups["研究跟进"][5]
    avoid = card.groups["回避"][5]
    assert follow.n == 1 and follow.excess is not None and follow.excess > 0
    assert avoid.n == 1 and avoid.excess is not None and avoid.excess < 0
    assert "前瞻记录" in card.notes[0]


def test_cli_ingest_and_scorecard(tmp_path: Path, examples_dir: Path) -> None:
    env = {
        "ZHIKU_KB": str(tmp_path),
        "ZHIKU_FIXTURES": str(examples_dir / "fixtures"),
        "ZHIKU_ANALYSIS_LLM": "0",
        "ZHIKU_LLM_BACKEND": "external",
    }
    result_path = tmp_path / "in.json"
    result_path.write_text(
        '{"schema":"zhiku.review_result.v1","asof":"2026-01-31","backend":"external",'
        '"judgments":[{"symbol":"000333","stance":"观察","confidence":0.4,'
        '"invalidation":"破净","long":[{"claim":"清单中性","cite":"roe"}],'
        '"short":[{"claim":"动量未知","cite":"momentum_12_1"}]}]}',
        encoding="utf-8",
    )
    from zhiku.config import Paths

    paths = Paths.load(kb=tmp_path, fixtures=examples_dir / "fixtures")
    pack = build_review_pack(
        asof=date(2026, 1, 31), candidates=[{"symbol": "000333"}],
        rows=[_row(symbol="000333")], masters=["graham"],
    )
    save_review_snapshot(paths.reports, pack)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["snapshot_id"] = pack["snapshot_id"]
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    ingested = RUNNER.invoke(app, ["--provider", "fixture", "ingest-review", str(result_path)], env=env)
    assert ingested.exit_code == 0, ingested.output
    scored = RUNNER.invoke(app, ["--provider", "fixture", "review-scorecard"], env=env)
    assert scored.exit_code == 0, scored.output
    assert "回避" in scored.output or "研究跟进" in scored.output or "公式排名" in scored.output


def test_run_ai_layer_mock_journals(tmp_path: Path, examples_dir: Path) -> None:
    from zhiku.ai.runner import run_ai_layer
    from zhiku.config import Paths
    from zhiku.pipeline.settings import PipelineSettings

    paths = Paths.load(kb=tmp_path, fixtures=examples_dir / "fixtures")
    paths.ensure_runtime_dirs()
    result, pack, notes = run_ai_layer(
        asof=date(2026, 10, 21),
        paths=paths,
        settings=PipelineSettings(),
        candidates=[{"symbol": "000333", "name": "虚构家电", "formula_score": 80}],
        rows=[
            _row(
                symbol="000333",
                name="虚构家电",
                pe=10,
                pb=1.2,
                roe=0.2,
                ocf=12,
                debt_asset=0.3,
                momentum_12_1=0.1,
                eps_growth=0.2,
            )
        ],
        news={
            "clusters": [
                {
                    "symbol": "000333",
                    "headline": "虚构家电首次中标超预期大单",
                    "summary": "公告称中标金额高于市场预期。",
                }
            ]
        },
        macro={"geo": {"temperature": "低"}},
        trailing={"universe_12_1": 0.12},
        backend=MockBackend(),
    )
    assert result is not None
    assert result.backend == "mock"
    assert result.judgments and result.judgments[0].long_case[0].cite
    assert result.snapshot_id
    assert result.regime is None and result.extractions == []
    assert pack is not None and pack.is_file()
    journal = DecisionJournal(paths.journal / "ai_decisions.jsonl")
    rows = journal.read()
    assert {"judgment", "review_result"} <= {r["kind"] for r in rows}
    assert all(r.get("ts") for r in rows)
    del notes


def test_docs_state_lookahead() -> None:
    text = Path("docs/ai_layer.md").read_text(encoding="utf-8")
    assert "Sarkar" in text and "Lopez-Lira" in text
    assert "TradingAgents" in text and "FinGPT" in text
    assert "review_pack.v1" in text
    assert "CURSOR_API_KEY" in text
    assert "cursor-sdk" in text
    assert ("crs" + "r_") not in text
