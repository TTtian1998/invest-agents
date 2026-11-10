"""Snapshot identity, evidence validation, and one-pass daily review, without network."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from zhiku.ai.backend import ExternalBackend, MockBackend
from zhiku.ai.journal import DecisionJournal
from zhiku.ai.pack import (
    build_review_pack,
    load_review_snapshot,
    save_review_snapshot,
    validate_review_pack,
    validate_review_result,
)
from zhiku.ai.runner import run_ai_layer
from zhiku.ai.schema import CitedClaim, Extraction, Judgment, RegimeAdvice, ReviewResult
from zhiku.config import Paths
from zhiku.pipeline.settings import PipelineSettings
from zhiku.review.audit import audit_top
from tests.test_screens import _row

DAY = date(2026, 10, 21)


def _pack(**changes) -> dict:
    args = {
        "asof": DAY,
        "candidates": [{"symbol": "AAA", "formula_score": 80}],
        "rows": [_row(symbol="AAA", roe=0.2)],
        "masters": ["graham"],
        "news": {"clusters": [{"symbol": "AAA", "headline": "公司公告", "summary": "收入增长"}]},
        "macro": {"geo": {"temperature": "低"}},
        "trailing": {"universe_12_1": 0.1},
    }
    args.update(changes)
    return build_review_pack(**args)


def _result(pack: dict) -> ReviewResult:
    return ReviewResult(
        asof=pack["asof"], snapshot_id=pack["snapshot_id"], backend="external",
        judgments=[Judgment(
            symbol=item["candidate"]["symbol"], stance="观察", confidence=0.4,
            invalidation="ROE 转负", long_case=[CitedClaim(claim="检查 ROE", cite="roe")],
        ) for item in pack["candidates"]],
    )


def _runner_args(tmp_path: Path, examples_dir: Path) -> dict:
    paths = Paths.load(kb=tmp_path, fixtures=examples_dir / "fixtures")
    paths.ensure_runtime_dirs()
    return {
        "asof": DAY,
        "paths": paths,
        "settings": PipelineSettings(),
        "candidates": [{"symbol": "AAA", "formula_score": 80}],
        "rows": [_row(symbol="AAA", roe=0.2)],
        "news": None,
        "macro": None,
    }


def test_snapshot_hash_is_deterministic_and_input_detached() -> None:
    news = {"clusters": [{"symbol": "AAA", "headline": "公告", "summary": "收入增长"}]}
    trailing = {"universe_12_1": 0.1, "nested": {"value": 2}}
    first = _pack(news=news, trailing=trailing)
    second = _pack(news=news, trailing={"nested": {"value": 2}, "universe_12_1": 0.1})
    assert first["snapshot_id"] == second["snapshot_id"]
    trailing["nested"]["value"] = 3
    news["clusters"][0]["headline"] = "后来公告"
    assert validate_review_pack(first) is first
    assert first["trailing"]["nested"]["value"] == 2
    assert first["news_excerpts"][0]["headline"] == "公告"


@pytest.mark.parametrize("changes", [
    {"asof": date(2026, 10, 22)},
    {"candidates": [{"symbol": "AAA", "formula_score": 81}]},
    {"candidates": [{"symbol": "AAA", "formula_score": 80}, {"symbol": "BBB"}],
     "rows": [_row(symbol="AAA", roe=0.2), _row(symbol="BBB")]},
    {"rows": [_row(symbol="AAA", roe=0.21)]},
    {"masters": ["buffett"]},
    {"news": {"clusters": [{"symbol": "AAA", "headline": "新公告"}]}},
    {"macro": {"geo": {"temperature": "高"}}},
    {"trailing": {"universe_12_1": 0.2}},
])
def test_snapshot_identity_covers_candidates_and_evidence(changes) -> None:
    assert _pack()["snapshot_id"] != _pack(**changes)["snapshot_id"]


def test_saved_snapshots_are_immutable_and_latest_alias_is_validated(tmp_path: Path) -> None:
    first = _pack()
    first_path = save_review_snapshot(tmp_path, first)
    original = first_path.read_bytes()
    newer = _pack(rows=[_row(symbol="AAA", roe=0.3)])
    save_review_snapshot(tmp_path, newer)
    assert first_path.read_bytes() == original
    assert load_review_snapshot(tmp_path, asof=DAY)["snapshot_id"] == newer["snapshot_id"]
    assert load_review_snapshot(tmp_path, asof=DAY, snapshot_id=first["snapshot_id"]) == first
    alias = tmp_path / f"{DAY}-review-pack.json"
    tampered = deepcopy(newer)
    tampered["candidates"][0]["snapshot"]["roe"] = 99
    alias.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_id"):
        load_review_snapshot(tmp_path, asof=DAY)
    with pytest.raises(ValueError, match="日期"):
        load_review_snapshot(tmp_path, asof="2026-10-20", snapshot_id=first["snapshot_id"])


def test_archive_tampering_cannot_be_hidden_by_valid_date_alias(tmp_path: Path) -> None:
    pack = _pack()
    archive = save_review_snapshot(tmp_path, pack)
    tampered = deepcopy(pack)
    tampered["trailing"]["universe_12_1"] = 999
    archive.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_id"):
        load_review_snapshot(tmp_path, asof=DAY)
    with pytest.raises(ValueError, match="snapshot_id"):
        save_review_snapshot(tmp_path, pack)


@pytest.mark.parametrize("field,value", [
    ("snapshot_id", ""),
    ("snapshot_id", "f" * 64),
    ("asof", "2026-10-20"),
])
def test_result_must_echo_exact_snapshot_and_date(field, value) -> None:
    pack = _pack()
    result = _result(pack)
    setattr(result, field, value)
    with pytest.raises(ValueError, match=field):
        validate_review_result(result, pack)


def test_result_checks_candidates_metadata_and_citations() -> None:
    pack = _pack()
    result = _result(pack)
    checked = validate_review_result(result, pack)
    assert checked.judgments[0].formula_rank == 1
    assert checked.judgments[0].formula_score == 80
    assert result.judgments[0].formula_rank is None  # validator does not mutate caller
    for field, value in [("symbol", "OUTSIDE"), ("formula_rank", 2), ("formula_score", 99)]:
        invalid = _result(pack)
        setattr(invalid.judgments[0], field, value)
        with pytest.raises(ValueError):
            validate_review_result(invalid, pack)
    invalid = _result(pack)
    invalid.judgments.append(invalid.judgments[0].model_copy())
    with pytest.raises(ValueError, match="重复"):
        validate_review_result(invalid, pack)
    invalid = _result(pack)
    invalid.judgments[0].long_case[0].cite = "unprovided_future_price"
    with pytest.raises(ValueError, match="引用"):
        validate_review_result(invalid, pack)


def test_result_cannot_cite_another_candidates_news() -> None:
    pack = _pack(news={"clusters": [{"symbol": "BBB", "headline": "独有未提供证据"}]})
    result = _result(pack)
    result.judgments[0].long_case[0].cite = "独有未提供证据"
    with pytest.raises(ValueError, match="引用"):
        validate_review_result(result, pack)


def test_journal_stores_coherent_result_and_retains_scorecard_events(tmp_path: Path) -> None:
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    pack = _pack()
    result = _result(pack)
    assert journal.append_result(result, pack=pack) == 2
    previous = journal.latest_result_for(pack)
    assert previous is not None
    assert previous.snapshot_id == pack["snapshot_id"]
    assert {entry["kind"] for entry in journal.read()} == {"judgment", "review_result"}
    assert journal.append_result(result, pack=pack) == 0
    assert len(journal.read()) == 2
    assert journal.latest_result_for(_pack(rows=[_row(symbol="AAA", roe=0.5)])) is None
    changed = _result(pack)
    changed.judgments[0].stance = "回避"
    with pytest.raises(ValueError, match="不同研判"):
        journal.append_result(changed, pack=pack)
    with pytest.raises(ValueError, match="pack"):
        journal.append_result(result)


def test_journal_rejects_stale_and_invalid_review_events(tmp_path: Path) -> None:
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    pack = _pack()
    journal.append({"kind": "judgment", "asof": pack["asof"], "symbol": "AAA", "stance": "研究跟进"})
    assert journal.latest_result_for(pack) is None
    invalid = _result(pack)
    invalid.judgments[0].symbol = "BBB"
    journal.append({"kind": "review_result", "asof": pack["asof"],
                    "snapshot_id": pack["snapshot_id"], "payload": invalid.to_dict()})
    assert journal.latest_result_for(pack) is None
    journal.path.write_text(journal.path.read_text() + "{broken\n", encoding="utf-8")
    assert journal.latest_result_for(pack) is None


def test_daily_checklist_never_uses_legacy_llm_or_debate(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("daily rules must not call legacy AI")

    monkeypatch.setenv("OPENAI_API_KEY", "fake-but-present")
    monkeypatch.setattr("zhiku.review.masters._call_openai", forbidden)
    monkeypatch.setattr("zhiku.debate.tradingagents_adapter.run_debate", forbidden)
    cards = audit_top([_row(symbol="AAA")], ["graham", "buffett"])
    assert len(cards) == 1 and cards[0].opinions
    assert cards[0].debate is None
    assert "补充" not in str(cards[0].to_dict())


def test_runner_uses_one_call_per_candidate_then_replays(tmp_path: Path, examples_dir: Path) -> None:
    class CountingBackend(MockBackend):
        calls = 0
        prompts: list[str] = []

        def complete_json(self, system, user):
            self.calls += 1
            self.prompts.append(user)
            return super().complete_json(system, user)

    args = _runner_args(tmp_path, examples_dir)
    args["candidates"].append({"symbol": "BBB", "formula_score": 79})
    args["rows"].append(_row(symbol="BBB", roe=0.15))
    backend = CountingBackend()
    result, path, _ = run_ai_layer(**args, backend=backend)
    assert result is not None and path is not None
    assert backend.calls == 2
    assert len(result.judgments) == 2 and result.extractions == [] and result.regime is None
    assert all(json.loads(prompt)["snapshot_id"] == result.snapshot_id for prompt in backend.prompts)
    first_bytes = (args["paths"].journal / "ai_decisions.jsonl").read_bytes()
    again, same_path, notes = run_ai_layer(**args, backend=backend)
    assert backend.calls == 2
    assert again == result and same_path == path
    assert any("复用" in note for note in notes)
    assert (args["paths"].journal / "ai_decisions.jsonl").read_bytes() == first_bytes


def test_external_ingest_reused_only_for_exact_snapshot(tmp_path: Path, examples_dir: Path) -> None:
    args = _runner_args(tmp_path, examples_dir)
    result, archive, _ = run_ai_layer(**args, backend=ExternalBackend())
    assert result is None and archive is not None
    journal = DecisionJournal(args["paths"].journal / "ai_decisions.jsonl")
    pack = load_review_snapshot(args["paths"].reports, asof=DAY)
    run_ai_layer(**args, backend=ExternalBackend())
    assert len(journal.read()) == 1  # repeated exports are idempotent
    journal.append_result(_result(pack), pack=pack)
    ingested, _, _ = run_ai_layer(**args, backend=ExternalBackend())
    assert ingested is not None and ingested.snapshot_id == pack["snapshot_id"]
    args["rows"] = [_row(symbol="AAA", roe=0.9)]
    stale, newer_archive, _ = run_ai_layer(**args, backend=ExternalBackend())
    assert stale is None and newer_archive != archive
    assert len([entry for entry in journal.read() if entry["kind"] == "judgment"]) == 1


def test_runner_preserves_failed_snapshot_for_retry(tmp_path: Path, examples_dir: Path) -> None:
    class FailedBackend:
        name = "broken"

        def complete_json(self, system, user):
            raise TimeoutError("offline test")

    args = _runner_args(tmp_path, examples_dir)
    result, path, notes = run_ai_layer(**args, backend=FailedBackend())
    assert result is None and path is not None and path.is_file()
    assert any("不装成已判断" in note for note in notes)
    assert not DecisionJournal(args["paths"].journal / "ai_decisions.jsonl").read()
    recovered, _, _ = run_ai_layer(**args, backend=MockBackend())
    assert recovered is not None


def test_complete_pack_backend_cannot_return_stale_same_day_result(tmp_path: Path, examples_dir: Path) -> None:
    class StaleBackend:
        name = "stale"

        def complete_review_pack(self, pack):
            result = _result(pack)
            result.snapshot_id = "f" * 64
            return result

    args = _runner_args(tmp_path, examples_dir)
    result, _, notes = run_ai_layer(**args, backend=StaleBackend())
    assert result is None and any("snapshot_id" in note for note in notes)
    assert not DecisionJournal(args["paths"].journal / "ai_decisions.jsonl").read()


def test_empty_review_and_ungrounded_extra_fields_are_rejected() -> None:
    pack = _pack()
    result = _result(pack)
    result.judgments = []
    with pytest.raises(ValueError, match="候选判断"):
        validate_review_result(result, pack)
    result = _result(pack)
    result.extractions = [Extraction(symbol="AAA", cite="")]
    with pytest.raises(ValueError, match="引用"):
        validate_review_result(result, pack)
    result = _result(pack)
    result.regime = RegimeAdvice(cite="unprovided_macro_indicator")
    with pytest.raises(ValueError, match="引用"):
        validate_review_result(result, pack)
