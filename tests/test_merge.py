"""公式分与 qlib 分合并，以及 qlib 缺席时的降级。"""

from __future__ import annotations

from zhiku.factors.merge import (
    OOS_UNVERIFIED_NOTE,
    load_forward_rank_ic,
    merge_scores,
    scale_merge_settings,
)
from zhiku.factors.qlib_adapter import predict_scores, qlib_available
from zhiku.pipeline.settings import MergeSettings


def test_merge_weighted() -> None:
    names = {"000333": "示例家电个股"}
    formula = {"000333": {"greenblatt": 80.0, "piotroski": 100.0}}  # avg 90
    qlib = {"000333": 60.0}
    merged = merge_scores(names, formula, qlib, MergeSettings(formula=0.6, qlib=0.4, formula_first=True))
    assert len(merged) == 1
    assert merged[0].formula_score == 90.0
    assert abs(merged[0].total - (0.6 * 90 + 0.4 * 60)) < 1e-9
    assert merged[0].source == "formula+qlib"


def test_merge_degrades_without_qlib() -> None:
    names = {"A": "甲", "B": "乙"}
    formula = {"A": {"greenblatt": 50.0}, "B": {"greenblatt": 70.0}}
    merged = merge_scores(names, formula, None, MergeSettings())
    assert [c.symbol for c in merged] == ["B", "A"]
    assert all(c.qlib_score is None for c in merged)
    assert all(c.source == "formula" for c in merged)
    assert merged[0].total == 70.0


def test_qlib_weight_locks_aux_when_rank_ic_nonpositive() -> None:
    base = MergeSettings()
    missing, minfo = scale_merge_settings(base, None)
    assert abs(missing.qlib - 0.10) < 1e-9
    assert minfo["oos_unverified"] is True
    assert OOS_UNVERIFIED_NOTE in minfo["note"]
    neg, ninfo = scale_merge_settings(base, -0.053)
    assert abs(neg.qlib - 0.10) < 1e-9
    assert ninfo["oos_unverified"] is True
    assert OOS_UNVERIFIED_NOTE in ninfo["note"]
    names = {"000333": "示例家电个股"}
    formula = {"000333": {"piotroski": 90.0}}
    merged = merge_scores(names, formula, {"000333": 60.0}, neg)
    assert abs(merged[0].total - (0.90 * 90 + 0.10 * 60)) < 1e-9


def test_qlib_weight_rolls_up_on_live_positive_rank_ic() -> None:
    base = MergeSettings()
    mid, info = scale_merge_settings(base, 0.015, source="live")
    assert info["oos_unverified"] is False
    assert abs(mid.qlib - (0.10 + 0.30 * 0.5)) < 1e-9
    strong, sinfo = scale_merge_settings(base, 0.06, source="live")
    assert abs(strong.qlib - 0.40) < 1e-9
    assert sinfo["qlib_weight_scale"] == 1.0


def test_merge_formula_first_drops_qlib_only() -> None:
    names = {"000333": "甲", "600519": "乙"}
    formula = {"000333": {"piotroski": 80.0}}
    qlib = {"000333": 40.0, "600519": 99.0}
    merged = merge_scores(names, formula, qlib, MergeSettings(formula_first=True))
    assert [c.symbol for c in merged] == ["000333"]
    open_pool = merge_scores(names, formula, qlib, MergeSettings(formula_first=False))
    assert {c.symbol for c in open_pool} == {"000333", "600519"}


def test_merge_fails_closed_when_formula_empty() -> None:
    names = {"600519": "乙"}
    merged = merge_scores(names, {}, {"600519": 70.0}, MergeSettings(formula_first=True))
    assert merged == []
    observations = merge_scores(names, {}, {"600519": 70.0}, MergeSettings(formula_first=False))
    assert observations[0].source == "qlib"


def test_load_forward_rank_ic_missing_is_locked_test(tmp_path) -> None:
    ic, src = load_forward_rank_ic(tmp_path)
    assert ic is None
    assert src == "locked_test"
    (tmp_path / "live_rank_ic.json").write_text('{"rank_ic": 0.02}', encoding="utf-8")
    ic2, src2 = load_forward_rank_ic(tmp_path)
    assert abs(float(ic2) - 0.02) < 1e-9
    assert src2 == "live"


def test_predict_scores_empty_when_qlib_missing() -> None:
    if qlib_available():
        return
    assert predict_scores(None, None) == {}
