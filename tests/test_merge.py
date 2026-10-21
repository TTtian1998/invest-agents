"""公式分与 qlib 分合并，以及 qlib 缺席时的降级。"""

from __future__ import annotations

from zhiku.factors.merge import merge_scores, scale_merge_settings
from zhiku.factors.qlib_adapter import predict_scores, qlib_available
from zhiku.pipeline.settings import MergeSettings


def test_merge_weighted() -> None:
    names = {"000333": "示例家电个股"}
    formula = {"000333": {"greenblatt": 80.0, "piotroski": 100.0}}  # avg 90
    qlib = {"000333": 60.0}
    merged = merge_scores(names, formula, qlib, MergeSettings(formula=0.6, qlib=0.4))
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


def test_qlib_weight_scales_by_rank_icir() -> None:
    base = MergeSettings(formula=0.6, qlib=0.4, icir_threshold=0.20)
    weak, info = scale_merge_settings(base, 0.05)
    assert abs(weak.qlib - 0.10) < 1e-9
    assert info["qlib_weight_scale"] == 0.25
    names = {"000333": "示例家电个股"}
    formula = {"000333": {"greenblatt": 80.0, "piotroski": 100.0}}
    merged = merge_scores(names, formula, {"000333": 60.0}, weak)
    assert abs(merged[0].total - (0.6 * 90 + 0.1 * 60) / 0.7) < 1e-9
    zero, zinfo = scale_merge_settings(base, -0.01)
    assert zero.qlib == 0.0
    assert zinfo["qlib_weight_scale"] == 0.0
    missing, minfo = scale_merge_settings(base, None)
    assert missing.qlib == 0.0
    assert "缺" in minfo["note"]
    strong, sinfo = scale_merge_settings(base, 0.40)
    assert abs(strong.qlib - 0.4) < 1e-9
    assert sinfo["qlib_weight_scale"] == 1.0


def test_predict_scores_empty_when_qlib_missing() -> None:
    if qlib_available():
        return
    assert predict_scores(None, None) == {}
