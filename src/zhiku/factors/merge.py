"""公式分与 qlib 分加权合并。qlib 缺失时只保留公式分。"""

from __future__ import annotations

from dataclasses import dataclass, field

from zhiku.pipeline.settings import MergeSettings
from zhiku.screens.runner import average_formula_score


@dataclass
class ScoredCandidate:
    symbol: str
    name: str
    formula_score: float
    qlib_score: float | None
    total: float
    screens: dict[str, float] = field(default_factory=dict)
    source: str = "formula"


def scale_merge_settings(
    settings: MergeSettings,
    rank_icir: float | None,
) -> tuple[MergeSettings, dict]:
    """量化权重按样本外 RankICIR / 阈值缩放，弱于阈值就降权，负 ICIR 权重视为 0。"""
    thr = float(settings.icir_threshold or 0.0)
    if rank_icir is None:
        scale = 0.0
        note = "缺样本外 RankICIR，量化权重降为 0"
    elif thr <= 0:
        scale = 1.0
        note = f"未设 RankICIR 阈值，量化权重保持 {settings.qlib:.2f}"
    else:
        scale = min(1.0, max(0.0, float(rank_icir) / thr))
        note = (
            f"量化分权重按样本外 RankICIR 自动缩放：{float(rank_icir):.4f} / 阈值 {thr:.2f}"
            f" → 缩放 {scale:.2f}，有效权重 {settings.qlib * scale:.2f}"
            + ("（弱于阈值已降权）" if scale < 1.0 else "")
        )
    scaled = settings.model_copy(update={"qlib": settings.qlib * scale})
    info = {
        "rank_icir": rank_icir,
        "icir_threshold": thr,
        "qlib_weight_base": settings.qlib,
        "qlib_weight_scale": scale,
        "qlib_weight_effective": scaled.qlib,
        "formula_weight": settings.formula,
        "note": note,
    }
    return scaled, info


def merge_scores(
    names: dict[str, str],
    formula: dict[str, dict[str, float]],
    qlib: dict[str, float] | None,
    settings: MergeSettings,
) -> list[ScoredCandidate]:
    qlib = qlib or {}
    symbols = set(formula) | set(qlib)
    wf, wq = settings.formula, settings.qlib
    wsum = wf + wq
    if wsum <= 0:
        wf, wq, wsum = 1.0, 0.0, 1.0
    out: list[ScoredCandidate] = []
    for symbol in symbols:
        per = formula.get(symbol, {})
        fscore = average_formula_score(per) if per else None
        qscore = qlib.get(symbol)
        if fscore is None and qscore is None:
            continue
        if qscore is None:
            total = float(fscore or 0.0)
            source = "formula"
        elif fscore is None:
            total = float(qscore)
            source = "qlib"
        else:
            total = (wf * fscore + wq * qscore) / wsum
            source = "formula+qlib"
        out.append(
            ScoredCandidate(
                symbol=symbol,
                name=names.get(symbol, symbol),
                formula_score=float(fscore or 0.0),
                qlib_score=qscore,
                total=total,
                screens=per,
                source=source,
            )
        )
    out.sort(key=lambda c: c.total, reverse=True)
    return out
