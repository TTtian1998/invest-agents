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
