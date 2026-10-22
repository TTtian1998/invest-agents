"""公式分与 qlib 分加权合并。量化分默认只做排序辅助；过闸公式优先。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from zhiku.pipeline.settings import MergeSettings
from zhiku.screens.runner import average_formula_score

LIVE_RANK_IC_NAME = "live_rank_ic.json"
AUX_QLIB_WEIGHT = 0.10
OOS_UNVERIFIED_NOTE = "量化样本外未验证"


@dataclass
class ScoredCandidate:
    symbol: str
    name: str
    formula_score: float
    qlib_score: float | None
    total: float
    screens: dict[str, float] = field(default_factory=dict)
    source: str = "formula"


def load_forward_rank_ic(models_dir: Path | None) -> tuple[float | None, str]:
    """前瞻实盘 RankIC。没有文件则视为锁定测试集未验证，不读验证集 ICIR。"""
    if models_dir is None:
        return None, "locked_test"
    path = Path(models_dir) / LIVE_RANK_IC_NAME
    if not path.is_file():
        return None, "locked_test"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "locked_test"
    if not isinstance(raw, dict):
        return None, "locked_test"
    val = raw.get("rank_ic")
    try:
        return (float(val) if val is not None else None), "live"
    except (TypeError, ValueError):
        return None, "locked_test"


def scale_merge_settings(
    settings: MergeSettings,
    rank_ic: float | None,
    *,
    source: str = "locked_test",
) -> tuple[MergeSettings, dict]:
    """量化权重：测试集 RankIC≤0 时锁 0.10 仅排序辅助；正的前瞻实盘 RankIC 再往上滚。"""
    floor = float(getattr(settings, "qlib_floor", None) or AUX_QLIB_WEIGHT)
    cap = float(getattr(settings, "qlib_cap", None) or 0.40)
    if cap < floor:
        cap = floor
    thr = float(getattr(settings, "rank_ic_threshold", None) or 0.03)
    formula_base = float(settings.formula)
    if rank_ic is None or float(rank_ic) <= 0:
        effective = floor
        scale = 0.0
        unverified = True
        if rank_ic is None:
            note = (
                f"{OOS_UNVERIFIED_NOTE}。未提供可用的前瞻 RankIC，量化分权重 {effective:.2f}，"
                "仅做过闸公式内排序辅助。以后按前瞻实盘 RankIC 滚动更新。"
            )
        else:
            note = (
                f"{OOS_UNVERIFIED_NOTE}。{source} RankIC {float(rank_ic):.4f}≤0，"
                f"量化分权重锁 {effective:.2f}，仅做过闸公式内排序辅助。"
                "以后按前瞻实盘 RankIC 滚动更新。"
            )
    else:
        frac = min(1.0, float(rank_ic) / thr) if thr > 0 else 1.0
        effective = floor + (cap - floor) * frac
        scale = frac
        unverified = False
        note = (
            f"前瞻实盘 RankIC {float(rank_ic):.4f} / 阈值 {thr:.2f}，"
            f"量化权重由 {floor:.2f} 滚到 {effective:.2f}（上限 {cap:.2f}）。"
        )
    formula_w = max(0.0, 1.0 - effective)
    if formula_base > 0 and abs(formula_base + float(settings.qlib) - 1.0) > 1e-9:
        # 配置不是归一化的也按有效量化权重点公式
        formula_w = max(0.0, formula_base)
    scaled = settings.model_copy(update={"qlib": effective, "formula": formula_w})
    info = {
        "rank_ic": rank_ic,
        "rank_ic_source": source,
        "rank_ic_threshold": thr,
        "qlib_weight_base": floor,
        "qlib_weight_cap": cap,
        "qlib_weight_scale": scale,
        "qlib_weight_effective": scaled.qlib,
        "formula_weight": scaled.formula,
        "formula_first": bool(getattr(settings, "formula_first", True)),
        "oos_unverified": unverified,
        "aux_qlib": True,
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
    formula_first = bool(getattr(settings, "formula_first", True))
    if formula_first:
        symbols = set(formula)
    else:
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
