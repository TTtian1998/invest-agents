"""排名与 0–100 分数。缺值的标的不参与该因子排名。"""

from __future__ import annotations

from zhiku.models import MissingData


def minmax_from_ranks(ranks: dict[str, float]) -> dict[str, float]:
    """排名 1 最好 → 100 分。"""
    if not ranks:
        return {}
    n = len(ranks)
    if n == 1:
        return {k: 100.0 for k in ranks}
    return {k: 100.0 * (n - r) / (n - 1) for k, r in ranks.items()}


def rank_higher_better(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return _avg_ranks(ordered)


def rank_lower_better(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    return _avg_ranks(ordered)


def _avg_ranks(ordered: list[tuple[str, float]]) -> dict[str, float]:
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def drop_missing(mapping: dict[str, float | MissingData]) -> dict[str, float]:
    return {k: v for k, v in mapping.items() if not isinstance(v, MissingData)}
