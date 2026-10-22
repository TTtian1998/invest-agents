"""AI 研究跟进 / 回避 在 T+5、T+20 对同池等权的超额，对照纯公式排名。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from zhiku.ai.journal import DecisionJournal
from zhiku.calendar import add_trading_days
from zhiku.models import Market, MissingData


HORIZONS = (5, 20)


@dataclass
class GroupHorizon:
    n: int
    pending: int
    excess: float | None


@dataclass
class Scorecard:
    asof_max: str | None
    horizons: list[int]
    groups: dict[str, dict[int, GroupHorizon]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"asof_max": self.asof_max, "notes": list(self.notes), "groups": {}}
        for name, hz in self.groups.items():
            out["groups"][name] = {
                str(h): {"n": g.n, "pending": g.pending, "excess": g.excess} for h, g in hz.items()
            }
        return out


def _snap(index: pd.DatetimeIndex, day: date) -> pd.Timestamp | None:
    ts = pd.Timestamp(day)
    if ts in index:
        return ts
    loc = index.searchsorted(ts, side="right") - 1
    if loc < 0:
        return None
    return index[int(loc)]


def _fwd_return(prices: pd.DataFrame, symbol: str, start: date, horizon: int) -> float | None:
    if symbol not in prices.columns:
        return None
    end = add_trading_days(Market.CN, start, horizon)
    if isinstance(end, MissingData):
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(prices.index))
    t0 = _snap(idx, start)
    t1 = _snap(idx, end if isinstance(end, date) else start)
    if t0 is None or t1 is None or t1 <= t0:
        return None
    p0 = prices.loc[t0, symbol]
    p1 = prices.loc[t1, symbol]
    if pd.isna(p0) or pd.isna(p1) or float(p0) == 0:
        return None
    return float(p1) / float(p0) - 1.0


def _ew_return(prices: pd.DataFrame, start: date, horizon: int, members: list[str] | None) -> float | None:
    cols = [c for c in (members or list(prices.columns)) if c in prices.columns]
    rets = [_fwd_return(prices, c, start, horizon) for c in cols]
    ok = [r for r in rets if r is not None]
    if len(ok) < 2:
        return None
    return float(sum(ok) / len(ok))


def build_scorecard(
    journal: DecisionJournal,
    prices: pd.DataFrame,
    *,
    formula_top_n: int = 10,
    members: list[str] | None = None,
    asof_on: date | None = None,
) -> Scorecard:
    """只用日志里当时写下的判断，避免事后用今天的模型回打历史新闻。"""
    rows = [r for r in journal.read() if r.get("kind") == "judgment" and r.get("stance")]
    if asof_on is not None:
        rows = [r for r in rows if r.get("asof") == asof_on.isoformat()]
    notes = [
        "文本因子/研判只评估决策日志里的前瞻记录；不用今天的模型给历史新闻重打分（Sarkar & Vafa 2024；Lopez-Lira & Tang 2023）。"
    ]
    groups = {
        "研究跟进": [r for r in rows if r.get("stance") == "研究跟进"],
        "回避": [r for r in rows if r.get("stance") == "回避"],
        "公式排名": [
            r
            for r in rows
            if r.get("formula_rank") is not None and int(r["formula_rank"]) <= formula_top_n
        ],
    }
    card = Scorecard(asof_max=max((r.get("asof") or "") for r in rows) if rows else None, horizons=list(HORIZONS), notes=notes)
    if not rows:
        card.notes.append("决策日志里还没有研判记录。")
        return card
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    for gname, items in groups.items():
        card.groups[gname] = {}
        for h in HORIZONS:
            excesses: list[float] = []
            pending = 0
            seen: set[tuple[str, str]] = set()
            for row in items:
                key = (str(row.get("asof")), str(row.get("symbol")))
                if key in seen or not row.get("symbol") or not row.get("asof"):
                    continue
                seen.add(key)
                try:
                    start = date.fromisoformat(str(row["asof"])[:10])
                except ValueError:
                    continue
                r_i = _fwd_return(px, str(row["symbol"]), start, h)
                r_ew = _ew_return(px, start, h, members)
                if r_i is None or r_ew is None:
                    pending += 1
                    continue
                excesses.append(r_i - r_ew)
            card.groups[gname][h] = GroupHorizon(
                n=len(excesses),
                pending=pending,
                excess=None if not excesses else float(sum(excesses) / len(excesses)),
            )
    return card
