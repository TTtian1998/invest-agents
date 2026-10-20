"""Point-in-time 财务与回测：按公告日对齐，处理退市，禁止前视。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from zhiku.screens.fundamentals import Fundamentals


class PitRow(BaseModel):
    """一条财务快照。used 时必须 announced <= asof。"""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    period: date | None = None
    announced: date
    delisted_on: date | None = None
    payload: dict = Field(default_factory=dict)


def load_pit_rows(path: Path) -> list[PitRow]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("rows") if isinstance(raw, dict) else raw
    out: list[PitRow] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        payload = {k: v for k, v in row.items() if k not in {"period", "announced", "delisted_on"}}
        out.append(
            PitRow(
                symbol=str(row.get("symbol")),
                period=_as_date(row.get("period")),
                announced=_as_date(row.get("announced")) or date.min,
                delisted_on=_as_date(row.get("delisted_on")),
                payload=payload,
            )
        )
    return out


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


class LookAheadError(ValueError):
    """使用了公告日之后的财务，构成前视偏差。"""


def asof_snapshot(rows: list[PitRow], asof: date, *, strict: bool = True) -> dict[str, Fundamentals]:
    """每个标的取 announced<=asof 的最新一条。period 日不得替代公告日。"""
    chosen: dict[str, PitRow] = {}
    for row in rows:
        if row.announced > asof:
            continue
        if row.delisted_on is not None and row.delisted_on <= asof:
            continue
        prev = chosen.get(row.symbol)
        if prev is None or row.announced > prev.announced:
            chosen[row.symbol] = row
    out: dict[str, Fundamentals] = {}
    for sym, row in chosen.items():
        if strict and row.announced > asof:
            raise LookAheadError(f"{sym} announced {row.announced} > asof {asof}")
        data = dict(row.payload)
        data["symbol"] = sym
        out[sym] = Fundamentals.model_validate(data)
    return out


def assert_no_lookahead(rows: list[PitRow], asof: date, used: list[Fundamentals]) -> None:
    by_sym: dict[str, list[PitRow]] = {}
    for row in rows:
        by_sym.setdefault(row.symbol, []).append(row)
    for fund in used:
        candidates = by_sym.get(fund.symbol) or []
        leaked = [r for r in candidates if r.announced > asof]
        if leaked and fund.roe in {r.payload.get("roe") for r in leaked}:
            raise LookAheadError(f"{fund.symbol} 使用了未公告财务（公告日晚于 {asof}）")
        for r in candidates:
            if r.announced > asof and r.payload.get("roe") == fund.roe and fund.roe not in {
                x.payload.get("roe") for x in candidates if x.announced <= asof
            }:
                raise LookAheadError(f"{fund.symbol} look-ahead via period 对齐")


def drop_survivors(panel: pd.DataFrame, delist: dict[str, date], asof: date) -> pd.DataFrame:
    """退市股在退市日前仍保留在面板中（避免幸存者偏差）；退市日后价格置空。"""
    del asof  # 历史面板按各自退市日截断，不按截面日丢列
    out = panel.copy()
    for col, end in delist.items():
        if col not in out.columns:
            continue
        out.loc[out.index > pd.Timestamp(end), col] = pd.NA
    return out


def universe_with_delist(symbols: list[str], delist: dict[str, date], asof: date) -> list[str]:
    out = []
    for s in symbols:
        end = delist.get(s)
        if end is not None and end <= asof:
            continue
        out.append(s)
    return out


def load_delist_map(path: Path) -> dict[str, date]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("delisted") if isinstance(raw, dict) else raw
    out: dict[str, date] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "")
        d = _as_date(row.get("delisted_on") or row.get("on"))
        if sym and d:
            out[sym] = d
    return out
