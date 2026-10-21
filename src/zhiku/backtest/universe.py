"""时点股票池：优先历史指数成分，否则月末已上市且有数据的全 A。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from zhiku.data.codes import is_cn_etf
from zhiku.data.universe_live import INDEX_ALIAS
from zhiku.models import MissingData


@dataclass
class PitUniverse:
    kind: str
    label: str
    membership: dict[pd.Timestamp, list[str]]
    all_symbols: list[str]
    coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "months": len(self.membership),
            "symbols": len(self.all_symbols),
            "coverage": self.coverage,
            "notes": list(self.notes),
        }


def month_end_dates(asof: date, years: int = 3) -> list[date]:
    start = date(asof.year - years, asof.month, 1)
    idx = pd.date_range(start, asof, freq="ME")
    out = [d.date() for d in idx]
    if not out or out[-1] != date(asof.year, asof.month, asof.day):
        last = date(asof.year, asof.month, 1) - timedelta(days=1)
        if last >= start and last not in out:
            out.append(last)
        if asof not in out:
            out.append(asof)
    return sorted({d for d in out if d <= asof})


def members_on(membership: dict, dt) -> set[str]:
    if not membership:
        return set()
    ts = pd.Timestamp(dt)
    if ts in membership:
        return set(membership[ts])
    keyed = sorted(pd.Timestamp(k) for k in membership)
    eligible = [k for k in keyed if k <= ts]
    if not eligible:
        return set()
    return set(membership[eligible[-1]])


def _normalize_code(code: str | None) -> str | None:
    if not code:
        return None
    key = str(code).lower()
    if key in INDEX_ALIAS:
        return INDEX_ALIAS[key]
    return str(code)


def ensure_index_member_history(
    store,
    dates: list[date],
    *,
    index_code: str,
    fetcher=None,
) -> tuple[int, list[str]]:
    """补齐缺失月末的历史成分。已有快照不重拉。"""
    notes: list[str] = []
    if fetcher is None or not dates:
        return 0, notes
    missing = [d for d in dates if not store.index_symbols(index_code, d)]
    if not missing:
        return 0, notes
    from zhiku.data.fetchers import baostock_index_members_hist

    fetched = 0
    batch = baostock_index_members_hist([(index_code, d) for d in missing])
    for day in missing:
        payload = batch.get((index_code, day))
        if isinstance(payload, MissingData) or not payload:
            notes.append(f"缺历史成分 {index_code}@{day}: {getattr(payload, 'reason', '空')}")
            continue
        store.upsert_index_members(index_code, day, list(payload))
        fetched += 1
    if fetched:
        notes.append(f"写入历史成分 {index_code} {fetched}/{len(missing)} 个时点")
    return fetched, notes


def _alla_members(
    store,
    dt: date,
    *,
    first_bar: dict[str, date],
    listings: dict[str, dict],
    delist: dict[str, date],
) -> list[str]:
    start = dt - timedelta(days=10)
    have = store.symbols_with_bar_between(start, dt, adjust="qfq")
    out: list[str] = []
    for symbol in have:
        if is_cn_etf(symbol):
            continue
        info = listings.get(symbol) or {}
        if str(info.get("instrument") or "stock") == "etf":
            continue
        listed = info.get("list_date")
        if listed:
            try:
                if date.fromisoformat(str(listed)[:10]) > dt:
                    continue
            except ValueError:
                pass
        first = first_bar.get(symbol)
        if first is not None and first > dt:
            continue
        gone = delist.get(symbol)
        if gone is not None and gone <= dt:
            continue
        dl = info.get("delist_date")
        if dl:
            try:
                if date.fromisoformat(str(dl)[:10]) <= dt:
                    continue
            except ValueError:
                pass
        out.append(symbol)
    return sorted(out)


def build_pit_universe(
    store,
    *,
    asof: date,
    years: int = 3,
    universe: str = "csi800",
    fetcher=None,
    allow_fetch: bool = True,
    prices: pd.DataFrame | None = None,
) -> PitUniverse:
    """
    优先用历史中证成分（沪深300+中证500=中证800）。
    拿不到足够时点则改用月末已上市且有日线的全 A，并在 label 里标明。
    """
    dates = month_end_dates(asof, years)
    index_code = _normalize_code(universe)
    notes: list[str] = []
    if allow_fetch and index_code and fetcher is not None:
        _, fetch_notes = ensure_index_member_history(
            store, dates, index_code=index_code, fetcher=fetcher
        )
        notes.extend(fetch_notes)

    hist_hits = 0
    hist_map: dict[pd.Timestamp, list[str]] = {}
    coverage: dict[str, dict[str, int]] = {}
    if index_code:
        for dt in dates:
            members = store.index_symbols_asof(index_code, dt)
            if len(members) >= 50:
                hist_hits += 1
                ts = pd.Timestamp(dt)
                if prices is not None and not prices.empty and ts in prices.index:
                    have = [s for s in members if s in prices.columns and pd.notna(prices.loc[ts, s])]
                else:
                    start = dt - timedelta(days=10)
                    have_set = store.symbols_with_bar_between(start, dt, adjust="qfq")
                    have = [s for s in members if s in have_set]
                hist_map[ts] = have
                coverage[dt.isoformat()] = {"members": len(members), "with_bars": len(have)}

    # 至少一半时点有历史成分才采用，避免只用今天一份名单回填
    if hist_map and hist_hits >= max(2, len(dates) // 2):
        all_symbols = sorted({s for v in hist_map.values() for s in v})
        label = (
            f"时点{universe}成分（baostock 沪深300+中证500 回查，"
            f"{hist_hits}/{len(dates)} 个时点，样本 {len(all_symbols)}）"
        )
        notes.append(label)
        notes.append("调出样本若本地无日线则当月剔除，可能仍低估调出拖累。")
        return PitUniverse(
            kind="index_hist",
            label=label,
            membership=hist_map,
            all_symbols=all_symbols,
            coverage=coverage,
            notes=notes,
        )

    listings_df = store.listings(market="CN")
    listings = {} if listings_df.empty else {str(r["symbol"]): r.to_dict() for _, r in listings_df.iterrows()}
    first_bar = store.first_bar_dates(adjust="qfq")
    delist = store.delist_map()
    alla: dict[pd.Timestamp, list[str]] = {}
    for dt in dates:
        members = _alla_members(
            store, dt, first_bar=first_bar, listings=listings, delist=delist
        )
        if prices is not None and not prices.empty:
            ts = pd.Timestamp(dt)
            if ts in prices.index:
                members = [s for s in members if s in prices.columns and pd.notna(prices.loc[ts, s])]
        alla[pd.Timestamp(dt)] = members
        coverage[dt.isoformat()] = {"members": len(members), "with_bars": len(members)}
    all_symbols = sorted({s for v in alla.values() for s in v})
    label = (
        f"替代池：各月末当时已上市且有日线的全 A（未拿到足够的{universe}历史成分，"
        f"{len(dates)} 个月，样本 {len(all_symbols)}）"
    )
    notes.append(label)
    if hist_hits:
        notes.append(f"历史成分仅覆盖 {hist_hits}/{len(dates)} 个时点，改用全 A 替代池。")
    return PitUniverse(
        kind="alla_listed",
        label=label,
        membership=alla,
        all_symbols=all_symbols,
        coverage=coverage,
        notes=notes,
    )
