"""live 模式从本地库构建 PIT 股票池。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from zhiku.config import Paths
from zhiku.data.codes import infer_market, is_cn_etf
from zhiku.data.compute import (
    attach_percentiles,
    consecutive_div_years,
    missing_ratio,
    snapshot_from_store,
)
from zhiku.data.store import MarketStore
from zhiku.events.pit import universe_with_delist
from zhiku.models import Market
from zhiku.screens.fundamentals import Fundamentals

INDEX_ALIAS = {
    "all": None,
    "csi800": "000906",
    "csi300": "000300",
    "csi500": "000905",
    "csi1000": "000852",
    "star50": "000688",
}


def store_from_paths(paths: Paths) -> MarketStore:
    return MarketStore(paths.data_dir)


def resolve_universe_symbols(store: MarketStore, universe: str) -> list[str]:
    key = (universe or "csi800").lower()
    if key == "all":
        df = store.listings(market="CN")
        if df.empty:
            return []
        return [s for s in df["symbol"].tolist() if not is_cn_etf(str(s))]
    code = INDEX_ALIAS.get(key, key)
    if code:
        members = store.index_symbols(code)
        if members:
            return members
    df = store.listings()
    return [] if df.empty else list(df["symbol"].tolist())


def _listing_map(store: MarketStore) -> dict[str, dict[str, Any]]:
    df = store.listings()
    if df.empty:
        return {}
    return {str(r["symbol"]): r.to_dict() for _, r in df.iterrows()}


def _spot_map(store: MarketStore, asof: date) -> dict[str, dict[str, Any]]:
    df = store.latest_spot(asof)
    if df.empty:
        return {}
    return {str(r["symbol"]): r.to_dict() for _, r in df.iterrows()}


def _prev_payload(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(history) < 2:
        return None
    return history[1].get("payload")


def _shares_issued(history: list[dict[str, Any]]) -> bool | None:
    if len(history) < 2:
        return None
    from zhiku.data.compute import pick_num

    curr = pick_num(history[0].get("payload"), "shares")
    prev = pick_num(history[1].get("payload"), "shares")
    if curr is None or prev is None:
        return None
    return curr > prev * 1.005


def build_fundamentals(
    store: MarketStore,
    asof: date,
    *,
    universe: str = "csi800",
    extra: list[str] | None = None,
) -> tuple[list[Fundamentals], dict[str, date], list[str]]:
    notes: list[str] = []
    if store.is_empty():
        return [], {}, ["本地库为空，请先 zk data sync"]
    symbols = resolve_universe_symbols(store, universe)
    for s in extra or []:
        if s not in symbols:
            symbols.append(s)
    delist = store.delist_map()
    symbols = universe_with_delist(symbols, delist, asof)
    listings = _listing_map(store)
    spots = _spot_map(store, asof)
    start = asof - timedelta(days=400)
    bars = store.bars(symbols, start=start, end=asof, adjust="qfq")
    close_panel = (
        bars.pivot(index="date", columns="symbol", values="close") if not bars.empty else pd.DataFrame()
    )
    amt_panel = (
        bars.pivot(index="date", columns="symbol", values="amount") if not bars.empty else pd.DataFrame()
    )
    rows: list[Fundamentals] = []
    skipped = 0
    for symbol in symbols:
        listing = listings.get(symbol, {"symbol": symbol, "name": symbol, "instrument": "stock"})
        if listing.get("delist_date"):
            try:
                if date.fromisoformat(str(listing["delist_date"])[:10]) <= asof:
                    skipped += 1
                    continue
            except ValueError:
                pass
        stmts = store.statements_asof(symbol, asof)
        if "yjbb" in stmts:
            income = dict(stmts.get("income") or {"payload": {}})
            payload = dict(income.get("payload") or {})
            extra = stmts["yjbb"].get("payload") or {}
            for k, v in extra.items():
                if v is not None and v != "":
                    payload[k] = v
            income["payload"] = payload
            stmts["income"] = income
        income_h = store.statements_history(symbol, "income", asof)
        balance_h = store.statements_history(symbol, "balance", asof)
        closes = close_panel[symbol] if symbol in close_panel.columns else None
        amounts = amt_panel[symbol] if symbol in amt_panel.columns else None
        div_df = store.dividends_asof(symbol, asof)
        ex_dates: list[date] = []
        if not div_df.empty:
            for raw in div_df["ex_date"].tolist():
                try:
                    ex_dates.append(date.fromisoformat(str(raw)[:10]))
                except ValueError:
                    continue
        spot = spots.get(symbol)
        mktcap = None if not spot else spot.get("mktcap")
        # 股息率：最近一年分红金额 / 市值。缺一则 None。
        div_yield = None
        if mktcap and ex_dates:
            year_amt = 0.0
            have = False
            for _, drow in div_df.iterrows():
                try:
                    ex = date.fromisoformat(str(drow["ex_date"])[:10])
                except ValueError:
                    continue
                if asof.year - 1 <= ex.year <= asof.year and drow.get("amount") is not None:
                    year_amt += float(drow["amount"])
                    have = True
            if have and float(mktcap) > 0:
                # fhps 派息常为每股；若数值很小按每股*股本无法取得，保持缺数据
                if year_amt > 1:  # 更像合计金额
                    div_yield = year_amt / float(mktcap)
        market = infer_market(symbol)
        row = snapshot_from_store(
            symbol=symbol,
            asof=asof,
            listing=listing,
            statements=stmts,
            prev_income=_prev_payload(income_h),
            prev_balance=_prev_payload(balance_h),
            spot=spot,
            closes=closes,
            amounts=amounts,
            div_years=consecutive_div_years(ex_dates, asof),
            div_yield=div_yield,
            shares_issued=_shares_issued(balance_h),
            market=market if isinstance(market, Market) else Market.CN,
        )
        rows.append(row)
    rows = attach_percentiles(rows)
    miss = missing_ratio(rows)
    notes.append(
        f"live 股票池 {len(rows)} 只（universe={universe}）；"
        f"截面缺数据比例 {miss:.1%}；退市已剔除 {skipped}。"
        "PE/PB 分位为当日截面分位，缺历史估值序列不插值。"
    )
    return rows, delist, notes
