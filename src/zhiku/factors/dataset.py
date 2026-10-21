"""量化样本：标签、中性化、时点财报特征。公告日对齐，不用测试集调参。"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from zhiku.data.compute import pick_num, ratio
from zhiku.data.store import MarketStore
from zhiku.factors.features import FEATURE_NAMES, make_feature_panel

# 冻结分割。改日期等于换测试集，消融结束后不得再动。
TRAIN_END = date(2025, 4, 30)
VALID_START = date(2025, 5, 1)
VALID_END = date(2025, 10, 31)
TEST_START = date(2025, 11, 1)

FUND_FEATURE_NAMES = [
    "f_roe",
    "f_gm",
    "f_ocf_ni",
    "f_debt",
    "f_rev_yoy",
    "f_pe",
    "f_pb",
    "f_div_yield",
    "f_log_size",
]
FLOW_FEATURE_NAMES = [
    "flow_turnover",
    "flow_amount_cs",
    "flow_vol_spike",
]
# LSY (2019) 在中国解释不了的主要是反转和换手；波动是 CAPM 异象。符号已按「高分=看多」对齐。
CH_FEATURE_NAMES = [
    "ch_rev_5",
    "ch_rev_20",
    "ch_low_turn",
    "ch_low_vol",
]
CH_EP_NAME = "ch_ep"


def split_of(ts: pd.Timestamp) -> str:
    d = pd.Timestamp(ts).date()
    if d <= TRAIN_END:
        return "train"
    if VALID_START <= d <= VALID_END:
        return "valid"
    if d >= TEST_START:
        return "test"
    return "gap"


def assign_split(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).map(split_of)


def label_t2_close(bars: pd.DataFrame) -> pd.Series:
    """Alpha158 口径：T+1 收到 T+2 收，不用当日收盘当下一段收益起点。"""
    close = _pivot(bars, "close")
    fwd = close.shift(-2) / close.shift(-1) - 1.0
    return _stack(fwd, "label")


def label_open_horizon(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """T+1 开盘 → T+horizon 收盘。"""
    close = _pivot(bars, "close")
    entry = _pivot(bars, "open").shift(-1) if "open" in bars.columns else close.shift(-1)
    fwd = close.shift(-horizon) / entry - 1.0
    return _stack(fwd, "label")


def excess_vs_pool(label: pd.Series, membership: dict | None = None) -> pd.Series:
    """减去当日同池等权，得到相对超额。"""
    if label.empty:
        return label
    frame = label.rename("label").reset_index()
    if membership:
        from zhiku.backtest.universe import members_on

        keep = []
        for dt, part in frame.groupby("date"):
            allowed = members_on(membership, dt)
            if allowed:
                keep.append(part[part["symbol"].isin(allowed)])
            else:
                keep.append(part)
        frame = pd.concat(keep, ignore_index=True) if keep else frame
    frame["label"] = frame["label"] - frame.groupby("date")["label"].transform("mean")
    out = frame.set_index(["date", "symbol"])["label"]
    out.name = "label"
    return out


def neutralize(
    values: pd.Series,
    *,
    industry: pd.Series | None = None,
    size: pd.Series | None = None,
) -> pd.Series:
    """按日截面：先减行业均值，再对 log size 回归取残差。缺行业归入「未知」。"""
    if values.empty:
        return values
    frame = values.rename("y").to_frame()
    frame["industry"] = (industry.reindex(values.index) if industry is not None else "未知").fillna("未知")
    frame["size"] = size.reindex(values.index) if size is not None else np.nan
    parts = []
    for _, g in frame.groupby(level="date"):
        y = g["y"].astype(float)
        y = y - y.groupby(g["industry"]).transform("mean")
        sz = g["size"].astype(float)
        ok = y.notna() & sz.notna()
        if int(ok.sum()) >= 8 and float(sz[ok].std() or 0) > 1e-12:
            x = sz[ok].to_numpy()
            yy = y[ok].to_numpy()
            x = np.column_stack([np.ones(len(x)), x])
            try:
                beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
                y = y.copy()
                y.loc[ok] = yy - x @ beta
            except np.linalg.LinAlgError:
                pass
        parts.append(y)
    out = pd.concat(parts)
    out.name = values.name
    return out.reindex(values.index)


def make_labels(
    bars: pd.DataFrame,
    *,
    kind: str,
    membership: dict | None = None,
) -> pd.Series:
    if kind == "t2_close":
        raw = label_t2_close(bars)
    elif kind == "t20_open":
        raw = label_open_horizon(bars, 20)
    elif kind == "t20_open_xs":
        raw = excess_vs_pool(label_open_horizon(bars, 20), membership)
    elif kind == "t5_open":
        raw = label_open_horizon(bars, 5)
    else:
        raise ValueError(f"未知标签 {kind}")
    return raw.replace([np.inf, -np.inf], np.nan)


def make_ch_features(px: pd.DataFrame) -> pd.DataFrame:
    """短反转、低换手、低波动。列已取负，高分对应文献里的多头腿。"""
    out = pd.DataFrame(index=px.index)
    if "ret_5" in px.columns:
        out["ch_rev_5"] = -px["ret_5"].astype(float)
    if "ret_20" in px.columns:
        out["ch_rev_20"] = -px["ret_20"].astype(float)
    if "turnover" in px.columns:
        out["ch_low_turn"] = -px["turnover"].astype(float)
    if "std_20" in px.columns:
        out["ch_low_vol"] = -px["std_20"].astype(float)
    return out.reindex(columns=CH_FEATURE_NAMES)


def add_ch_ep(fund: pd.DataFrame) -> pd.DataFrame:
    if fund is None or fund.empty or "f_pe" not in fund.columns:
        return fund
    pe = fund["f_pe"].astype(float)
    fund = fund.copy()
    fund[CH_EP_NAME] = np.where((pe > 0) & pe.notna(), 1.0 / pe, np.nan)
    return fund


def last_cross_section(panel: pd.DataFrame, asof: date) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(panel.index.get_level_values("date"))
    ok = idx[idx <= pd.Timestamp(asof)]
    if len(ok) == 0:
        return panel.iloc[0:0].droplevel("date") if isinstance(panel.index, pd.MultiIndex) else panel.iloc[0:0]
    return panel.xs(ok.max(), level="date")


def asof_score_frame(store: MarketStore, symbols: list[str], asof: date, features: tuple[str, ...]) -> pd.DataFrame:
    """预测用：只取 asof 当日或之前最后一行，不读未来。"""
    from zhiku.factors.features import asof_feature_row

    bars = store.bars(symbols, end=asof, adjust="qfq")
    px = asof_feature_row(bars, asof)
    parts: list[pd.DataFrame] = []
    if "px" in features:
        parts.append(px)
    if "ch" in features:
        parts.append(make_ch_features(px))
    if "flow" in features:
        flow = last_cross_section(load_flow_features(store, symbols), asof)
        if not flow.empty:
            parts.append(flow)
    if "fund" in features:
        fund = add_ch_ep(load_fund_features(store, symbols, [asof]))
        if not fund.empty:
            parts.append(last_cross_section(fund, asof) if isinstance(fund.index, pd.MultiIndex) else fund)
    if not parts:
        return px
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def load_price_features(store: MarketStore, symbols: list[str] | None) -> pd.DataFrame:
    bars = store.bars(symbols, adjust="qfq")
    return make_feature_panel(bars)


def load_flow_features(store: MarketStore, symbols: list[str] | None) -> pd.DataFrame:
    bars = store.bars(symbols, adjust="qfq")
    if bars.empty:
        return pd.DataFrame(columns=FLOW_FEATURE_NAMES)
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = (
        df.pivot(index="date", columns="symbol", values="volume").sort_index()
        if "volume" in df.columns
        else close * np.nan
    )
    amount = (
        df.pivot(index="date", columns="symbol", values="amount").sort_index()
        if "amount" in df.columns
        else close * np.nan
    )
    turnover = (
        df.pivot(index="date", columns="symbol", values="turnover").sort_index()
        if "turnover" in df.columns
        else close * np.nan
    )
    vol_spike = volume / volume.rolling(20).mean().replace(0, np.nan) - 1.0
    amt_cs = amount.rank(axis=1, pct=True)
    stacked = {
        "flow_turnover": turnover.stack(),
        "flow_amount_cs": amt_cs.stack(),
        "flow_vol_spike": vol_spike.stack(),
    }
    out = pd.concat(stacked, axis=1)
    out.index = out.index.set_names(["date", "symbol"])
    return out.reindex(columns=FLOW_FEATURE_NAMES)


def load_fund_features(store: MarketStore, symbols: list[str], calendar: Iterable) -> pd.DataFrame:
    """公告日 ≤ 特征日的最新财报。PE/PB 只用当时已公告字段 + 未复权收盘，不拿当日 spot 回填历史。"""
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar))).sort_values().unique()
    if len(cal) == 0 or not symbols:
        return pd.DataFrame(columns=FUND_FEATURE_NAMES)
    stmts = store.statements_frame(symbols)
    divs = store.dividends_frame(symbols)
    none = store.bars(symbols, adjust="none")
    close_none = None
    if not none.empty:
        none = none.copy()
        none["date"] = pd.to_datetime(none["date"])
        close_none = none.pivot(index="date", columns="symbol", values="close").sort_index()

    snaps: list[dict[str, Any]] = []
    if not stmts.empty:
        work = stmts.dropna(subset=["announced"]).copy()
        work["announced"] = pd.to_datetime(work["announced"])
        work = work.sort_values(["symbol", "announced"])
        for symbol, part in work.groupby("symbol"):
            latest: dict[str, dict[str, Any]] = {}
            for announced, day in part.groupby("announced"):
                for _, row in day.iterrows():
                    try:
                        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict):
                        latest[str(row["kind"])] = payload
                income = latest.get("income") or {}
                balance = latest.get("balance") or {}
                cash = latest.get("cashflow") or {}
                ni = pick_num(income, "net_income")
                equity = pick_num(balance, "equity")
                rev = pick_num(income, "revenue")
                cogs = pick_num(income, "cogs")
                gm = (rev - cogs) / rev if rev not in (None, 0) and cogs is not None else None
                ocf = pick_num(cash, "ocf")
                snaps.append(
                    {
                        "symbol": str(symbol),
                        "announced": pd.Timestamp(announced),
                        "f_roe": ratio(ni, equity),
                        "f_gm": gm,
                        "f_ocf_ni": ratio(ocf, ni) if ni not in (None, 0) else None,
                        "f_debt": ratio(pick_num(balance, "total_liab"), pick_num(balance, "total_assets")),
                        "f_rev_yoy": _yoy(income),
                        "f_eps": pick_num(income, "eps"),
                        "f_bps": pick_num(income, "bps"),
                        "f_equity": equity,
                    }
                )
    snap_df = pd.DataFrame(snaps)
    rows = []
    for symbol in symbols:
        part = snap_df[snap_df["symbol"] == symbol].sort_values("announced") if not snap_df.empty else pd.DataFrame()
        base = pd.DataFrame({"date": cal})
        if not part.empty:
            merged = pd.merge_asof(
                base, part, left_on="date", right_on="announced", direction="backward"
            )
        else:
            merged = base
            for col in ("f_roe", "f_gm", "f_ocf_ni", "f_debt", "f_rev_yoy", "f_eps", "f_bps", "f_equity"):
                merged[col] = np.nan
        px = None
        if close_none is not None and symbol in close_none.columns:
            px = close_none[symbol].reindex(cal, method="ffill")
            merged["px"] = px.to_numpy()
        else:
            merged["px"] = np.nan
        pe = np.where(
            (merged["f_eps"].notna()) & (merged["f_eps"] != 0) & merged["px"].notna(),
            merged["px"] / merged["f_eps"],
            np.nan,
        )
        pb = np.where(
            (merged["f_bps"].notna()) & (merged["f_bps"] != 0) & merged["px"].notna(),
            merged["px"] / merged["f_bps"],
            np.nan,
        )
        size = np.where(
            merged["f_equity"].notna() & (merged["f_equity"] > 0),
            np.log(merged["f_equity"].astype(float)),
            np.nan,
        )
        yld = _div_yield_on(divs, symbol, cal, merged["px"])
        out = pd.DataFrame(
            {
                "date": cal,
                "symbol": symbol,
                "f_roe": merged.get("f_roe"),
                "f_gm": merged.get("f_gm"),
                "f_ocf_ni": merged.get("f_ocf_ni"),
                "f_debt": merged.get("f_debt"),
                "f_rev_yoy": merged.get("f_rev_yoy"),
                "f_pe": pe,
                "f_pb": pb,
                "f_div_yield": yld,
                "f_log_size": size,
            }
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=FUND_FEATURE_NAMES + [CH_EP_NAME])
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "symbol"]).reindex(columns=FUND_FEATURE_NAMES)
    return add_ch_ep(panel)


def _div_yield_on(divs: pd.DataFrame, symbol: str, cal: pd.DatetimeIndex, px: pd.Series) -> np.ndarray:
    if divs is None or divs.empty:
        return np.full(len(cal), np.nan)
    part = divs[divs["symbol"] == symbol].copy()
    if part.empty:
        return np.full(len(cal), np.nan)
    asof = part["announced"].fillna(part["ex_date"])
    part = part.assign(asof=pd.to_datetime(asof, errors="coerce")).dropna(subset=["asof"])
    out = []
    px_s = pd.Series(px.to_numpy() if hasattr(px, "to_numpy") else px, index=cal)
    for dt in cal:
        window = part[(part["asof"] <= dt) & (part["asof"] >= dt - pd.Timedelta(days=365))]
        cash = window["amount"].sum(min_count=1)
        price = px_s.get(dt)
        if cash is None or pd.isna(cash) or price is None or pd.isna(price) or float(price) == 0:
            out.append(np.nan)
        else:
            # 派息字段多为「每10股派」或每股；超过 2 当成每10股
            unit = float(cash) / 10.0 if float(cash) > 2 else float(cash)
            out.append(unit / float(price))
    return np.array(out, dtype=float)


def _yoy(income: dict[str, Any]) -> float | None:
    for key in ("营业总收入同比", "营业收入同比", "净利润同比"):
        if key in income:
            try:
                val = float(income[key])
            except (TypeError, ValueError):
                continue
            return val / 100.0 if abs(val) > 2 else val
    return None


def attach_industry_size(
    index: pd.MultiIndex,
    store: MarketStore,
    fund: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    symbols = [str(s) for s in index.get_level_values("symbol").unique()]
    ind_map = store.listing_industry(symbols)
    industry = pd.Series(
        [ind_map.get(str(s), "未知") or "未知" for s in index.get_level_values("symbol")],
        index=index,
        name="industry",
    )
    if fund is not None and "f_log_size" in fund.columns:
        size = fund["f_log_size"].reindex(index)
    else:
        size = pd.Series(np.nan, index=index)
    return industry, size


def _pivot(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    if bars is None or bars.empty or field not in bars.columns:
        return pd.DataFrame()
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="symbol", values=field).sort_index().astype(float)


def _stack(panel: pd.DataFrame, name: str) -> pd.Series:
    if panel is None or panel.empty:
        return pd.Series(dtype=float, name=name)
    s = panel.stack()
    s.name = name
    s.index = s.index.set_names(["date", "symbol"])
    return s
