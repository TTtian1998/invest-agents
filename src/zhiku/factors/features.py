"""不依赖 pyqlib 的量价特征（Alpha 风格子集），供训练与预测。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ma_gap_5",
    "ma_gap_20",
    "ma_gap_60",
    "std_20",
    "std_60",
    "vol_ma_ratio",
    "hl_spread",
    "turnover",
    "mom_12_1",
]


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = a / b.replace(0, np.nan)
    return out


def make_feature_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """bars 需含 date, symbol, close, 可选 high/low/volume/turnover。返回 MultiIndex (date, symbol)。"""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index().astype(float)
    high = (
        df.pivot(index="date", columns="symbol", values="high").sort_index().astype(float)
        if "high" in df.columns
        else close
    )
    low = (
        df.pivot(index="date", columns="symbol", values="low").sort_index().astype(float)
        if "low" in df.columns
        else close
    )
    volume = (
        df.pivot(index="date", columns="symbol", values="volume").sort_index().astype(float)
        if "volume" in df.columns
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    turnover = (
        df.pivot(index="date", columns="symbol", values="turnover").sort_index().astype(float)
        if "turnover" in df.columns
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    feats: dict[str, pd.DataFrame] = {
        "ret_1": close.pct_change(1),
        "ret_5": close.pct_change(5),
        "ret_10": close.pct_change(10),
        "ret_20": close.pct_change(20),
        "ret_60": close.pct_change(60),
        "ma_gap_5": _safe_div(close, close.rolling(5).mean()) - 1,
        "ma_gap_20": _safe_div(close, close.rolling(20).mean()) - 1,
        "ma_gap_60": _safe_div(close, close.rolling(60).mean()) - 1,
        "std_20": close.pct_change().rolling(20).std(),
        "std_60": close.pct_change().rolling(60).std(),
        "vol_ma_ratio": _safe_div(volume, volume.rolling(20).mean()) - 1,
        "hl_spread": _safe_div(high - low, close),
        "turnover": turnover,
        "mom_12_1": _safe_div(close.shift(21), close.shift(252)) - 1,
    }
    stacked = []
    for name, panel in feats.items():
        s = panel.stack()
        s.name = name
        stacked.append(s)
    out = pd.concat(stacked, axis=1)
    out.index = out.index.set_names(["date", "symbol"])
    return out.reindex(columns=FEATURE_NAMES)


def label_forward(bars: pd.DataFrame, horizon: int = 5) -> pd.Series:
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index().astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    s = fwd.stack()
    s.name = "label"
    s.index = s.index.set_names(["date", "symbol"])
    return s


def asof_feature_row(bars: pd.DataFrame, asof: date) -> pd.DataFrame:
    feats = make_feature_panel(bars)
    if feats.empty:
        return feats
    idx = feats.index.get_level_values("date")
    target = pd.Timestamp(asof)
    available = idx[idx <= target]
    if len(available) == 0:
        return feats.iloc[0:0]
    last = available.max()
    return feats.xs(last, level="date")
