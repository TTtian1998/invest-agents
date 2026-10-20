"""A 股 / ETF 行情：复用 akshare，不另写爬虫。"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Callable

from zhiku.calendar import SHANGHAI
from zhiku.data.base import asof_quote
from zhiku.data.cache import SQLiteCache
from zhiku.models import Bar, Market, MissingData, Quote


def _call_first_available(
    names: list[str],
    *args: Any,
    module: Any | None = None,
    **kwargs: Any,
) -> Any | MissingData:
    """按顺序尝试 akshare 函数名，兼容不同版本。单测可传入假 module。"""
    if module is None:
        import akshare as ak

        module = ak
    errors: list[str] = []
    for name in names:
        fn = getattr(module, name, None)
        if fn is None:
            errors.append(f"{name}: 不存在")
            continue
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    return MissingData("调用失败: " + "; ".join(errors))


def is_cn_etf(symbol: str) -> bool:
    code = symbol.split(".")[0]
    return len(code) == 6 and code[0] in {"5", "1"}


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).replace("/", "-")[:10]
    if "-" in text:
        return date.fromisoformat(text)
    return datetime.strptime(str(value)[:8], "%Y%m%d").date()


def _col(df: Any, *candidates: str) -> str | None:
    cols = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _bars_from_df(symbol: str, df: Any) -> list[Bar] | MissingData:
    if df is None or getattr(df, "empty", True):
        return MissingData("日线为空", symbol=symbol)
    date_col = _col(df, "日期", "date", "day")
    open_col = _col(df, "开盘", "open")
    high_col = _col(df, "最高", "high")
    low_col = _col(df, "最低", "low")
    close_col = _col(df, "收盘", "close")
    vol_col = _col(df, "成交量", "volume")
    if not all([date_col, open_col, high_col, low_col, close_col]):
        return MissingData("日线缺列", symbol=symbol)
    bars: list[Bar] = []
    for _, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol,
                date=_to_date(row[date_col]),
                open=float(row[open_col]),
                high=float(row[high_col]),
                low=float(row[low_col]),
                close=float(row[close_col]),
                volume=float(row[vol_col] or 0) if vol_col else 0.0,
                market=Market.CN,
            )
        )
    bars.sort(key=lambda b: b.date)
    return bars


def _extract_1455(df: Any, symbol: str, on: date) -> float | MissingData:
    if df is None or getattr(df, "empty", True):
        return MissingData("无 1 分钟线", symbol=symbol, field="price_1455", on=on)
    time_col = _col(df, "时间", "datetime", "day", "日期")
    close_col = _col(df, "收盘", "close")
    if time_col is None or close_col is None:
        return MissingData("分钟线缺列", symbol=symbol, field="price_1455", on=on)
    for _, row in df.iterrows():
        stamp = str(row[time_col])
        if "14:55" in stamp:
            try:
                return float(row[close_col])
            except (TypeError, ValueError):
                return MissingData("14:55 价无法解析", symbol=symbol, field="price_1455", on=on)
    return MissingData("1 分钟线中没有 14:55 那根", symbol=symbol, field="price_1455", on=on)


class AShareProvider:
    name = "ashare"

    def __init__(
        self,
        cache: SQLiteCache | None = None,
        caller: Callable[..., Any] | None = None,
    ) -> None:
        self.cache = cache
        self._caller = caller or _call_first_available

    def get_daily_bars(
        self, symbol: str, start: date, end: date, *, market: Market | None = None
    ) -> list[Bar] | MissingData:
        del market
        code = symbol.split(".")[0]
        if is_cn_etf(code):
            raw = self._caller(
                ["fund_etf_hist_em"],
                symbol=code,
                period="daily",
                start_date=_yyyymmdd(start),
                end_date=_yyyymmdd(end),
                adjust="qfq",
            )
        else:
            raw = self._caller(
                ["stock_zh_a_hist"],
                symbol=code,
                period="daily",
                start_date=_yyyymmdd(start),
                end_date=_yyyymmdd(end),
                adjust="qfq",
            )
        if isinstance(raw, MissingData):
            return raw
        return _bars_from_df(symbol, raw)

    def get_price_1455(self, symbol: str, on: date) -> float | MissingData:
        code = symbol.split(".")[0]
        start = datetime.combine(on, time(14, 50), tzinfo=SHANGHAI).strftime("%Y-%m-%d %H:%M:%S")
        end = datetime.combine(on, time(15, 0), tzinfo=SHANGHAI).strftime("%Y-%m-%d %H:%M:%S")
        names = (
            ["fund_etf_hist_min_em"]
            if is_cn_etf(code)
            else ["stock_zh_a_hist_min_em"]
        )
        raw = self._caller(
            names,
            symbol=code,
            start_date=start,
            end_date=end,
            period="1",
            adjust="",
        )
        if isinstance(raw, MissingData):
            return raw
        return _extract_1455(raw, symbol, on)

    def get_quote(
        self, symbol: str, on: date, *, market: Market | None = None
    ) -> Quote | MissingData:
        del market
        bars = self.get_daily_bars(symbol, on, on)
        if isinstance(bars, MissingData):
            return bars
        match = next((b for b in bars if b.date == on), None)
        if match is None:
            return MissingData("无当日日线", symbol=symbol, on=on)
        return asof_quote(symbol, on, match.close, "close", Market.CN)

    def get_earnings_dates(self, symbol: str) -> list[date] | MissingData:
        return MissingData("A 股财报日期请用公告，不走 yfinance", symbol=symbol)

    def get_lhb(self, symbol: str, on: date) -> object | MissingData:
        code = symbol.split(".")[0]
        raw = self._caller(
            ["stock_lhb_detail_em", "stock_lhb_stock_detail_em", "stock_lhb_hyyyb_em"],
            symbol=code,
            start_date=_yyyymmdd(on),
            end_date=_yyyymmdd(on),
        )
        if isinstance(raw, MissingData):
            return raw
        return raw

    def get_margin(self, symbol: str, on: date) -> object | MissingData:
        code = symbol.split(".")[0]
        raw = self._caller(
            [
                "stock_margin_detail_sse",
                "stock_margin_detail_szse",
                "stock_margin_underlying_info_szse",
            ],
            date=_yyyymmdd(on),
            symbol=code,
        )
        if isinstance(raw, MissingData):
            return raw
        return raw

    def get_unlock(self, symbol: str, on: date) -> object | MissingData:
        code = symbol.split(".")[0]
        raw = self._caller(
            [
                "stock_restricted_release_detail_em",
                "stock_restricted_release_queue_em",
                "stock_restricted_release_stockholder_em",
            ],
            symbol=code,
            start_date=_yyyymmdd(on),
            end_date=_yyyymmdd(on),
        )
        if isinstance(raw, MissingData):
            return raw
        return raw
