"""港股 / 美股：复用 yfinance。港股代码形如 0700.HK、2800.HK。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from zhiku.calendar import EASTERN, SHANGHAI
from zhiku.data.base import asof_quote, infer_market
from zhiku.models import Bar, Market, MissingData, Quote


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = getattr(value, "to_pydatetime", None)
    if callable(ts):
        return ts().date()
    return date.fromisoformat(str(value)[:10])


class GlobalProvider:
    name = "yfinance"

    def get_daily_bars(
        self, symbol: str, start: date, end: date, *, market: Market | None = None
    ) -> list[Bar] | MissingData:
        mkt = infer_market(symbol, market)
        try:
            import yfinance as yf
        except Exception as exc:  # noqa: BLE001
            return MissingData(f"yfinance 不可用: {exc}", symbol=symbol)
        try:
            hist = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001
            return MissingData(f"yfinance 拉取失败: {exc}", symbol=symbol)
        if hist is None or hist.empty:
            return MissingData("日线为空", symbol=symbol)
        bars: list[Bar] = []
        for idx, row in hist.iterrows():
            d = _to_date(idx)
            if d < start or d > end:
                continue
            try:
                bars.append(
                    Bar(
                        symbol=symbol,
                        date=d,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row.get("Volume") or 0),
                        market=mkt,
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                return MissingData(f"日线解析失败: {exc}", symbol=symbol, on=d)
        if not bars:
            return MissingData("日线为空", symbol=symbol)
        return bars

    def get_price_1455(self, symbol: str, on: date) -> float | MissingData:
        return MissingData("price_1455 仅适用于 CN", symbol=symbol, field="price_1455", on=on)

    def get_quote(
        self, symbol: str, on: date, *, market: Market | None = None
    ) -> Quote | MissingData:
        mkt = infer_market(symbol, market)
        bars = self.get_daily_bars(symbol, on, on, market=mkt)
        if isinstance(bars, MissingData):
            return bars
        match = next((b for b in bars if b.date == on), None)
        if match is None:
            return MissingData("无当日日线", symbol=symbol, on=on)
        note = ""
        tz = SHANGHAI
        source = "close"
        if mkt is Market.US:
            note = f"美东交易日 {on.isoformat()}"
            tz = EASTERN
            source = "close (US)"
        asof = datetime(on.year, on.month, on.day, 16, 0, tzinfo=tz)
        return asof_quote(symbol, on, match.close, source, mkt, note=note, at=asof)

    def get_earnings_dates(self, symbol: str) -> list[date] | MissingData:
        try:
            import yfinance as yf
        except Exception as exc:  # noqa: BLE001
            return MissingData(f"yfinance 不可用: {exc}", symbol=symbol)
        try:
            raw = yf.Ticker(symbol).get_earnings_dates()
        except Exception as exc:  # noqa: BLE001
            return MissingData(f"财报日期不可用: {exc}", symbol=symbol)
        if raw is None or getattr(raw, "empty", True):
            return MissingData("无财报日期", symbol=symbol)
        days: list[date] = []
        index = getattr(raw, "index", raw)
        for value in index:
            try:
                days.append(_to_date(value))
            except Exception:  # noqa: BLE001
                continue
        if not days:
            return MissingData("无财报日期", symbol=symbol)
        return days

    def get_lhb(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("龙虎榜仅适用于 CN", symbol=symbol, on=on)

    def get_margin(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("两融仅适用于 CN", symbol=symbol, on=on)

    def get_unlock(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("解禁仅适用于 CN", symbol=symbol, on=on)
