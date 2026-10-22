"""行情提供者 Protocol 与 live 路由。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

from zhiku.models import Bar, Market, MissingData, Quote


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def get_daily_bars(
        self, symbol: str, start: date, end: date, *, market: Market | None = None
    ) -> list[Bar] | MissingData: ...

    def get_price_1455(self, symbol: str, on: date) -> float | MissingData: ...

    def get_quote(
        self, symbol: str, on: date, *, market: Market | None = None
    ) -> Quote | MissingData: ...

    def get_earnings_dates(self, symbol: str) -> list[date] | MissingData: ...

    def get_lhb(self, symbol: str, on: date) -> object | MissingData: ...

    def get_margin(self, symbol: str, on: date) -> object | MissingData: ...

    def get_unlock(self, symbol: str, on: date) -> object | MissingData: ...


def infer_market(symbol: str, market: Market | None = None) -> Market:
    if market is not None:
        return market
    upper = symbol.upper()
    if upper.endswith(".HK"):
        return Market.HK
    if upper.endswith(".US") or (upper.isalpha() and not upper[0].isdigit()):
        return Market.US
    return Market.CN


class LiveProvider:
    """按市场把请求分给 AShareProvider / GlobalProvider。"""

    name = "live"

    def __init__(self, cn: MarketDataProvider, glob: MarketDataProvider) -> None:
        self._cn = cn
        self._glob = glob

    def _route(self, symbol: str, market: Market | None) -> MarketDataProvider:
        mkt = infer_market(symbol, market)
        return self._cn if mkt is Market.CN else self._glob

    def get_daily_bars(
        self, symbol: str, start: date, end: date, *, market: Market | None = None
    ) -> list[Bar] | MissingData:
        return self._route(symbol, market).get_daily_bars(symbol, start, end, market=market)

    def get_price_1455(self, symbol: str, on: date) -> float | MissingData:
        return self._route(symbol, Market.CN).get_price_1455(symbol, on)

    def get_quote(
        self, symbol: str, on: date, *, market: Market | None = None
    ) -> Quote | MissingData:
        return self._route(symbol, market).get_quote(symbol, on, market=market)

    def get_earnings_dates(self, symbol: str) -> list[date] | MissingData:
        return self._route(symbol, None).get_earnings_dates(symbol)

    def get_lhb(self, symbol: str, on: date) -> object | MissingData:
        return self._route(symbol, Market.CN).get_lhb(symbol, on)

    def get_margin(self, symbol: str, on: date) -> object | MissingData:
        return self._route(symbol, Market.CN).get_margin(symbol, on)

    def get_unlock(self, symbol: str, on: date) -> object | MissingData:
        return self._route(symbol, Market.CN).get_unlock(symbol, on)


def asof_quote(
    symbol: str,
    on: date,
    price: float,
    source: str,
    market: Market | None = None,
    note: str = "",
    at: datetime | None = None,
) -> Quote:
    from zhiku.calendar import SHANGHAI, now_shanghai

    when = at if at is not None else datetime(on.year, on.month, on.day, 15, 0, tzinfo=SHANGHAI)
    return Quote(
        symbol=symbol,
        asof=now_shanghai(when),
        price=price,
        source=source,
        market=market,
        note=note,
    )
