"""从 CSV 读离线行情，供示例和离线测试使用。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from zhiku.data.base import asof_quote, infer_market
from zhiku.models import Bar, Market, MissingData, Quote


class FixtureProvider:
    name = "fixture"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._bars: list[Bar] | None = None

    def _load_bars(self) -> list[Bar] | MissingData:
        if self._bars is not None:
            return self._bars
        path = self.root / "bars.csv"
        if not path.is_file():
            return MissingData(f"找不到离线行情 {path}")
        df = pd.read_csv(path)
        required = {"symbol", "date", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            return MissingData(f"bars.csv 缺列: {', '.join(sorted(missing))}")
        bars: list[Bar] = []
        for _, row in df.iterrows():
            p1455 = row.get("price_1455")
            price_1455 = None
            if p1455 is not None and str(p1455).strip() not in {"", "nan", "None"}:
                try:
                    price_1455 = float(p1455)
                except (TypeError, ValueError):
                    price_1455 = None
            symbol = str(row["symbol"])
            mkt_raw = row.get("market")
            market = None
            if mkt_raw is not None and str(mkt_raw).strip() not in {"", "nan"}:
                market = Market(str(mkt_raw).strip())
            else:
                market = infer_market(symbol)
            bars.append(
                Bar(
                    symbol=symbol,
                    date=date.fromisoformat(str(row["date"])[:10]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]) if "volume" in df.columns else 0.0,
                    price_1455=price_1455,
                    market=market,
                )
            )
        bars.sort(key=lambda b: (b.symbol, b.date))
        self._bars = bars
        return bars

    def get_daily_bars(
        self, symbol: str, start: date, end: date, *, market: Market | None = None
    ) -> list[Bar] | MissingData:
        del market
        bars = self._load_bars()
        if isinstance(bars, MissingData):
            return bars
        picked = [b for b in bars if b.symbol == symbol and start <= b.date <= end]
        if not picked:
            return MissingData("无日线", symbol=symbol)
        return picked

    def get_price_1455(self, symbol: str, on: date) -> float | MissingData:
        bars = self._load_bars()
        if isinstance(bars, MissingData):
            return bars
        match = next((b for b in bars if b.symbol == symbol and b.date == on), None)
        if match is None:
            return MissingData("无当日日线", symbol=symbol, field="price_1455", on=on)
        if match.price_1455 is None:
            return MissingData("无 14:55 价", symbol=symbol, field="price_1455", on=on)
        return float(match.price_1455)

    def get_quote(
        self, symbol: str, on: date, *, market: Market | None = None
    ) -> Quote | MissingData:
        bars = self.get_daily_bars(symbol, on, on, market=market)
        if isinstance(bars, MissingData):
            return bars
        match = next((b for b in bars if b.date == on), None)
        if match is None:
            return MissingData("无当日日线", symbol=symbol, on=on)
        mkt = market or match.market or infer_market(symbol)
        note = f"美东交易日 {on.isoformat()}" if mkt is Market.US else ""
        source = "close (US)" if mkt is Market.US else "close"
        return asof_quote(symbol, on, match.close, source, mkt, note=note)

    def latest_quotes(self, symbols: list[str]) -> dict[str, Quote | MissingData]:
        bars = self._load_bars()
        out: dict[str, Quote | MissingData] = {}
        if isinstance(bars, MissingData):
            for s in symbols:
                out[s] = bars
            return out
        for symbol in symbols:
            series = [b for b in bars if b.symbol == symbol]
            if not series:
                out[symbol] = MissingData("无日线", symbol=symbol)
                continue
            last = series[-1]
            mkt = last.market or infer_market(symbol)
            note = f"美东交易日 {last.date.isoformat()}" if mkt is Market.US else ""
            source = "close (US)" if mkt is Market.US else "close"
            out[symbol] = asof_quote(symbol, last.date, last.close, source, mkt, note=note)
        return out

    def get_earnings_dates(self, symbol: str) -> list[date] | MissingData:
        path = self.root / "earnings.csv"
        if not path.is_file():
            return MissingData("fixture 无财报日期", symbol=symbol)
        df = pd.read_csv(path)
        days = [
            date.fromisoformat(str(row["date"])[:10])
            for _, row in df.iterrows()
            if str(row["symbol"]) == symbol
        ]
        if not days:
            return MissingData("fixture 无财报日期", symbol=symbol)
        return days

    def get_lhb(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("fixture 无龙虎榜", symbol=symbol, on=on)

    def get_margin(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("fixture 无两融", symbol=symbol, on=on)

    def get_unlock(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("fixture 无解禁", symbol=symbol, on=on)
