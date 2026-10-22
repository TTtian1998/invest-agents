"""共享测试夹具：内存行情、临时知识库。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from zhiku.models import Bar, Market, MissingData, Quote


class MemoryProvider:
    name = "memory"

    def __init__(self, bars: list[Bar] | None = None) -> None:
        self.bars = list(bars or [])
        self.quotes: dict[tuple[str, date], Quote | MissingData] = {}

    def get_daily_bars(
        self, symbol: str, start: date, end: date, *, market: Market | None = None
    ) -> list[Bar] | MissingData:
        del market
        picked = [b for b in self.bars if b.symbol == symbol and start <= b.date <= end]
        if not picked:
            return MissingData("无日线", symbol=symbol)
        return picked

    def get_price_1455(self, symbol: str, on: date) -> float | MissingData:
        match = next((b for b in self.bars if b.symbol == symbol and b.date == on), None)
        if match is None or match.price_1455 is None:
            return MissingData("无 14:55 价", symbol=symbol, field="price_1455", on=on)
        return float(match.price_1455)

    def get_quote(
        self, symbol: str, on: date, *, market: Market | None = None
    ) -> Quote | MissingData:
        key = (symbol, on)
        if key in self.quotes:
            return self.quotes[key]
        bars = self.get_daily_bars(symbol, on, on, market=market)
        if isinstance(bars, MissingData):
            return bars
        match = next((b for b in bars if b.date == on), None)
        if match is None:
            return MissingData("无当日日线", symbol=symbol, on=on)
        from zhiku.data.base import asof_quote

        return asof_quote(symbol, on, match.close, "close", market or match.market)

    def get_earnings_dates(self, symbol: str) -> list[date] | MissingData:
        return MissingData("测试无财报", symbol=symbol)

    def get_lhb(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("测试无龙虎榜", symbol=symbol, on=on)

    def get_margin(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("测试无两融", symbol=symbol, on=on)

    def get_unlock(self, symbol: str, on: date) -> object | MissingData:
        return MissingData("测试无解禁", symbol=symbol, on=on)


def bar(
    symbol: str,
    d: str,
    close: float,
    *,
    open_: float | None = None,
    low: float | None = None,
    high: float | None = None,
    volume: float = 1000.0,
    price_1455: float | None = None,
    market: Market = Market.CN,
) -> Bar:
    o = close if open_ is None else open_
    return Bar(
        symbol=symbol,
        date=date.fromisoformat(d),
        open=o,
        high=high if high is not None else max(o, close),
        low=low if low is not None else min(o, close),
        close=close,
        volume=volume,
        price_1455=price_1455,
        market=market,
    )


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def tmp_kb(tmp_path: Path, examples_dir: Path) -> Path:
    kb = tmp_path / "kb"
    (kb / "rules").mkdir(parents=True)
    (kb / "portfolio").mkdir()
    (kb / "journal").mkdir()
    (kb / "reports").mkdir()
    for src in (examples_dir / "kb" / "rules").glob("*.yaml"):
        (kb / "rules" / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    pos = (examples_dir / "kb" / "portfolio" / "positions.yaml").read_text(encoding="utf-8")
    (kb / "portfolio" / "positions.yaml").write_text(pos, encoding="utf-8")
    return kb


