"""行情提供者与本地全市场库。"""

from __future__ import annotations

from zhiku.data.ashare import AShareProvider
from zhiku.data.base import LiveProvider, MarketDataProvider
from zhiku.data.fixture import FixtureProvider
from zhiku.data.global_ import GlobalProvider
from zhiku.data.store import MarketStore

__all__ = [
    "AShareProvider",
    "FixtureProvider",
    "GlobalProvider",
    "LiveProvider",
    "MarketDataProvider",
    "MarketStore",
]
