"""行情提供者。"""

from __future__ import annotations

from zhiku.data.ashare import AShareProvider
from zhiku.data.base import LiveProvider, MarketDataProvider
from zhiku.data.fixture import FixtureProvider
from zhiku.data.global_ import GlobalProvider

__all__ = [
    "AShareProvider",
    "FixtureProvider",
    "GlobalProvider",
    "LiveProvider",
    "MarketDataProvider",
]
