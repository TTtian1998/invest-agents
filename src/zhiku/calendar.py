"""交易日历与时区。全部对外时间用 Asia/Shanghai；美股日期保留美东交易日。"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from zhiku.models import Market, MissingData

SHANGHAI = ZoneInfo("Asia/Shanghai")
HONG_KONG = ZoneInfo("Asia/Hong_Kong")
EASTERN = ZoneInfo("America/New_York")

CN_CLOSE = time(15, 0)
HK_CLOSE = time(16, 10)
US_CLOSE = time(16, 0)

logger = logging.getLogger("zhiku.calendar")

_CN_CALENDAR_CACHE: dict[tuple[int, bool], set[date]] = {}
_CALENDAR_NETWORK_ALLOWED: ContextVar[bool] = ContextVar("calendar_network_allowed", default=True)
_CN_CALENDAR_WARNED = False


def now_shanghai(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI)
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI)
    return now.astimezone(SHANGHAI)


def parse_now(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return now_shanghai(dt)


def session_close_time(market: Market) -> time:
    if market is Market.CN:
        return CN_CLOSE
    if market is Market.HK:
        return HK_CLOSE
    return US_CLOSE


def _weekdays_between(start: date, end: date) -> set[date]:
    out: set[date] = set()
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.add(cur)
        cur += timedelta(days=1)
    return out


@contextmanager
def calendar_network(*, enabled: bool) -> Iterator[None]:
    """Scope calendar networking without changing process environment variables.

    Live callers still need ``ZHIKU_LIVE_CALENDAR``. An offline parent scope
    cannot be re-enabled by a nested caller; the previous policy is restored
    even when the enclosed operation raises.
    """
    token = _CALENDAR_NETWORK_ALLOWED.set(_CALENDAR_NETWORK_ALLOWED.get() and enabled)
    try:
        yield
    finally:
        _CALENDAR_NETWORK_ALLOWED.reset(token)


def _want_live_calendar() -> bool:
    return _CALENDAR_NETWORK_ALLOWED.get() and os.environ.get("ZHIKU_LIVE_CALENDAR", "").lower() in {"1", "true", "yes"}


def load_cn_trade_dates(year: int) -> set[date]:
    """A 股交易日：live 模式走 akshare；失败或离线时退化为工作日并警告。"""
    global _CN_CALENDAR_WARNED
    want_live = _want_live_calendar()
    cache_key = (year, want_live)
    if cache_key in _CN_CALENDAR_CACHE:
        return _CN_CALENDAR_CACHE[cache_key]
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    if want_live:
        try:
            import akshare as ak

            df = ak.tool_trade_date_hist_sina()
            col = "trade_date" if "trade_date" in df.columns else df.columns[0]
            days: set[date] = set()
            for value in df[col]:
                text = str(value).replace("/", "-")[:10]
                d = (
                    date.fromisoformat(text)
                    if "-" in text
                    else datetime.strptime(text, "%Y%m%d").date()
                )
                if d.year == year:
                    days.add(d)
            if days:
                _CN_CALENDAR_CACHE[cache_key] = days
                return days
            raise RuntimeError("交易日历为空")
        except Exception as exc:  # noqa: BLE001 — 失败即退化，不让日历拖垮判定
            if not _CN_CALENDAR_WARNED:
                logger.warning("A 股交易日历不可用（%s），退化为工作日", exc)
                _CN_CALENDAR_WARNED = True
    days = _weekdays_between(start, end)
    _CN_CALENDAR_CACHE[cache_key] = days
    return days


def is_trading_day(market: Market, d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if market is Market.CN:
        return d in load_cn_trade_dates(d.year)
    return True


def previous_trading_day(market: Market, d: date) -> date:
    cur = d - timedelta(days=1)
    for _ in range(30):
        if is_trading_day(market, cur):
            return cur
        cur -= timedelta(days=1)
    return cur


def next_trading_day(market: Market, d: date) -> date:
    cur = d + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(market, cur):
            return cur
        cur += timedelta(days=1)
    return cur


def trading_days_on_or_before(market: Market, end: date, count: int) -> list[date]:
    """含 end（若为交易日）在内的最近 count 个交易日，从旧到新。"""
    out: list[date] = []
    cur = end
    guard = 0
    while len(out) < count and guard < 90:
        guard += 1
        if is_trading_day(market, cur):
            out.append(cur)
        cur -= timedelta(days=1)
    out.reverse()
    return out


def add_trading_days(market: Market, start: date, n: int) -> date | MissingData:
    """start 之后第 n 个交易日（n>=1 不含 start 当天）。"""
    if n <= 0:
        return start
    cur = start
    seen = 0
    for _ in range(n * 4 + 10):
        cur += timedelta(days=1)
        if is_trading_day(market, cur):
            seen += 1
            if seen == n:
                return cur
    return MissingData(f"无法推算 {start} 之后第 {n} 个交易日", on=start)


def session_complete(market: Market, on: date, now: datetime | None = None) -> bool:
    """该市场在 ``on`` 这一交易日是否已经收盘。"""
    clock = now_shanghai(now)
    if market is Market.US:
        asof_us = clock.astimezone(EASTERN)
        if on < asof_us.date():
            return True
        if on > asof_us.date():
            return False
        return asof_us.time() >= US_CLOSE
    if on < clock.date():
        return True
    if on > clock.date():
        return False
    return clock.time() >= session_close_time(market)


def last_completed_session(market: Market, now: datetime | None = None) -> date:
    clock = now_shanghai(now)
    if market is Market.US:
        asof_us = clock.astimezone(EASTERN)
        d = asof_us.date()
        if asof_us.time() < US_CLOSE:
            d = previous_trading_day(market, d)
        while not is_trading_day(market, d):
            d = previous_trading_day(market, d)
        return d
    d = clock.date()
    if clock.time() < session_close_time(market):
        d = previous_trading_day(market, d)
    while not is_trading_day(market, d):
        d = previous_trading_day(market, d)
    return d


def week_bounds(end: date) -> tuple[date, date]:
    """含 end 的周一到周五。"""
    monday = end - timedelta(days=end.weekday())
    friday = monday + timedelta(days=4)
    return monday, friday
