"""同步用限速与重试。失败返回 MissingData，不抛到流水线。"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable

from zhiku.models import MissingData

_LAST: dict[str, float] = {}


def sync_sleep_seconds(override: float | None = None) -> float:
    if override is not None:
        return max(0.0, float(override))
    try:
        return max(0.0, float(os.environ.get("ZHIKU_SYNC_SLEEP", "0.2")))
    except ValueError:
        return 0.2


def throttle(bucket: str = "sync", *, seconds: float | None = None) -> None:
    wait = sync_sleep_seconds(seconds) + random.uniform(0.0, 0.05)
    if wait <= 0:
        _LAST[bucket] = time.monotonic()
        return
    last = _LAST.get(bucket, 0.0)
    gap = time.monotonic() - last
    if gap < wait:
        time.sleep(wait - gap)
    _LAST[bucket] = time.monotonic()


def retry_call(
    fn: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    bucket: str = "sync",
    sleep: float | None = None,
    symbol: str | None = None,
    field: str | None = None,
    **kwargs: Any,
) -> Any | MissingData:
    last: Exception | None = None
    attempts = max(0, retries) + 1
    for i in range(attempts):
        throttle(bucket, seconds=sleep)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — 数据源失败记缺数据
            last = exc
            time.sleep(0.4 * (i + 1))
    return MissingData(f"重试失败: {last}", symbol=symbol, field=field)


def call_with_timeout(
    fn: Callable[..., Any],
    timeout: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """线程超时；超时不杀线程，只放弃等待。挂死源请用子进程。"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result(timeout=timeout)


def timed_call(
    fn: Callable[..., Any],
    timeout: float,
    *args: Any,
    symbol: str | None = None,
    field: str | None = None,
    **kwargs: Any,
) -> Any | MissingData:
    try:
        return call_with_timeout(fn, timeout, *args, **kwargs)
    except FuturesTimeout:
        return MissingData(f"{field or 'call'} 超时 {timeout:.0f}s", symbol=symbol, field=field)
