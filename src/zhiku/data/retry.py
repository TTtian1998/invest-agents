"""同步用限速与重试。失败返回 MissingData，不抛到流水线。"""

from __future__ import annotations

import os
import random
import time
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
