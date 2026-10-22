"""新闻源共用：限流、重试、缓存。单源失败不抛到流水线。"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import date
from typing import Any

from zhiku.data.cache import SQLiteCache
from zhiku.models import MissingData

_UA = "zhiku-news/0.2 (personal research; polite client)"
_LAST: dict[str, float] = {}


def _interval() -> float:
    try:
        return max(0.2, float(os.environ.get("ZHIKU_NEWS_MIN_INTERVAL", "0.8")))
    except ValueError:
        return 0.8


def throttle(bucket: str = "default") -> None:
    wait = _interval() + random.uniform(0.05, 0.25)
    last = _LAST.get(bucket, 0.0)
    gap = time.monotonic() - last
    if gap < wait:
        time.sleep(wait - gap)
    _LAST[bucket] = time.monotonic()


def http_get(
    url: str,
    *,
    timeout: float = 12.0,
    retries: int = 2,
    headers: dict[str, str] | None = None,
) -> str | MissingData:
    hdrs = {"User-Agent": _UA}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("gbk", errors="replace")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
    return MissingData(f"HTTP 失败: {last}")


def cached_json(
    cache: SQLiteCache | None,
    source: str,
    key: str,
    on: date,
    fetcher,
    *,
    min_interval: float | None = None,
) -> Any | MissingData:
    if cache is not None:
        hit = cache.get(source, key, on, "news")
        if hit is not None:
            return hit
    if min_interval is None:
        throttle(source)
    elif min_interval > 0:
        time.sleep(min_interval)
    body = fetcher()
    if isinstance(body, MissingData):
        return body
    try:
        data = json.loads(body) if isinstance(body, str) else body
    except json.JSONDecodeError as exc:
        return MissingData(f"JSON 无法解析: {exc}")
    if cache is not None:
        cache.set(source, key, on, "news", data)
    return data
