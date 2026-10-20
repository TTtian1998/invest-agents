"""资金流向：东财 push2his 主源，新浪备源。北向个股净买入已停披露。

这是数据层里唯一允许直接发 HTTP 的模块（akshare 没有稳定封装的日度资金流）。
所有东财请求串行限流：间隔 ≥1s + 0.1–0.5s 抖动，可由 EM_MIN_INTERVAL 覆盖。
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date

from zhiku.models import MissingData

_LOCK = threading.Lock()
_LAST_EM_CALL = 0.0
_UA = "zhiku-mvp/0.1 (personal research; not a crawler farm)"


@dataclass(frozen=True)
class FundFlowDay:
    """单日主力资金流（单位：元，未知则为 None）。"""

    date: date
    main_net: float | None
    small_net: float | None = None
    medium_net: float | None = None
    large_net: float | None = None
    super_net: float | None = None
    source: str = "eastmoney"


def _min_interval() -> float:
    try:
        return max(0.0, float(os.environ.get("EM_MIN_INTERVAL", "1.0")))
    except ValueError:
        return 1.0


def _throttle() -> None:
    global _LAST_EM_CALL
    wait = _min_interval() + random.uniform(0.1, 0.5)
    with _LOCK:
        now = time.monotonic()
        gap = now - _LAST_EM_CALL
        if gap < wait:
            time.sleep(wait - gap)
        _LAST_EM_CALL = time.monotonic()


def reset_throttle_for_tests() -> None:
    global _LAST_EM_CALL
    with _LOCK:
        _LAST_EM_CALL = 0.0


def _http_get(url: str, timeout: float = 15.0) -> str | MissingData:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return MissingData(f"HTTP 失败: {exc}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def _em_secid(symbol: str) -> str:
    code = symbol.split(".")[0]
    if code.startswith(("6", "5", "9")):
        return f"1.{code}"
    return f"0.{code}"


def _parse_em_kline(payload: str, symbol: str) -> list[FundFlowDay] | MissingData:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return MissingData(f"东财 JSON 无法解析: {exc}", symbol=symbol)
    klines = (
        (data.get("data") or {}).get("klines")
        if isinstance(data, dict)
        else None
    )
    if not klines:
        return MissingData("东财资金流为空", symbol=symbol)
    out: list[FundFlowDay] = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 6:
            continue
        try:
            d = date.fromisoformat(parts[0][:10])
            main = float(parts[5]) if parts[5] not in {"", "None"} else None
        except (TypeError, ValueError):
            continue
        out.append(
            FundFlowDay(
                date=d,
                main_net=main,
                small_net=_maybe_float(parts, 1),
                medium_net=_maybe_float(parts, 2),
                large_net=_maybe_float(parts, 3),
                super_net=_maybe_float(parts, 4),
                source="eastmoney",
            )
        )
    if not out:
        return MissingData("东财资金流为空", symbol=symbol)
    return out


def _maybe_float(parts: list[str], idx: int) -> float | None:
    if idx >= len(parts) or parts[idx] in {"", "None", "-"}:
        return None
    try:
        return float(parts[idx])
    except ValueError:
        return None


def fetch_eastmoney_daykline(symbol: str) -> list[FundFlowDay] | MissingData:
    """东财 ``fflow/daykline``。失败返回 MissingData，不估算。"""
    _throttle()
    secid = urllib.parse.quote(_em_secid(symbol))
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        f"?lmt=0&klt=101&secid={secid}"
        "&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
    )
    body = _http_get(url)
    if isinstance(body, MissingData):
        body.symbol = symbol
        return body
    return _parse_em_kline(body, symbol)


def _sina_code(symbol: str) -> str:
    code = symbol.split(".")[0]
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _parse_sina(payload: str, symbol: str) -> list[FundFlowDay] | MissingData:
    text = payload.strip()
    if not text:
        return MissingData("新浪资金流为空", symbol=symbol)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 新浪偶发返回无引号 key 的伪 JSON
        try:
            import ast

            data = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            return MissingData(f"新浪资金流无法解析: {exc}", symbol=symbol)
    if not isinstance(data, list) or not data:
        return MissingData("新浪资金流为空", symbol=symbol)
    out: list[FundFlowDay] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        raw_date = row.get("opendate") or row.get("date") or row.get("rq")
        raw_main = row.get("netamount") or row.get("r0_net") or row.get("main_net")
        if raw_date is None:
            continue
        try:
            d = date.fromisoformat(str(raw_date)[:10])
            main = float(raw_main) if raw_main not in {None, ""} else None
        except (TypeError, ValueError):
            continue
        out.append(FundFlowDay(date=d, main_net=main, source="sina"))
    if not out:
        return MissingData("新浪资金流为空", symbol=symbol)
    return out


def fetch_sina_moneyflow(symbol: str) -> list[FundFlowDay] | MissingData:
    """新浪 ``MoneyFlow.ssl_qsfx_zjlrqs`` 备源。"""
    daima = urllib.parse.quote(_sina_code(symbol))
    url = (
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
        f"/MoneyFlow.ssl_qsfx_zjlrqs?daima={daima}"
    )
    body = _http_get(url)
    if isinstance(body, MissingData):
        body.symbol = symbol
        return body
    return _parse_sina(body, symbol)


def get_fund_flow(symbol: str) -> list[FundFlowDay] | MissingData:
    """主源东财，失败则新浪。两次都失败则 MissingData。"""
    primary = fetch_eastmoney_daykline(symbol)
    if not isinstance(primary, MissingData):
        return primary
    backup = fetch_sina_moneyflow(symbol)
    if not isinstance(backup, MissingData):
        return backup
    return MissingData(
        f"东财失败（{primary.reason}）；新浪失败（{backup.reason}）",
        symbol=symbol,
    )


def northbound_stock_net_buy(symbol: str, on: date | None = None) -> MissingData:
    """北向个股净买入：交易所已停披露，写死缺数据，不估算。"""
    del on
    return MissingData("交易所已停披露", symbol=symbol, field="northbound_net_buy")
