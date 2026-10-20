"""akshare 为主、yfinance 补充；baostock / tushare 有 token 时作备用。"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Callable

import pandas as pd

from zhiku.data.ashare import _call_first_available, _col, _to_date
from zhiku.data.codes import (
    em_code,
    hk_hist_code,
    infer_market,
    is_cn_etf,
    limit_threshold,
    normalize_cn,
)
from zhiku.data.retry import retry_call
from zhiku.models import Market, MissingData

INDEX_CSI = {
    "csi300": "000300",
    "csi500": "000905",
    "csi1000": "000852",
    "star50": "000688",
    "csi800": "000906",
}


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() in {"", "-", "None", "nan"}:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in {float("inf"), float("-inf")}:  # NaN / inf
        return None
    return f


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return _to_date(value)
    except Exception:  # noqa: BLE001
        text = str(value).strip()[:10].replace("/", "-")
        if len(text) >= 8 and text[4] == "-":
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None
        digits = "".join(ch for ch in str(value) if ch.isdigit())[:8]
        if len(digits) == 8:
            try:
                return date.fromisoformat(f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}")
            except ValueError:
                return None
        return None


class MarketFetcher:
    """可注入的抓取接口，测试用假实现。"""

    def cn_listings(self) -> list[dict] | MissingData:
        raise NotImplementedError

    def cn_delisted(self) -> list[dict] | MissingData:
        raise NotImplementedError

    def cn_suspended(self, on: date) -> list[str] | MissingData:
        raise NotImplementedError

    def index_members(self, index_code: str) -> list[str] | MissingData:
        raise NotImplementedError

    def cn_spot(self) -> list[dict] | MissingData:
        raise NotImplementedError

    def cn_bars(self, symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame | MissingData:
        raise NotImplementedError

    def cn_statement(self, symbol: str, kind: str) -> list[dict] | MissingData:
        raise NotImplementedError

    def cn_dividends(self, symbol: str) -> list[dict] | MissingData:
        raise NotImplementedError

    def global_members(self, name: str) -> list[str] | MissingData:
        raise NotImplementedError

    def global_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame | MissingData:
        raise NotImplementedError

    def global_financials(self, symbol: str) -> dict[str, list[dict]] | MissingData:
        raise NotImplementedError


class AkshareFetcher(MarketFetcher):
    def __init__(
        self,
        *,
        caller: Callable[..., Any] | None = None,
        sleep: float | None = None,
        retries: int = 3,
    ) -> None:
        self._caller = caller or _call_first_available
        self.sleep = sleep
        self.retries = retries

    def _ak(
        self,
        names: list[str],
        *args: Any,
        md_symbol: str | None = None,
        field: str | None = None,
        **kwargs: Any,
    ):
        def _run():
            raw = self._caller(names, *args, **kwargs)
            if isinstance(raw, MissingData):
                raise RuntimeError(raw.reason)
            return raw

        return retry_call(
            _run,
            retries=self.retries,
            bucket="akshare",
            sleep=self.sleep,
            symbol=md_symbol or kwargs.get("symbol"),
            field=field,
        )

    def cn_listings(self) -> list[dict] | MissingData:
        raw = self._ak(["stock_info_a_code_name"], field="listings")
        if isinstance(raw, MissingData):
            return raw
        df = raw
        code_col = _col(df, "code", "代码")
        name_col = _col(df, "name", "名称")
        if code_col is None:
            return MissingData("A 股列表缺代码列")
        out: list[dict] = []
        for _, row in df.iterrows():
            code = normalize_cn(str(row[code_col]))
            name = str(row[name_col]) if name_col else ""
            out.append(
                {
                    "symbol": code,
                    "market": Market.CN.value,
                    "name": name,
                    "st": "ST" in name.upper(),
                    "instrument": "etf" if is_cn_etf(code) else "stock",
                }
            )
        extra = self._listing_dates()
        if not isinstance(extra, MissingData):
            by = {r["symbol"]: r for r in extra}
            for row in out:
                hit = by.get(row["symbol"])
                if hit:
                    row["list_date"] = hit.get("list_date")
                    row["industry"] = hit.get("industry") or row.get("industry")
        return out

    def _listing_dates(self) -> list[dict] | MissingData:
        rows: list[dict] = []
        for names, kwargs in (
            (["stock_info_sh_name_code"], {"symbol": "主板A股"}),
            (["stock_info_sz_name_code"], {"symbol": "A股列表"}),
            (["stock_info_bj_name_code"], {}),
        ):
            raw = self._ak(names, field="list_date", **kwargs)
            if isinstance(raw, MissingData):
                continue
            df = raw
            code_col = _col(df, "证券代码", "A股代码", "code", "代码")
            date_col = _col(df, "上市日期", "list_date")
            ind_col = _col(df, "所属行业", "行业", "industry")
            if code_col is None:
                continue
            for _, row in df.iterrows():
                code = normalize_cn(str(row[code_col]))
                rows.append(
                    {
                        "symbol": code,
                        "list_date": _as_date(row[date_col]) if date_col else None,
                        "industry": str(row[ind_col]) if ind_col else None,
                    }
                )
        return rows or MissingData("缺上市日期表")

    def cn_delisted(self) -> list[dict] | MissingData:
        out: list[dict] = []
        for names in (
            ["stock_info_sh_delist"],
            ["stock_info_sz_delist"],
            ["stock_zh_a_stop_em"],
        ):
            raw = self._ak(names, field="delist")
            if isinstance(raw, MissingData):
                continue
            df = raw
            code_col = _col(df, "公司代码", "证券代码", "代码", "code")
            date_col = _col(df, "终止上市日期", "退市日期", "delist_date", "日期")
            name_col = _col(df, "公司名称", "证券简称", "名称", "name")
            if code_col is None:
                continue
            for _, row in df.iterrows():
                code = normalize_cn(str(row[code_col]))
                out.append(
                    {
                        "symbol": code,
                        "delist_date": _as_date(row[date_col]) if date_col else None,
                        "name": str(row[name_col]) if name_col else "",
                    }
                )
        return out or MissingData("缺退市列表")

    def cn_suspended(self, on: date) -> list[str] | MissingData:
        raw = self._ak(["stock_tfp_em"], date=_yyyymmdd(on), field="suspended")
        if isinstance(raw, MissingData):
            return raw
        df = raw
        code_col = _col(df, "代码", "code")
        if code_col is None:
            return MissingData("停牌表缺代码列")
        return [normalize_cn(str(v)) for v in df[code_col].tolist()]

    def index_members(self, index_code: str) -> list[str] | MissingData:
        code = INDEX_CSI.get(index_code, index_code)
        raw = self._ak(
            ["index_stock_cons_csindex", "index_stock_cons", "index_stock_cons_sina"],
            symbol=code,
            field="index",
        )
        if isinstance(raw, MissingData):
            return raw
        df = raw
        code_col = _col(df, "成分券代码", "品种代码", "code", "代码", "constituentCode")
        if code_col is None:
            return MissingData(f"指数 {code} 缺成分代码列")
        return [normalize_cn(str(v)) for v in df[code_col].tolist()]

    def cn_spot(self) -> list[dict] | MissingData:
        raw = self._ak(["stock_zh_a_spot_em", "stock_zh_a_spot"], field="spot")
        if isinstance(raw, MissingData):
            return raw
        df = raw
        code_col = _col(df, "代码", "code")
        if code_col is None:
            return MissingData("行情截面缺代码列")
        name_col = _col(df, "名称", "name")
        pe_col = _col(df, "市盈率-动态", "市盈率", "pe")
        pb_col = _col(df, "市净率", "pb")
        cap_col = _col(df, "总市值", "mktcap")
        amt_col = _col(df, "成交额", "amount")
        out = []
        for _, row in df.iterrows():
            code = normalize_cn(str(row[code_col]))
            name = str(row[name_col]) if name_col else ""
            out.append(
                {
                    "symbol": code,
                    "name": name,
                    "pe": _num(row[pe_col]) if pe_col else None,
                    "pb": _num(row[pb_col]) if pb_col else None,
                    "mktcap": _num(row[cap_col]) if cap_col else None,
                    "amount": _num(row[amt_col]) if amt_col else None,
                    "st": "ST" in name.upper(),
                }
            )
        return out

    def cn_bars(self, symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame | MissingData:
        code = normalize_cn(symbol)
        adj = "" if adjust in {"none", "raw", ""} else "qfq"
        names = ["fund_etf_hist_em"] if is_cn_etf(code) else ["stock_zh_a_hist"]
        raw = self._ak(
            names,
            symbol=code,
            period="daily",
            start_date=_yyyymmdd(start),
            end_date=_yyyymmdd(end),
            adjust=adj,
            md_symbol=code,
            field="bars",
        )
        if isinstance(raw, MissingData):
            return raw
        return _normalize_cn_bars(code, raw, adjust="none" if adj == "" else "qfq")

    def cn_statement(self, symbol: str, kind: str) -> list[dict] | MissingData:
        code = em_code(symbol)
        names = {
            "income": ["stock_profit_sheet_by_report_em"],
            "balance": ["stock_balance_sheet_by_report_em"],
            "cashflow": ["stock_cash_flow_sheet_by_report_em"],
        }.get(kind)
        if not names:
            return MissingData(f"未知报表 {kind}", symbol=symbol)
        raw = self._ak(names, symbol=code, md_symbol=normalize_cn(symbol), field=kind)
        if isinstance(raw, MissingData):
            return raw
        return _statements_from_df(normalize_cn(symbol), kind, raw)

    def cn_dividends(self, symbol: str) -> list[dict] | MissingData:
        code = normalize_cn(symbol)
        raw = self._ak(
            ["stock_fhps_em", "stock_history_dividend_detail"],
            symbol=code,
            md_symbol=code,
            field="dividend",
        )
        if isinstance(raw, MissingData):
            return raw
        df = raw
        ex_col = _col(df, "除权除息日", "除息日", "ex_date")
        ann_col = _col(df, "公告日期", "announced")
        amt_col = _col(df, "派息", "现金分红", "分红金额", "amount")
        if ex_col is None:
            return MissingData("分红表缺除权日", symbol=symbol)
        rows = []
        for _, row in df.iterrows():
            ex = _as_date(row[ex_col])
            if ex is None:
                continue
            rows.append(
                {
                    "symbol": code,
                    "ex_date": ex,
                    "announced": _as_date(row[ann_col]) if ann_col else ex,
                    "amount": _num(row[amt_col]) if amt_col else None,
                }
            )
        return rows

    def global_members(self, name: str) -> list[str] | MissingData:
        key = name.lower()
        if key in {"hsi", "hstech", "hscei"}:
            return _hk_index_members(self, key)
        if key in {"sp500", "ndx", "nasdaq100"}:
            return _us_index_members(key)
        return MissingData(f"未知全球名单 {name}")

    def global_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame | MissingData:
        market = infer_market(symbol)
        if market is Market.HK:
            raw = self._ak(
                ["stock_hk_hist"],
                symbol=hk_hist_code(symbol),
                period="daily",
                start_date=_yyyymmdd(start),
                end_date=_yyyymmdd(end),
                adjust="qfq",
                md_symbol=symbol,
                field="bars",
            )
            if not isinstance(raw, MissingData):
                return _normalize_cn_bars(symbol, raw, adjust="qfq", market=Market.HK)
        return _yfinance_bars(symbol, start, end)

    def global_financials(self, symbol: str) -> dict[str, list[dict]] | MissingData:
        return _yfinance_financials(symbol)


def _normalize_cn_bars(
    symbol: str, df: Any, *, adjust: str, market: Market = Market.CN
) -> pd.DataFrame | MissingData:
    if df is None or getattr(df, "empty", True):
        return MissingData("日线为空", symbol=symbol)
    date_col = _col(df, "日期", "date", "day")
    open_col = _col(df, "开盘", "open")
    high_col = _col(df, "最高", "high")
    low_col = _col(df, "最低", "low")
    close_col = _col(df, "收盘", "close")
    vol_col = _col(df, "成交量", "volume")
    amt_col = _col(df, "成交额", "amount")
    turn_col = _col(df, "换手率", "turnover", "换手")
    pct_col = _col(df, "涨跌幅", "pct_chg", "chg")
    if not all([date_col, open_col, high_col, low_col, close_col]):
        return MissingData("日线缺列", symbol=symbol)
    rows = []
    st = "ST" in symbol.upper()
    thr = limit_threshold(symbol, st=st)
    for _, row in df.iterrows():
        d = _as_date(row[date_col])
        if d is None:
            continue
        close = _num(row[close_col])
        high = _num(row[high_col])
        low = _num(row[low_col])
        pct = _num(row[pct_col]) if pct_col else None
        if close is None:
            continue
        limit_up = None
        limit_down = None
        if pct is not None and high is not None and low is not None and market is Market.CN:
            # 东财涨跌幅是百分数（10.02 = 10.02%）
            near_pct = (thr - 0.002) * 100
            limit_up = pct >= near_pct and close >= high * 0.998
            limit_down = pct <= -near_pct and close <= low * 1.002
        rows.append(
            {
                "symbol": symbol,
                "market": market.value,
                "date": d,
                "adjust": adjust,
                "open": _num(row[open_col]),
                "high": high,
                "low": low,
                "close": close,
                "volume": _num(row[vol_col]) if vol_col else None,
                "amount": _num(row[amt_col]) if amt_col else None,
                "turnover": _num(row[turn_col]) if turn_col else None,
                "pct_chg": pct,
                "limit_up": int(bool(limit_up)) if limit_up is not None else None,
                "limit_down": int(bool(limit_down)) if limit_down is not None else None,
            }
        )
    if not rows:
        return MissingData("日线为空", symbol=symbol)
    return pd.DataFrame(rows)


def _statements_from_df(symbol: str, kind: str, df: Any) -> list[dict] | MissingData:
    if df is None or getattr(df, "empty", True):
        return MissingData("报表为空", symbol=symbol, field=kind)
    period_col = _col(df, "REPORT_DATE", "报告日", "报告期", "period")
    notice_col = _col(df, "NOTICE_DATE", "公告日期", "announced", "公告日")
    rows = []
    for _, row in df.iterrows():
        period = _as_date(row[period_col]) if period_col else None
        announced = _as_date(row[notice_col]) if notice_col else period
        if announced is None:
            continue
        payload: dict[str, Any] = {}
        for col, val in row.items():
            key = str(col)
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:  # noqa: BLE001
                    val = str(val)
            if isinstance(val, float) and val != val:
                payload[key] = None
            else:
                payload[key] = val if not isinstance(val, (pd.Timestamp,)) else str(val)[:10]
        rows.append(
            {
                "symbol": symbol,
                "kind": kind,
                "period": period or announced,
                "announced": announced,
                "payload": payload,
            }
        )
    return rows or MissingData("报表无公告日", symbol=symbol, field=kind)


def _hk_index_members(fetcher: AkshareFetcher, name: str) -> list[str] | MissingData:
    sina = {"hsi": "HSI", "hstech": "HSTECH", "hscei": "HSCEI"}.get(name, name.upper())
    raw = fetcher._ak(["index_stock_cons_sina"], symbol=sina, field="hk_index")
    if not isinstance(raw, MissingData):
        df = raw
        col = _col(df, "code", "代码", "品种代码")
        if col is not None:
            out = []
            for v in df[col].tolist():
                code = str(v).split(".")[0].zfill(4)
                out.append(f"{code}.HK")
            if out:
                return out
    return MissingData(f"缺港股指数成分 {name}")


def _us_index_members(name: str) -> list[str] | MissingData:
    urls = {
        "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "ndx": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    }
    url = urls.get(name)
    if not url:
        return MissingData(f"未知美股名单 {name}")
    try:
        tables = pd.read_html(url)
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"读取成分失败: {exc}")
    for table in tables:
        col = None
        for cand in ("Symbol", "Ticker", "symbol"):
            if cand in table.columns:
                col = cand
                break
        if col is None:
            continue
        symbols = [str(v).strip().upper() for v in table[col].tolist() if str(v).strip()]
        if len(symbols) >= 20:
            return symbols
    return MissingData(f"解析成分失败 {name}")


def _yfinance_bars(symbol: str, start: date, end: date) -> pd.DataFrame | MissingData:
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"yfinance 不可用: {exc}", symbol=symbol)
    try:
        hist = yf.Ticker(symbol).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"yfinance 拉取失败: {exc}", symbol=symbol)
    if hist is None or hist.empty:
        return MissingData("日线为空", symbol=symbol)
    market = infer_market(symbol)
    rows = []
    for idx, row in hist.iterrows():
        d = _as_date(idx)
        if d is None or d < start or d > end:
            continue
        close = _num(row.get("Close"))
        if close is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "market": market.value,
                "date": d,
                "adjust": "none",
                "open": _num(row.get("Open")),
                "high": _num(row.get("High")),
                "low": _num(row.get("Low")),
                "close": close,
                "volume": _num(row.get("Volume")),
                "amount": None,
                "turnover": None,
                "pct_chg": None,
                "limit_up": 0,
                "limit_down": 0,
            }
        )
        adj = _num(row.get("Adj Close"))
        if adj is not None:
            qfq = dict(rows[-1])
            qfq["adjust"] = "qfq"
            qfq["close"] = adj
            rows.append(qfq)
    if not rows:
        return MissingData("日线为空", symbol=symbol)
    return pd.DataFrame(rows)


def _yfinance_financials(symbol: str) -> dict[str, list[dict]] | MissingData:
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"yfinance 不可用: {exc}", symbol=symbol)
    try:
        ticker = yf.Ticker(symbol)
        mapping = {
            "income": ticker.financials,
            "balance": ticker.balance_sheet,
            "cashflow": ticker.cashflow,
        }
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"yfinance 财务失败: {exc}", symbol=symbol)
    out: dict[str, list[dict]] = {}
    any_row = False
    for kind, df in mapping.items():
        rows: list[dict] = []
        if df is None or getattr(df, "empty", True):
            out[kind] = rows
            continue
        for col in df.columns:
            period = _as_date(col)
            if period is None:
                continue
            payload = {str(idx): _num(df.loc[idx, col]) for idx in df.index}
            rows.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "period": period,
                    "announced": period,  # yfinance 无公告日：用报告期并在上层标注缺数据风险
                    "payload": payload,
                    "announced_is_period": True,
                }
            )
            any_row = True
        out[kind] = rows
    if not any_row:
        return MissingData("无财务", symbol=symbol)
    return out


def baostock_bars(symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame | MissingData:
    try:
        import baostock as bs
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"baostock 不可用: {exc}", symbol=symbol)
    code = normalize_cn(symbol)
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    adj = "1" if adjust == "qfq" else "3"
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return MissingData(f"baostock 登录失败: {lg.error_msg}", symbol=symbol)
        rs = bs.query_history_k_data_plus(
            f"{prefix}.{code}",
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag=adj,
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rec = rs.get_row_data()
            rows.append(rec)
        bs.logout()
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"baostock 拉取失败: {exc}", symbol=symbol)
    if not rows:
        return MissingData("日线为空", symbol=symbol)
    df = pd.DataFrame(rows, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "涨跌幅"])
    return _normalize_cn_bars(code, df, adjust=adjust)


def tushare_bars(symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame | MissingData:
    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_PRO_TOKEN")
    if not token:
        return MissingData("未配置 TUSHARE_TOKEN", symbol=symbol)
    try:
        import tushare as ts
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"tushare 不可用: {exc}", symbol=symbol)
    code = normalize_cn(symbol)
    suffix = "SH" if code.startswith(("6", "9")) else "SZ"
    try:
        pro = ts.pro_api(token)
        kwargs = {
            "ts_code": f"{code}.{suffix}",
            "start_date": _yyyymmdd(start),
            "end_date": _yyyymmdd(end),
        }
        if adjust == "qfq":
            kwargs["adj"] = "qfq"
            df = pro.pro_bar(**kwargs)
        else:
            df = pro.daily(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"tushare 拉取失败: {exc}", symbol=symbol)
    if df is None or df.empty:
        return MissingData("日线为空", symbol=symbol)
    rename = {
        "trade_date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "vol": "成交量",
        "amount": "成交额",
        "pct_chg": "涨跌幅",
    }
    df = df.rename(columns=rename)
    return _normalize_cn_bars(code, df, adjust=adjust)
