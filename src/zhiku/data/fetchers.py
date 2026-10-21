"""akshare 为主、yfinance 补充；baostock / tushare 有 token 时作备用。"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from zhiku.data.ashare import _call_first_available, _col, _to_date
from zhiku.data.codes import (
    em_code,
    hk_hist_code,
    infer_market,
    is_cn_etf,
    limit_threshold,
    normalize_bs,
    normalize_cn,
    yfinance_cn,
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

    def index_members_hist(self, index_code: str, on: date) -> list[str] | MissingData:
        return MissingData(f"未实现历史成分 {index_code}@{on}")

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

    def bulk_financials(self, period: date, kind: str) -> list[dict] | MissingData:
        return MissingData("未提供整表财务")

    def bulk_dividends(self, period: date) -> list[dict] | MissingData:
        return MissingData("未提供整表分红")

    def cn_statement_detail(self, symbol: str) -> list[dict] | MissingData:
        return MissingData("未提供明细财务", symbol=symbol)


_EM_HIST_FAILS = 0
_EM_HIST_SKIP_AFTER = 3


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

    def index_members_hist(self, index_code: str, on: date) -> list[str] | MissingData:
        batch = baostock_index_members_hist([(index_code, on)])
        return batch.get((index_code, on), MissingData(f"缺历史成分 {index_code}@{on}"))

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
        open_col = _col(df, "今开", "开盘", "open")
        high_col = _col(df, "最高", "high")
        low_col = _col(df, "最低", "low")
        close_col = _col(df, "最新价", "收盘", "close")
        vol_col = _col(df, "成交量", "volume")
        pct_col = _col(df, "涨跌幅", "pct_chg")
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
                    "open": _num(row[open_col]) if open_col else None,
                    "high": _num(row[high_col]) if high_col else None,
                    "low": _num(row[low_col]) if low_col else None,
                    "close": _num(row[close_col]) if close_col else None,
                    "volume": _num(row[vol_col]) if vol_col else None,
                    "pct_chg": _num(row[pct_col]) if pct_col else None,
                    "st": "ST" in name.upper(),
                }
            )
        return out

    def cn_bars(self, symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame | MissingData:
        code = normalize_cn(symbol)
        adj_flag = "none" if adjust in {"none", "raw", ""} else "qfq"
        adj = "" if adj_flag == "none" else "qfq"
        errors: list[str] = []
        # 东财常被断开，只试一次；随后新浪 / 腾讯 / yfinance。
        if is_cn_etf(code):
            raw = self._ak(
                ["fund_etf_hist_em"],
                symbol=code,
                period="daily",
                start_date=_yyyymmdd(start),
                end_date=_yyyymmdd(end),
                adjust=adj,
                md_symbol=code,
                field="bars",
            )
            if not isinstance(raw, MissingData):
                out = _normalize_cn_bars(code, raw, adjust=adj_flag)
                if not isinstance(out, MissingData):
                    return out
            errors.append(str(getattr(raw, "reason", raw)))
            bao = baostock_bars(code, start, end, adj_flag)
            if not isinstance(bao, MissingData):
                return bao
        else:
            global _EM_HIST_FAILS
            if _EM_HIST_FAILS < _EM_HIST_SKIP_AFTER:
                raw = retry_call(
                    lambda: self._caller(
                        ["stock_zh_a_hist"],
                        symbol=code,
                        period="daily",
                        start_date=_yyyymmdd(start),
                        end_date=_yyyymmdd(end),
                        adjust=adj,
                    ),
                    retries=0,
                    bucket="akshare-em",
                    sleep=self.sleep,
                    symbol=code,
                    field="bars",
                )
                if not isinstance(raw, MissingData):
                    out = _normalize_cn_bars(code, raw, adjust=adj_flag)
                    if not isinstance(out, MissingData):
                        _EM_HIST_FAILS = 0
                        return out
                _EM_HIST_FAILS += 1
                errors.append("eastmoney hist failed")
            bao = baostock_bars(code, start, end, adj_flag)
            if not isinstance(bao, MissingData):
                return bao
            errors.append("baostock failed")
            yf = _yfinance_bars(yfinance_cn(code), start, end)
            if not isinstance(yf, MissingData):
                yf = yf.copy()
                yf["symbol"] = code
                yf["market"] = "CN"
                want = "qfq" if adj_flag == "qfq" else "none"
                if "adjust" in yf.columns:
                    yf = yf[yf["adjust"] == want]
                if not yf.empty:
                    return yf
            errors.append("yfinance failed")
        return MissingData("日线失败: " + "; ".join(errors), symbol=code)

    def bulk_financials(self, period: date, kind: str) -> list[dict] | MissingData:
        """全市场一张表。kind=income/balance/cashflow/yjbb。"""
        date_s = period.strftime("%Y%m%d")
        api = {
            "income": (["stock_lrb_em"], "income"),
            "balance": (["stock_zcfz_em"], "balance"),
            "cashflow": (["stock_xjll_em"], "cashflow"),
            "yjbb": (["stock_yjbb_em"], "income"),
        }.get(kind)
        if not api:
            return MissingData(f"未知整表 {kind}")
        names, stmt_kind = api
        raw = self._ak(names, date=date_s, field=f"bulk:{kind}")
        if isinstance(raw, MissingData):
            return raw
        return _bulk_statements(raw, stmt_kind, period)

    def bulk_dividends(self, period: date) -> list[dict] | MissingData:
        date_s = period.strftime("%Y%m%d")
        raw = self._ak(["stock_fhps_em"], date=date_s, field="bulk:div")
        if isinstance(raw, MissingData):
            return raw
        if raw is None or getattr(raw, "empty", True):
            return MissingData(f"整表分红为空 {period}")
        code_col = _col(raw, "代码", "股票代码", "code")
        ex_col = _col(raw, "除权除息日", "除息日", "ex_date")
        ann_col = _col(raw, "预案公告日", "最新公告日期", "公告日期", "announced")
        amt_col = _col(raw, "现金分红-现金分红比例", "现金分红", "派息", "amount")
        yield_col = _col(raw, "现金分红-股息率", "股息率", "yield")
        if code_col is None or ex_col is None:
            return MissingData("整表分红缺代码或除权日")
        rows: list[dict] = []
        for _, row in raw.iterrows():
            ex = _as_date(row[ex_col])
            if ex is None:
                continue
            rows.append(
                {
                    "symbol": normalize_cn(str(row[code_col])),
                    "ex_date": ex,
                    "announced": _as_date(row[ann_col]) if ann_col else ex,
                    "amount": _num(row[amt_col]) if amt_col else None,
                    "yield": _num(row[yield_col]) if yield_col else None,
                }
            )
        return rows or MissingData(f"整表分红无行 {period}")

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

    def cn_statement_detail(self, symbol: str) -> list[dict] | MissingData:
        """新浪三大表；失败再试 tushare。baostock 季频只作比率补充。"""
        from zhiku.data.detail_fin import map_sina_statement, sina_stock

        code = normalize_cn(symbol)
        stock = sina_stock(code)
        rows: list[dict] = []
        errors: list[str] = []
        for kind, sheet in {
            "income": "利润表",
            "balance": "资产负债表",
            "cashflow": "现金流量表",
        }.items():
            raw = self._ak(
                ["stock_financial_report_sina"],
                stock=stock,
                symbol=sheet,
                md_symbol=code,
                field=f"sina:{kind}",
            )
            if isinstance(raw, MissingData):
                errors.append(f"{sheet}:{raw.reason}")
                continue
            mapped = map_sina_statement(code, kind, raw)
            if isinstance(mapped, MissingData):
                errors.append(f"{sheet}:{mapped.reason}")
                continue
            rows.extend(mapped)
        if rows:
            return rows
        ts = _tushare_statements(code)
        if not isinstance(ts, MissingData):
            return ts
        return MissingData(
            "未提供明细财务: " + ("; ".join(errors) or "新浪三大表为空"),
            symbol=code,
        )

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


_BULK_MAP = {
    "净利润": "净利润",
    "净利润-净利润": "净利润",
    "营业总收入": "营业总收入",
    "营业总收入-营业总收入": "营业总收入",
    "营业利润": "营业利润",
    "利润总额": "利润总额",
    "营业总支出-财务费用": "利息费用",
    "资产-货币资金": "货币资金",
    "资产-应收账款": "应收账款",
    "资产-存货": "存货",
    "资产-总资产": "资产总计",
    "负债-总负债": "负债合计",
    "股东权益合计": "股东权益合计",
    "资产负债率": "资产负债率",
    "经营性现金流-现金流量净额": "经营活动产生的现金流量净额",
    "净资产收益率": "净资产收益率",
    "销售毛利率": "毛利率",
    "每股收益": "每股收益",
    "每股净资产": "每股净资产",
}


def _bulk_statements(df: Any, kind: str, period: date) -> list[dict] | MissingData:
    if df is None or getattr(df, "empty", True):
        return MissingData(f"整表 {kind} 为空")
    code_col = _col(df, "股票代码", "code")
    name_col = _col(df, "股票简称", "名称", "name")
    notice_col = _col(df, "公告日期", "最新公告日期", "NOTICE_DATE")
    if code_col is None:
        return MissingData(f"整表 {kind} 缺代码列")
    rows: list[dict] = []
    for _, row in df.iterrows():
        code = normalize_cn(str(row[code_col]))
        announced = _as_date(row[notice_col]) if notice_col else period
        if announced is None:
            announced = period
        payload: dict[str, Any] = {}
        for col, val in row.items():
            payload[str(col)] = None if isinstance(val, float) and val != val else val
        for src, dest in _BULK_MAP.items():
            if src in df.columns and dest not in payload:
                payload[dest] = _num(row[src])
            elif src in df.columns:
                payload[dest] = _num(row[src])
        if name_col:
            payload["name"] = str(row[name_col])
        rows.append(
            {
                "symbol": code,
                "kind": kind,
                "period": period,
                "announced": announced,
                "payload": payload,
            }
        )
    return rows or MissingData(f"整表 {kind} 无行")


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


def baostock_index_members_hist(
    jobs: list[tuple[str, date]],
    *,
    timeout: int = 480,
) -> dict[tuple[str, date], list[str] | MissingData]:
    """子进程按日回查沪深300/中证500，并集即中证800。"""
    import json
    import subprocess
    import sys

    if not jobs:
        return {}
    worker = Path(__file__).resolve().with_name("_bs_index.py")
    if not worker.is_file():
        return {job: MissingData("缺少 baostock 成分 worker") for job in jobs}
    payload = {
        "jobs": [
            {"index": INDEX_CSI.get(code, code), "date": on.isoformat()}
            for code, on in jobs
        ]
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {job: MissingData("baostock 历史成分超时") for job in jobs}
    except Exception as exc:  # noqa: BLE001
        return {job: MissingData(f"baostock 历史成分失败: {exc}") for job in jobs}
    raw_text = (proc.stdout or "").strip()
    brace = raw_text.find("{")
    if brace < 0:
        err = (proc.stderr or "").strip()[:200]
        return {job: MissingData(f"baostock 历史成分无输出 {err}") for job in jobs}
    try:
        body = json.loads(raw_text[brace:])
    except json.JSONDecodeError:
        return {job: MissingData("baostock 历史成分无法解析") for job in jobs}
    by_key: dict[tuple[str, str], dict] = {}
    for row in body.get("results") or []:
        by_key[(str(row.get("index") or ""), str(row.get("date") or "")[:10])] = row
    out: dict[tuple[str, date], list[str] | MissingData] = {}
    for code, on in jobs:
        raw_code = INDEX_CSI.get(code, code)
        row = by_key.get((raw_code, on.isoformat()))
        if not row:
            out[(code, on)] = MissingData(f"缺历史成分 {raw_code}@{on}")
            continue
        codes = [normalize_bs(str(v)) for v in (row.get("codes") or []) if v]
        codes = [c for c in codes if len(c) == 6 and c.isdigit()]
        if codes:
            out[(code, on)] = list(dict.fromkeys(codes))
        else:
            out[(code, on)] = MissingData(row.get("error") or f"空成分 {raw_code}@{on}")
    return out


def baostock_bars(symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame | MissingData:
    """子进程拉 baostock：超时 terminate，避免登录连接被挂死请求拖垮。"""
    import json
    import subprocess
    import sys

    code = normalize_cn(symbol)
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    adj = "1" if adjust == "qfq" else "3"
    worker = Path(__file__).resolve().with_name("_bs_query.py")
    if not worker.is_file():
        return MissingData("缺少 baostock worker", symbol=symbol)
    cmd = [
        sys.executable,
        str(worker),
        f"{prefix}.{code}",
        start.isoformat(),
        end.isoformat(),
        adj,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=18,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MissingData("baostock 超时", symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"baostock 拉取失败: {exc}", symbol=symbol)
    raw_text = (proc.stdout or "").strip()
    if not raw_text:
        err = (proc.stderr or "").strip()[:200]
        return MissingData(f"baostock 无输出 {err}", symbol=symbol)
    brace = raw_text.find("{")
    if brace < 0:
        return MissingData("baostock 输出无法解析", symbol=symbol)
    try:
        payload = json.loads(raw_text[brace:])
    except json.JSONDecodeError:
        return MissingData("baostock 输出无法解析", symbol=symbol)
    if payload.get("error") and not payload.get("rows"):
        return MissingData(f"baostock {payload.get('error')}", symbol=symbol)
    rows = payload.get("rows") or []
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


def _tushare_statements(symbol: str) -> list[dict] | MissingData:
    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_PRO_TOKEN")
    if not token:
        return MissingData("未配置 TUSHARE_TOKEN", symbol=symbol)
    try:
        import tushare as ts
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"tushare 不可用: {exc}", symbol=symbol)
    code = normalize_cn(symbol)
    suffix = "SH" if code.startswith(("6", "9")) else "SZ"
    ts_code = f"{code}.{suffix}"
    try:
        pro = ts.pro_api(token)
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"tushare 登录失败: {exc}", symbol=symbol)
    from zhiku.data.detail_fin import merge_payload

    kind_api = {
        "income": ("income", {"operate_profit": "营业利润", "n_income": "净利润", "revenue": "营业总收入", "int_exp": "利息费用", "total_profit": "利润总额"}),
        "balance": (
            "balancesheet",
            {
                "total_cur_assets": "流动资产合计",
                "total_cur_liab": "流动负债合计",
                "fix_assets": "固定资产净额",
                "money_cap": "货币资金",
                "st_borr": "短期借款",
                "lt_borr": "长期借款",
                "total_assets": "资产总计",
                "total_liab": "负债合计",
                "total_hldr_eqy_exc_min_int": "股东权益合计",
            },
        ),
        "cashflow": ("cashflow", {"n_cashflow_act": "经营活动产生的现金流量净额"}),
    }
    rows: list[dict] = []
    for kind, (api, mapping) in kind_api.items():
        fn = getattr(pro, api, None)
        if fn is None:
            continue
        try:
            df = fn(ts_code=ts_code)
        except Exception:
            continue
        if df is None or getattr(df, "empty", True):
            continue
        for _, raw in df.iterrows():
            period = _as_date(raw.get("end_date"))
            announced = _as_date(raw.get("ann_date") or raw.get("f_ann_date"))
            if period is None:
                continue
            payload: dict = {}
            for src, dest in mapping.items():
                val = _num(raw.get(src))
                if val is not None:
                    payload[dest] = val
            if not payload:
                continue
            rows.append(
                {
                    "symbol": code,
                    "kind": kind,
                    "period": period,
                    "announced": announced or period,
                    "announced_is_period": announced is None,
                    "payload": merge_payload({}, payload),
                    "source": "tushare",
                }
            )
    return rows or MissingData("tushare 财报为空", symbol=symbol)
