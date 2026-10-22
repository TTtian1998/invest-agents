"""新浪/同花顺三大表 + baostock/tushare 明细财务。公告日 PIT，不插值。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from zhiku.data.codes import normalize_cn
from zhiku.models import MissingData


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() in {"", "-", "None", "nan"}:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in {float("inf"), float("-inf")}:
        return None
    return f


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
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

SINA_KIND = {
    "income": "利润表",
    "balance": "资产负债表",
    "cashflow": "现金流量表",
}

# 源列名 → 计算层别名（compute._ALIASES 能认的中文键）
SINA_FIELDS: dict[str, tuple[str, ...]] = {
    "营业利润": ("营业利润", "operate_profit"),
    "利息费用": ("利息费用", "财务费用"),
    "利润总额": ("利润总额", "税前利润"),
    "净利润": ("净利润", "归属于母公司股东的净利润", "归母净利润"),
    "营业总收入": ("营业总收入", "营业收入"),
    "营业成本": ("营业成本",),
    "经营活动产生的现金流量净额": ("经营活动产生的现金流量净额", "经营现金流量净额"),
    "流动资产合计": ("流动资产合计",),
    "流动负债合计": ("流动负债合计",),
    "固定资产净额": ("固定资产净额", "固定资产净值", "固定资产"),
    "货币资金": ("货币资金",),
    "短期借款": ("短期借款",),
    "长期借款": ("长期借款",),
    "应付债券": ("应付债券",),
    "一年内到期的非流动负债": ("一年内到期的非流动负债",),
    "资产总计": ("资产总计", "资产合计"),
    "负债合计": ("负债合计", "负债总计"),
    "股东权益合计": ("股东权益合计", "所有者权益合计"),
    "实收资本(或股本)": ("实收资本(或股本)", "股本", "实收资本"),
    "存货": ("存货",),
    "应收账款": ("应收账款",),
    "每股收益": ("每股收益", "基本每股收益", "epsTTM"),
    "每股净资产": ("每股净资产", "BPS"),
}


def sina_stock(symbol: str) -> str:
    """akshare stock_financial_report_sina 要 sh600519 / sz000333。"""
    code = normalize_cn(symbol)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _cell(row: Any, *names: str) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
        for col in row.index:
            if str(col).strip() == name:
                return row[col]
    return None


def map_sina_statement(
    symbol: str,
    kind: str,
    df: Any,
    *,
    announced_lookup: dict[str, date] | None = None,
) -> list[dict[str, Any]] | MissingData:
    """把新浪三大表映射成 store.upsert_statements 行。

    公告日优先用「公告日期」；没有则查已入库同报告期；再没有用报告期并标 announced_is_period。
    """
    if df is None or getattr(df, "empty", True):
        return MissingData(f"新浪{SINA_KIND.get(kind, kind)}为空", symbol=symbol)
    announced_lookup = announced_lookup or {}
    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        period = _as_date(_cell(raw, "报告日", "报告期", "report_date"))
        if period is None:
            continue
        announced = _as_date(_cell(raw, "公告日期", "公告日", "notice_date"))
        announced_is_period = False
        if announced is None:
            announced = announced_lookup.get(period.isoformat())
        if announced is None:
            announced = period
            announced_is_period = True
        payload: dict[str, Any] = {}
        for dest, aliases in SINA_FIELDS.items():
            val = _num(_cell(raw, dest, *aliases))
            if val is not None:
                payload[dest] = val
        if not payload:
            continue
        rows.append(
            {
                "symbol": normalize_cn(symbol),
                "kind": kind,
                "period": period,
                "announced": announced,
                "announced_is_period": announced_is_period,
                "payload": payload,
                "source": "sina",
            }
        )
    return rows or MissingData(f"新浪{SINA_KIND.get(kind, kind)}无有效行", symbol=symbol)


def merge_payload(base: dict[str, Any] | None, overlay: dict[str, Any] | None) -> dict[str, Any]:
    """明细覆盖整表同名字段；空值不覆盖已有值。"""
    out = dict(base or {})
    for key, val in (overlay or {}).items():
        if val is None or val == "":
            continue
        out[key] = val
    return out


def _cell_dict(row: dict[str, Any], *names: str) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in row:
            return row[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def map_baostock_statement(symbol: str, kind: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """baostock 季报 → store 行。只用 pubDate 作公告日；没有则用 statDate 并标 announced_is_period。"""
    period = _as_date(_cell_dict(row, "statDate", "stat_date"))
    announced = _as_date(_cell_dict(row, "pubDate", "pub_date"))
    announced_is_period = False
    if announced is None and period is not None:
        announced = period
        announced_is_period = True
    if announced is None:
        return None
    payload: dict[str, Any] = {}
    if kind == "income":
        mapping = {
            "净利润": ("netProfit", "净利润"),
            "营业总收入": ("MBRevenue", "主营营业收入", "营业收入"),
            "每股收益": ("epsTTM", "eps", "每股收益"),
            "净资产收益率": ("roeAvg", "ROE"),
            "毛利率": ("gpMargin", "销售毛利率"),
            "销售净利率": ("npMargin",),
            "总股本": ("totalShare", "总股本"),
            "流通股本": ("liqaShare",),
        }
    elif kind == "balance":
        mapping = {
            "资产负债率": ("liabilityToAsset", "资产负债率"),
            "流动比率": ("currentRatio",),
            "速动比率": ("quickRatio",),
            "权益乘数": ("assetToEquity",),
        }
    else:
        mapping = {
            "经营现金流比净利润": ("CFOToNP",),
            "经营现金流比营收": ("CFOToOR",),
        }
    for dest, aliases in mapping.items():
        val = _num(_cell_dict(row, dest, *aliases))
        if val is not None:
            payload[dest] = val
    if not payload:
        return None
    return {
        "symbol": normalize_cn(symbol),
        "kind": kind,
        "period": period or announced,
        "announced": announced,
        "announced_is_period": announced_is_period,
        "payload": payload,
        "source": "baostock",
    }


def map_baostock_forecast(symbol: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """业绩预告。公告日用 profitForcastExpPubDate；修正幅度取上下限中点。"""
    period = _as_date(
        _cell_dict(row, "profitForcastExpStatDate", "statDate", "stat_date")
    )
    announced = _as_date(
        _cell_dict(row, "profitForcastExpPubDate", "pubDate", "pub_date")
    )
    announced_is_period = False
    if announced is None and period is not None:
        announced = period
        announced_is_period = True
    if announced is None:
        return None
    up = _num(_cell_dict(row, "profitForcastChgPctUp", "incrementExp", "increment"))
    down = _num(_cell_dict(row, "profitForcastChgPctDwn"))
    increment = None
    if up is not None and down is not None:
        increment = (up + down) / 2.0
    elif up is not None:
        increment = up
    elif down is not None:
        increment = down
    if increment is not None and abs(increment) > 2:
        increment = increment / 100.0
    payload = {
        "profit_forecast": _num(_cell_dict(row, "profitForcastExp", "profitForecastExp", "profitForcast")),
        "profit_last": _num(_cell_dict(row, "profitForcastLastYear", "profitForecastLastYear")),
        "increment": increment,
        "increment_last": _num(_cell_dict(row, "incrementLastYear")),
        "forecast_type": _cell_dict(row, "profitForcastType", "type"),
        "forecast_range": _cell_dict(row, "profitForcastAbstract", "range"),
        "chg_pct_up": up,
        "chg_pct_down": down,
    }
    if all(payload.get(k) is None for k in ("profit_forecast", "profit_last", "increment")):
        return None
    return {
        "symbol": normalize_cn(symbol),
        "kind": "forecast",
        "period": period or announced,
        "announced": announced,
        "announced_is_period": announced_is_period,
        "payload": payload,
        "source": "baostock",
    }


def map_baostock_dividend(symbol: str, row: dict[str, Any]) -> dict[str, Any] | None:
    ex = _as_date(
        _cell_dict(row, "dividOperateDate", "dividRegistDate", "除权除息日", "ex_date")
    )
    announced = _as_date(
        _cell_dict(
            row,
            "dividPlanAnnounceDate",
            "dividPreNoticeDate",
            "dividAgmPdate",
            "公告日期",
            "announced",
        )
    )
    if ex is None and announced is None:
        return None
    amt = _num(
        _cell_dict(
            row,
            "dividCashPsBeforeTax",
            "dividCashPsAfterTax",
            "payout",
            "payouts",
            "派息",
        )
    )
    return {
        "symbol": normalize_cn(symbol),
        "ex_date": ex or announced,
        "announced": announced or ex,
        "amount": amt,
        "source": "baostock",
    }


def map_ak_dividend_history(symbol: str, df: Any) -> list[dict[str, Any]] | MissingData:
    """akshare 历史分红（stock_history_dividend_detail / fhps 明细）。"""
    if df is None or getattr(df, "empty", True):
        return MissingData("历史分红为空", symbol=symbol)
    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        ex = _as_date(_cell(raw, "除权除息日", "除息日", "ex_date", "实施日期"))
        announced = _as_date(_cell(raw, "公告日期", "预案公告日", "最新公告日期", "announced"))
        if ex is None and announced is None:
            continue
        amt = _num(_cell(raw, "派息", "现金分红", "分红", "每股派息", "现金分红-现金分红比例", "amount"))
        yld = _num(_cell(raw, "股息率", "现金分红-股息率", "yield"))
        rows.append(
            {
                "symbol": normalize_cn(symbol),
                "ex_date": ex or announced,
                "announced": announced or ex,
                "amount": amt,
                "yield": yld,
                "source": "akshare",
            }
        )
    return rows or MissingData("历史分红无有效行", symbol=symbol)

