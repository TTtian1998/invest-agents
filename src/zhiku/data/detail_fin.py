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
