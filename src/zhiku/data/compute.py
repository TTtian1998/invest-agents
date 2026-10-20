"""从报表/日线算出选股字段。缺字段保持 None，不插值。"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from zhiku.models import Market
from zhiku.screens.fundamentals import Fundamentals

# 东方财富 / yfinance 常见列名。按顺序取第一个能转成数字的。
_ALIASES: dict[str, tuple[str, ...]] = {
    "ebit": ("息税前利润", "EBIT", "Operate Profit", "OPERATE_PROFIT", "营业利润"),
    "interest": ("利息费用", "Interest Expense", "INTEREST_EXPENSE", "财务费用"),
    "total_profit": ("利润总额", "Pretax Income", "TOTAL_PROFIT", "税前利润"),
    "net_income": ("净利润", "Net Income", "NETPROFIT", "归母净利润", "归属于母公司股东的净利润"),
    "revenue": ("营业总收入", "营业收入", "Total Revenue", "OPERATE_INCOME", "营业收入"),
    "cogs": ("营业成本", "Cost Of Revenue", "OPERATE_COST"),
    "gross_margin_pct": ("毛利率", "GROSS_MARGIN"),
    "ocf": (
        "经营活动产生的现金流量净额",
        "Operating Cash Flow",
        "NETCASH_OPERATE",
        "经营现金流量净额",
    ),
    "current_assets": ("流动资产合计", "Current Assets", "TOTAL_CURRENT_ASSETS"),
    "cash": ("货币资金", "Cash And Cash Equivalents", "MONETARYFUNDS", "Cash"),
    "current_liab": ("流动负债合计", "Current Liabilities", "TOTAL_CURRENT_LIAB"),
    "st_debt": ("短期借款", "Short Long Term Debt", "SHORT_LOAN", "短期债务"),
    "lt_debt": ("长期借款", "Long Term Debt", "LONG_LOAN"),
    "bonds": ("应付债券", "Bonds Payable", "BOND_PAYABLE"),
    "current_lt": ("一年内到期的非流动负债", "Current Portion Of Long Term Debt"),
    "fixed_assets": ("固定资产", "Net PPE", "FIXED_ASSET", "固定资产净值"),
    "total_assets": ("资产总计", "Total Assets", "TOTAL_ASSETS"),
    "total_liab": ("负债合计", "Total Liab", "TOTAL_LIABILITIES", "负债总计"),
    "equity": ("股东权益合计", "Stockholders Equity", "TOTAL_EQUITY", "所有者权益合计"),
    "shares": ("实收资本(或股本)", "Ordinary Shares Number", "SHARE_CAPITAL", "股本"),
    "inventory": ("存货", "Inventory", "INVENTORY"),
    "receivables": ("应收账款", "Net Receivables", "ACCOUNTS_RECE"),
}


def pick_num(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    for alias in _ALIASES.get(key, (key,)):
        if alias in payload:
            val = _to_float(payload[alias])
            if val is not None:
                return val
        # 大小写不敏感
        lower = {str(k).lower(): v for k, v in payload.items()}
        if alias.lower() in lower:
            val = _to_float(lower[alias.lower()])
            if val is not None:
                return val
    return None


def _to_float(value: Any) -> float | None:
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


def ebit_of(income: dict[str, Any] | None) -> float | None:
    direct = pick_num(income, "ebit")
    if direct is not None:
        return direct
    total = pick_num(income, "total_profit")
    interest = pick_num(income, "interest")
    if total is not None and interest is not None:
        return total + interest
    return None


def nwc_of(balance: dict[str, Any] | None) -> float | None:
    ca = pick_num(balance, "current_assets")
    cash = pick_num(balance, "cash")
    cl = pick_num(balance, "current_liab")
    st = pick_num(balance, "st_debt")
    if ca is None or cl is None:
        return None
    # Greenblatt：流动资产-现金-（流动负债-无息流动负债近似用短债加回）
    cash_v = 0.0 if cash is None else cash
    st_v = 0.0 if st is None else st
    return ca - cash_v - (cl - st_v)


def nfa_of(balance: dict[str, Any] | None) -> float | None:
    return pick_num(balance, "fixed_assets")


def interest_bearing(balance: dict[str, Any] | None) -> float | None:
    parts = [
        pick_num(balance, "st_debt"),
        pick_num(balance, "lt_debt"),
        pick_num(balance, "bonds"),
        pick_num(balance, "current_lt"),
    ]
    if all(p is None for p in parts):
        return None
    return sum(p or 0.0 for p in parts)


def ev_of(mktcap: float | None, balance: dict[str, Any] | None) -> float | None:
    if mktcap is None:
        return None
    debt = interest_bearing(balance)
    cash = pick_num(balance, "cash")
    if debt is None and cash is None:
        return None
    return mktcap + (debt or 0.0) - (cash or 0.0)


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def gross_margin_of(income: dict[str, Any] | None) -> float | None:
    pct = pick_num(income, "gross_margin_pct")
    if pct is not None:
        return pct / 100.0 if pct > 1 else pct
    rev = pick_num(income, "revenue")
    cogs = pick_num(income, "cogs")
    if rev is None or cogs is None or rev == 0:
        return None
    return (rev - cogs) / rev


def consecutive_div_years(ex_dates: list[date], asof: date) -> int | None:
    years = sorted({d.year for d in ex_dates if d <= asof}, reverse=True)
    if not years:
        return None
    if years[0] < asof.year - 1:
        return 0
    n = 0
    expect = years[0]
    for y in years:
        if y == expect:
            n += 1
            expect -= 1
        elif y > expect:
            continue
        else:
            break
    return n


def momentum_12_1(closes: pd.Series) -> float | None:
    s = closes.dropna().astype(float).sort_index()
    if len(s) < 252:
        return None
    m1 = float(s.iloc[-22]) if len(s) >= 22 else None
    m12 = float(s.iloc[-252])
    if m1 is None or m12 == 0:
        return None
    return (m1 / m12) - 1.0


def realized_vol(closes: pd.Series, window: int = 60) -> float | None:
    s = closes.dropna().astype(float).sort_index()
    if len(s) < window + 1:
        return None
    ret = s.pct_change().dropna()
    if len(ret) < window:
        return None
    return float(ret.iloc[-window:].std(ddof=0)) * (252**0.5)


def amount_avg(amounts: pd.Series, window: int = 20) -> float | None:
    s = amounts.dropna().astype(float)
    if s.empty:
        return None
    return float(s.iloc[-window:].mean())


def snapshot_from_store(
    *,
    symbol: str,
    asof: date,
    listing: dict[str, Any],
    statements: dict[str, dict[str, Any]],
    prev_income: dict[str, Any] | None,
    prev_balance: dict[str, Any] | None,
    spot: dict[str, Any] | None,
    closes: pd.Series | None,
    amounts: pd.Series | None,
    div_years: int | None,
    div_yield: float | None,
    shares_issued: bool | None,
    market: Market = Market.CN,
) -> Fundamentals:
    income = (statements.get("income") or {}).get("payload")
    balance = (statements.get("balance") or {}).get("payload")
    cashflow = (statements.get("cashflow") or {}).get("payload")
    mktcap = None if not spot else _to_float(spot.get("mktcap"))
    pe = None if not spot else _to_float(spot.get("pe"))
    pb = None if not spot else _to_float(spot.get("pb"))
    amount = None if not spot else _to_float(spot.get("amount"))
    if amount is None and amounts is not None:
        amount = amount_avg(amounts)
    ni = pick_num(income, "net_income")
    assets = pick_num(balance, "total_assets")
    equity = pick_num(balance, "equity")
    ocf = pick_num(cashflow, "ocf")
    gm = gross_margin_of(income)
    gm_prev = gross_margin_of(prev_income)
    roa = ratio(ni, assets)
    roa_prev = ratio(pick_num(prev_income, "net_income"), pick_num(prev_balance, "total_assets"))
    current = ratio(pick_num(balance, "current_assets"), pick_num(balance, "current_liab"))
    current_prev = ratio(
        pick_num(prev_balance, "current_assets"), pick_num(prev_balance, "current_liab")
    )
    lt = ratio(pick_num(balance, "lt_debt"), assets)
    lt_prev = ratio(pick_num(prev_balance, "lt_debt"), pick_num(prev_balance, "total_assets"))
    ato = ratio(pick_num(income, "revenue"), assets)
    ato_prev = ratio(pick_num(prev_income, "revenue"), pick_num(prev_balance, "total_assets"))
    roe = ratio(ni, equity)
    debt_asset = ratio(pick_num(balance, "total_liab"), assets)
    ebit = ebit_of(income)
    nwc = nwc_of(balance)
    nfa = nfa_of(balance)
    ev = ev_of(mktcap, balance)
    name = (spot or {}).get("name") or listing.get("name") or ""
    st = bool(listing.get("st")) or "ST" in str(name).upper()
    mom = momentum_12_1(closes) if closes is not None else None
    vol = realized_vol(closes) if closes is not None else None
    gm3 = [x for x in (gm_prev, gm) if x is not None]
    roe3 = [roe] if roe is not None else []
    return Fundamentals(
        symbol=symbol,
        market=market,
        name=str(name),
        industry=str(listing.get("industry") or ""),
        instrument=str(listing.get("instrument") or "stock"),
        st=st,
        delist_risk=bool(listing.get("delist_date")),
        mktcap=mktcap,
        amount_avg=amount,
        pe=pe,
        pb=pb,
        dividend_yield=div_yield,
        dividend_years=div_years,
        roe=roe,
        roe_3y=roe3,
        roa=roa,
        roa_prev=roa_prev,
        gross_margin=gm,
        gross_margin_prev=gm_prev,
        gross_margin_3y=gm3,
        ebit=ebit,
        nwc=nwc,
        nfa=nfa,
        ev=ev,
        ocf=ocf,
        net_income=ni,
        debt_asset=debt_asset,
        lt_debt_asset=lt,
        lt_debt_asset_prev=lt_prev,
        current_ratio=current,
        current_ratio_prev=current_prev,
        shares_issued=shares_issued,
        asset_turnover=ato,
        asset_turnover_prev=ato_prev,
        momentum_12_1=mom,
        volatility=vol,
    )


def attach_percentiles(rows: list[Fundamentals]) -> list[Fundamentals]:
    """截面 PE/PB 分位。没有估值的保持 None，不拿均值填。"""
    pes = [(i, r.pe) for i, r in enumerate(rows) if r.pe is not None and r.pe > 0]
    pbs = [(i, r.pb) for i, r in enumerate(rows) if r.pb is not None and r.pb > 0]

    def _pct(pairs: list[tuple[int, float]]) -> dict[int, float]:
        if not pairs:
            return {}
        ordered = sorted(pairs, key=lambda x: x[1])
        n = len(ordered)
        return {idx: rank / max(n - 1, 1) for rank, (idx, _) in enumerate(ordered)}

    pe_p = _pct(pes)
    pb_p = _pct(pbs)
    out = []
    for i, row in enumerate(rows):
        data = row.model_dump()
        data["pe_percentile"] = pe_p.get(i)
        data["pb_percentile"] = pb_p.get(i)
        out.append(Fundamentals.model_validate(data))
    return out


def missing_ratio(rows: list[Fundamentals], fields: list[str] | None = None) -> float:
    keys = fields or [
        "pe",
        "pb",
        "mktcap",
        "amount_avg",
        "roe",
        "ocf",
        "debt_asset",
        "ebit",
        "ev",
        "net_income",
        "pe_percentile",
        "pb_percentile",
    ]
    if not rows or not keys:
        return 1.0
    missing = 0
    total = 0
    for row in rows:
        data = row.model_dump()
        for k in keys:
            total += 1
            if data.get(k) is None or data.get(k) == []:
                missing += 1
    return missing / total if total else 1.0
