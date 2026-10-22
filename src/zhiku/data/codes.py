"""市场代码规范化。"""

from __future__ import annotations

from zhiku.models import Market


def bare(symbol: str) -> str:
    return symbol.split(".")[0].upper()


def infer_market(symbol: str) -> Market:
    upper = symbol.upper()
    if upper.endswith(".HK"):
        return Market.HK
    if upper.endswith(".US") or (upper.isalpha() and not upper[0].isdigit()):
        return Market.US
    return Market.CN


def normalize_cn(symbol: str) -> str:
    code = bare(symbol)
    if len(code) < 6 and code.isdigit():
        code = code.zfill(6)
    return code


def normalize_bs(symbol: str) -> str:
    """baostock ``sh.600000`` / ``sz.000333`` → 六位代码。"""
    text = str(symbol).strip()
    if "." in text:
        left, right = text.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return normalize_cn(right)
    return normalize_cn(text)


def sina_code(symbol: str) -> str:
    """新浪 / 腾讯日线前缀。"""
    code = normalize_cn(symbol)
    if code.startswith(("6", "9", "5")):
        return "sh" + code
    return "sz" + code


def yfinance_cn(symbol: str) -> str:
    code = normalize_cn(symbol)
    if code.startswith(("6", "9", "5")):
        return f"{code}.SS"
    return f"{code}.SZ"


def em_code(symbol: str) -> str:
    """东方财富财务报表接口用 SH/SZ/BJ 前缀。"""
    code = normalize_cn(symbol)
    if code.startswith(("6", "9")):
        return "SH" + code
    if code.startswith(("8", "4")):
        return "BJ" + code
    return "SZ" + code


def qlib_instrument(symbol: str) -> str:
    market = infer_market(symbol)
    if market is Market.HK:
        return "HK" + bare(symbol).zfill(4)
    if market is Market.US:
        return bare(symbol)
    code = normalize_cn(symbol)
    return em_code(code)


def hk_hist_code(symbol: str) -> str:
    return bare(symbol).zfill(5)


def is_cn_etf(symbol: str) -> bool:
    code = normalize_cn(symbol)
    return len(code) == 6 and code[0] in {"5", "1"}


def limit_threshold(symbol: str, *, st: bool = False) -> float:
    if infer_market(symbol) is not Market.CN:
        return 9.9  # 港美股不按 A 股涨跌停处理
    if st:
        return 0.05
    code = normalize_cn(symbol)
    if code.startswith("688") or code.startswith("300"):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10
