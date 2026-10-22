"""用 akshare 拉 A 股财务/估值的尽力而为适配。失败返回空并让上层用 fixture 或记缺数据。

字段映射参考 myhhub/stock 等 A 股选股项目的公开思路，不复制其源码。
"""

from __future__ import annotations

from zhiku.data.ashare import _call_first_available
from zhiku.models import MissingData
from zhiku.screens.fundamentals import Fundamentals


def fetch_cn_spot_row(symbol: str) -> Fundamentals | MissingData:
    raw = _call_first_available(
        ["stock_zh_a_spot_em", "stock_zh_a_spot"],
    )
    if isinstance(raw, MissingData):
        return raw
    try:
        df = raw
        code_col = "代码" if "代码" in df.columns else df.columns[0]
        hit = df[df[code_col].astype(str).str.zfill(6) == symbol.zfill(6)]
        if hit.empty:
            return MissingData("行情截面无此代码", symbol=symbol)
        row = hit.iloc[0]
        name = str(row["名称"]) if "名称" in hit.columns else ""
        st = "ST" in name.upper() or "st" in name
        pe = float(row["市盈率-动态"]) if "市盈率-动态" in hit.columns else None
        pb = float(row["市净率"]) if "市净率" in hit.columns else None
        mktcap = float(row["总市值"]) if "总市值" in hit.columns else None
        amount = float(row["成交额"]) if "成交额" in hit.columns else None
        return Fundamentals(
            symbol=symbol,
            name=name,
            st=st,
            pe=pe,
            pb=pb,
            mktcap=mktcap,
            amount_avg=amount,
        )
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"解析截面失败: {exc}", symbol=symbol)
