"""截面财务快照。缺字段保持 None，计算端返回 MissingData，不估算。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from zhiku.models import Market


class Fundamentals(BaseModel):
    """单标的截面财务 / 估值 / 量价。数值均为调用方提供（fixture 或数据源），不内插。"""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    market: Market = Market.CN
    name: str = ""
    industry: str = ""
    instrument: str = "stock"  # stock | etf
    industry_leader: bool = False
    st: bool = False
    delist_risk: bool = False
    mktcap: float | None = None
    amount_avg: float | None = None
    fx_to_cny: float = 1.0
    pe: float | None = None
    pb: float | None = None
    pe_percentile: float | None = None
    pb_percentile: float | None = None
    dividend_yield: float | None = None
    dividend_years: int | None = None
    roe: float | None = None
    roe_3y: list[float] = Field(default_factory=list)
    roa: float | None = None
    roa_prev: float | None = None
    gross_margin: float | None = None
    gross_margin_prev: float | None = None
    gross_margin_3y: list[float] = Field(default_factory=list)
    ebit: float | None = None
    nwc: float | None = None
    nfa: float | None = None
    ev: float | None = None
    ocf: float | None = None
    net_income: float | None = None
    debt_asset: float | None = None
    lt_debt_asset: float | None = None
    lt_debt_asset_prev: float | None = None
    current_ratio: float | None = None
    current_ratio_prev: float | None = None
    shares_issued: bool | None = None
    asset_turnover: float | None = None
    asset_turnover_prev: float | None = None
    momentum_12_1: float | None = None
    volatility: float | None = None

    @property
    def roe_3y_mean(self) -> float | None:
        if not self.roe_3y:
            return self.roe
        return sum(self.roe_3y) / len(self.roe_3y)

    @property
    def amount_cny(self) -> float | None:
        if self.amount_avg is None:
            return None
        return self.amount_avg * self.fx_to_cny

    @property
    def mktcap_cny(self) -> float | None:
        if self.mktcap is None:
            return None
        return self.mktcap * self.fx_to_cny


def load_fundamentals(path: Path) -> list[Fundamentals]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "stocks" in raw:
        rows = raw["stocks"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError(f"{path} 应为列表或带 stocks 键的 mapping")
    return [Fundamentals.model_validate(row) for row in rows]


def by_symbol(rows: list[Fundamentals]) -> dict[str, Fundamentals]:
    return {row.symbol: row for row in rows}
