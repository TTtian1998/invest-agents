"""核心数据模型。规则用 pydantic 校验；行情用 dataclass 避免过度包装。"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Market(str, Enum):
    CN = "CN"
    HK = "HK"
    US = "US"


class SignalStatus(str, Enum):
    TRIGGERED = "TRIGGERED"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    MISSING_DATA = "MISSING_DATA"
    PREVIEW = "PREVIEW"
    NOT_CLOSED = "NOT_CLOSED"


STATUS_CN: dict[SignalStatus, str] = {
    SignalStatus.TRIGGERED: "已触发",
    SignalStatus.NOT_TRIGGERED: "未触发",
    SignalStatus.MISSING_DATA: "缺数据",
    SignalStatus.PREVIEW: "预警",
    SignalStatus.NOT_CLOSED: "未收盘",
}

SELL_ACTIONS = frozenset({"sell_qty", "sell_fraction", "sell_all"})
FIELD_VALUES = ("price_1455", "close", "open", "low", "volume_ratio")
OP_VALUES = ("lt", "le", "gt", "ge")
ACTION_VALUES = ("sell_qty", "sell_fraction", "sell_all", "alert")


class MissingData:
    """任一数据源失败时的统一返回，报告里写「缺数据」，不估算、不插值。"""

    def __init__(
        self,
        reason: str,
        symbol: str | None = None,
        field: str | None = None,
        on: date | None = None,
    ) -> None:
        self.reason = reason
        self.symbol = symbol
        self.field = field
        self.on = on

    def __repr__(self) -> str:
        bits = [self.reason]
        if self.symbol:
            bits.append(f"symbol={self.symbol}")
        if self.field:
            bits.append(f"field={self.field}")
        if self.on:
            bits.append(f"on={self.on.isoformat()}")
        return f"MissingData({', '.join(bits)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MissingData):
            return NotImplemented
        return (
            self.reason == other.reason
            and self.symbol == other.symbol
            and self.field == other.field
            and self.on == other.on
        )


class Bar(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    price_1455: float | None = None
    market: Market | None = None

    def value_of(self, field: str) -> float | MissingData:
        if field == "price_1455":
            if self.price_1455 is None:
                return MissingData("无 14:55 价", symbol=self.symbol, field=field, on=self.date)
            return float(self.price_1455)
        mapping = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        if field not in mapping:
            return MissingData(f"未知字段 {field}", symbol=self.symbol, field=field, on=self.date)
        return float(mapping[field])


class Quote(BaseModel):
    symbol: str
    asof: datetime
    price: float
    source: str
    market: Market | None = None
    note: str = ""


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")

    qty: float = Field(gt=0)
    cost: float = Field(ge=0)


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: Literal["price_1455", "close", "open", "low", "volume_ratio"]
    op: Literal["lt", "le", "gt", "ge"]
    level: float
    consecutive: int = Field(default=1, ge=1)
    buffer_pct: float = Field(default=0.0)

    def effective_level(self) -> float:
        """lt/le 下移 buffer；gt/ge 上移 buffer。"""
        if self.op in {"lt", "le"}:
            return self.level * (1.0 - self.buffer_pct / 100.0)
        return self.level * (1.0 + self.buffer_pct / 100.0)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sell_qty", "sell_fraction", "sell_all", "alert"]
    qty: float | None = None
    fraction: float | None = None
    cumulative: bool = False

    @model_validator(mode="after")
    def _check_payload(self) -> Action:
        if self.type == "sell_qty" and self.qty is None:
            raise ValueError("sell_qty 需要提供 qty")
        if self.type == "sell_fraction":
            if self.fraction is None or not (0.0 < self.fraction <= 1.0):
                raise ValueError("fraction 必须在 (0, 1] 内")
        return self

    def sell_qty(self, position_qty: float, extra: float = 0.0) -> float:
        if self.type == "alert":
            return 0.0
        if self.type == "sell_all":
            own = position_qty
        elif self.type == "sell_fraction":
            own = position_qty * float(self.fraction or 0.0)
        else:
            own = float(self.qty or 0.0)
        if self.cumulative:
            return own + extra
        return own


class Rule(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    when: Condition
    action: Action
    supersedes: list[str] = Field(default_factory=list)
    and_conditions: list[Condition] = Field(default_factory=list, alias="and")


class SymbolBook(BaseModel):
    """一份 ``rules/*.yaml``：单标的规则 + 虚构持仓元数据。"""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    market: Market
    name: str = ""
    growth: bool = False
    add_blocked: bool = False
    position: Position
    rules: list[Rule]

    @model_validator(mode="after")
    def _check_ids(self) -> SymbolBook:
        ids = [r.id for r in self.rules]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"规则 id 重复: {', '.join(dupes)}")
        known = set(ids)
        for rule in self.rules:
            for sid in rule.supersedes:
                if sid not in known:
                    raise ValueError(f"supersedes 指向不存在的 id: {sid}")
        return self


class Signal(BaseModel):
    rule_id: str
    symbol: str
    market: Market
    status: SignalStatus
    actual: float | None = None
    effective_level: float | None = None
    price_source: str
    asof: datetime
    reason: str
    action: Action | None = None
    superseded_by: str | None = None
    sell_qty: float | None = None
    volume_ratio: float | None = None

    @property
    def status_cn(self) -> str:
        return STATUS_CN.get(self.status, self.status.value)


class TriggerEvent(BaseModel):
    rule_id: str
    symbol: str
    market: Market
    date: date
    trigger_price: float
    effective_level: float
    action: Action
    sell_qty: float | None = None
    price_source: str = "close"
    name: str = ""


class PortfolioPosition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    market: Market
    qty: float = Field(gt=0)
    cost: float = Field(ge=0)
    growth: bool = False
    add_blocked: bool = False
    name: str = ""


class Portfolio(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fx: dict[str, float] = Field(default_factory=dict)
    freeze_threshold: float = 0.60
    unfreeze_threshold: float = 0.50
    positions: list[PortfolioPosition]

    @field_validator("fx", mode="before")
    @classmethod
    def _fx_keys(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        return {str(k).upper().replace("/", "_"): float(v) for k, v in dict(value).items()}


class HitLabel(str, Enum):
    HIT = "HIT"
    FALSE_ALARM = "FALSE_ALARM"
    NEUTRAL = "NEUTRAL"
    PENDING = "PENDING"


HIT_CN: dict[HitLabel, str] = {
    HitLabel.HIT: "兑现",
    HitLabel.FALSE_ALARM: "误报",
    HitLabel.NEUTRAL: "中性",
    HitLabel.PENDING: "待定",
}
