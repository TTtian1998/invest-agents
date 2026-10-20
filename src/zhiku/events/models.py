"""事件驱动层的结构化记录。缺字段保持空，不编造。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SessionLabel = Literal["盘前", "盘中", "盘后", "隔夜", "缺数据"]
CrowdingBucket = Literal["high", "mid", "low", "unknown"]


class SurpriseCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = ""
    kind: str = ""  # eps | revenue | macro | other
    consensus: float | None = None
    actual: float | None = None
    sigma: float | None = None
    surprise_z: float | None = None
    surprise_missing: str | None = None
    implied_move: float | None = None
    actual_move: float | None = None
    vs_implied: str = ""
    source: str = ""
    asof: date | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class NoveltyCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    first_seen: datetime | None = None
    first_source: str = ""
    novelty: float | None = None
    session: SessionLabel = "缺数据"
    session_market: str = ""
    earliest_executable: str = "缺数据"
    age_hours: float | None = None
    half_life_hours: float | None = None
    decay: float = 1.0
    stale: bool = False

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class CrowdingCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = ""
    score: float | None = None
    bucket: CrowdingBucket = "unknown"
    margin_change: float | None = None
    northbound: float | None = None
    main_net: float | None = None
    lhb_concentration: float | None = None
    turnover_pct: float | None = None
    put_call: float | None = None
    short_interest: float | None = None
    retail_heat: float | None = None
    missing: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class GraphImpact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chain_id: str
    chain_name: str = ""
    kind: str = "company"  # company | geo
    symbol: str = ""
    name: str = ""
    hop: int = 1
    direction: str = "同向"
    role: str = ""

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class Emergency(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active: bool = False
    reasons: list[str] = Field(default_factory=list)
    playbook: list[str] = Field(default_factory=list)
    affected: list[str] = Field(default_factory=list)
    hedges: list[str] = Field(default_factory=list)
    freeze_adds: bool = False

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class ExpiryAlert(BaseModel):
    symbol: str
    headline: str = ""
    time_stop_on: date | None = None
    reason: str = "未兑现，撤"
    kind: str = "time_stop"  # time_stop | invalidation
