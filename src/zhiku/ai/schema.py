"""研判任务包与写回的固定格式。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PACK_SCHEMA = "zhiku.review_pack.v1"
RESULT_SCHEMA = "zhiku.review_result.v1"

Stance = Literal["研究跟进", "观察", "回避"]


class CitedClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim: str
    cite: str

    @field_validator("cite")
    @classmethod
    def _cite_required(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("每条主张必须有 cite")
        return text


class Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str | None = None
    event_type: str = "其它"
    direction: int = 0
    surprise: float | None = None
    novelty: float | None = None
    cite: str = "缺数据"
    headline: str = ""

    @field_validator("direction", mode="before")
    @classmethod
    def _dir(cls, value: Any) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError):
            return 0
        return -1 if n < 0 else 1 if n > 0 else 0


class Judgment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    name: str = ""
    stance: Stance
    confidence: float
    invalidation: str
    time_stop_days: int = 20
    long_case: list[CitedClaim] = Field(default_factory=list)
    short_case: list[CitedClaim] = Field(default_factory=list)
    formula_score: float | None = None
    formula_rank: int | None = None
    backend: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip(cls, value: Any) -> float:
        try:
            n = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, n))


class RegimeAdvice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    style: str = "混合"
    value_or_growth: str = "缺数据"
    prefer_screens: list[str] = Field(default_factory=list)
    reason: str = ""
    cite: str = "缺数据"


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_name: str = Field(default=RESULT_SCHEMA, alias="schema")
    asof: str
    extractions: list[Extraction] = Field(default_factory=list)
    judgments: list[Judgment] = Field(default_factory=list)
    regime: RegimeAdvice | None = None
    backend: str = ""
    notes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", by_alias=True)
