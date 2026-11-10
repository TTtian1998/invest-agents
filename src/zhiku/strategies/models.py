"""Immutable, JSON-serializable contracts for the research-only ETF pilot."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrategySpec(BaseModel):
    """Declared ex-ante parameters; no data-driven fitting or live execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    strategy_id: str = Field(default="etf_momentum", min_length=1)
    version: str = Field(default="1", min_length=1)
    universe: tuple[str, ...] = Field(min_length=1)
    universe_known_at: date
    source_mode: Literal["demo", "research"] = "demo"
    lookback: int = Field(default=20, gt=0, strict=True)
    top_n: int = Field(default=2, gt=0, strict=True)
    rebalance_every: int = Field(default=20, gt=0, strict=True)
    max_weight: float = Field(default=0.3, gt=0, le=1)
    max_gross: float = Field(default=0.8, gt=0, le=1)
    initial_cash: float = Field(default=100_000, gt=0)
    cost_bps: float = Field(default=10, ge=0)
    slippage_bps: float = Field(default=5, ge=0)
    lot_size: int = Field(default=100, gt=0, strict=True)
    min_momentum: float = Field(default=0, ge=-1)

    @field_validator("universe")
    @classmethod
    def unique_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not s or s.strip() != s for s in value):
            raise ValueError("universe identifiers must be nonempty and have no surrounding whitespace")
        if len(set(value)) != len(value):
            raise ValueError("universe identifiers must be unique")
        return tuple(sorted(value))

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class StrategyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    asof: date
    method: Literal["momentum", "equal_weight"]
    signals: dict[str, float]
    target_weights: dict[str, float]
    cash_weight: float
    rebalance_schedule: dict[str, Any]
    evidence: dict[str, Any]
    fingerprint: str
    spec_fingerprint: str
