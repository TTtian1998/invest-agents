"""新闻、事件与多空机会的结构化记录。缺字段保持空，不编造。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from zhiku.models import Market

NewsTag = Literal["事实", "情绪", "市场解读", "未经证实", "未经官方证实"]
Freshness = Literal["NEW", "已price-in", "旧闻翻炒"]
Reaction = Literal["未反应", "已反应", "过度反应"]
Direction = Literal["做多", "做空", "对冲"]
Tier = Literal["质量档", "交易档", "只观察"]
GroupName = Literal["持仓相关", "AI科技", "宏观", "硬日历", "地缘", "其它"]


class NewsItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    source: str
    source_kind: str = "media"  # official | media | social | rating | lhb | rss
    symbol: str | None = None
    name: str = ""
    market: Market | None = None
    headline: str
    url: str = ""
    published: datetime | None = None
    summary: str = ""
    catalyst: str = "other"
    raw_body: str = ""
    source_grade: int | None = None
    source_grade_label: str = ""
    official_url: str = ""
    attribution: str = ""
    exclusive: bool = False
    paywall: bool = False


class InstitutionReview(BaseModel):
    """三级信源必须做辩证审查。缺数据保留空，不编造命中率。"""

    model_config = ConfigDict(extra="ignore")

    institution: str = ""
    institution_stance: str = ""
    track_record: str = ""
    key_assumptions: str = ""
    consensus_gap: str = ""
    counter_view: str = ""

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class ClusteredNews(BaseModel):
    """去重聚合后的一条事件。"""

    fingerprint: str
    symbol: str | None = None
    name: str = ""
    market: Market | None = None
    headline: str
    catalyst: str = "other"
    items: list[NewsItem] = Field(default_factory=list)
    tags: list[NewsTag] = Field(default_factory=list)
    freshness: Freshness = "NEW"
    groups: list[GroupName] = Field(default_factory=list)
    reaction: Reaction | None = None
    excess_return: float | None = None
    volume_ratio: float | None = None
    main_net: float | None = None
    reaction_missing: str | None = None
    chain_ids: list[str] = Field(default_factory=list)
    source_grade: int | None = None
    source_grade_label: str = ""
    unofficial: bool = False
    exclusive: bool = False
    official_url: str = ""
    official_followup: str | None = None
    institution_review: InstitutionReview | None = None


class MarketReaction(BaseModel):
    symbol: str
    on: date
    excess_return: float | None = None
    volume_ratio: float | None = None
    main_net: float | None = None
    missing: str | None = None


class Opportunity(BaseModel):
    symbol: str
    name: str
    direction: Direction
    tier: Tier
    score: float
    catalyst: str
    trigger_headline: str
    sources: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    price: float | None = None
    lot_notional: float | None = None
    entry_logic: str
    stop: float | None = None
    max_hold_days: int | None = None
    holding_related: bool | None = None
    overlap_screens: list[str] = Field(default_factory=list)
    short_vehicle: str | None = None
    special_permission: str | None = None
    risk: str
    reaction: Reaction | None = None
    hist_winrate: float | None = None
    event_window: str | None = None
    chains: list[str] = Field(default_factory=list)
    analysis: dict | None = None
    source_grade: int | None = None
    source_grade_label: str = ""
    institution_review: dict | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class SourceError(BaseModel):
    source: str
    reason: str
