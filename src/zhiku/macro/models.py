"""宏观事件、预期差、剧本与窗口的结构化记录。缺字段保持空，不编造。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Importance = Literal[1, 2, 3, 4, 5]
Scenario = Literal["hawkish", "inline", "dovish"]
RateDecision = Literal["hike", "cut", "hold"]
GeoTemp = Literal["低", "中", "高", "缺数据"]

ANCHOR_ASSETS: tuple[tuple[str, str], ...] = (
    ("US2Y", "美债2Y"),
    ("US10Y", "美债10Y"),
    ("DXY", "美元指数"),
    ("USDCNH", "离岸人民币"),
    ("GOLD", "黄金"),
    ("WTI", "WTI原油"),
    ("BRENT", "Brent原油"),
    ("COPPER", "铜"),
    ("NQ", "纳指期货"),
    ("ES", "标普期货"),
    ("HSTECH", "恒生科技"),
    ("CSI300", "沪深300"),
    ("VIX", "VIX"),
)

ANCHOR_LABELS: dict[str, str] = dict(ANCHOR_ASSETS)

# yfinance 代码。没有对应 ticker 的不进 live 锚定。
ANCHOR_YF: dict[str, str] = {
    "US2Y": "^IRX",
    "US10Y": "^TNX",
    "DXY": "DX-Y.NYB",
    "USDCNH": "USDCNH=X",
    "GOLD": "GC=F",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "COPPER": "HG=F",
    "NQ": "NQ=F",
    "ES": "ES=F",
    "HSTECH": "^HSTECH",
    "CSI300": "000300.SS",
    "VIX": "^VIX",
}

EVENT_TYPE_CN: dict[str, str] = {
    "US_NFP": "美国非农",
    "US_UNEMP": "美国失业率",
    "US_CPI": "美国CPI",
    "US_CORE_CPI": "美国核心CPI",
    "US_PCE": "美国PCE",
    "US_PPI": "美国PPI",
    "US_GDP": "美国GDP",
    "US_ISM_MFG": "美国ISM制造业PMI",
    "US_ISM_SVC": "美国ISM服务业PMI",
    "US_RETAIL": "美国零售销售",
    "FOMC": "FOMC利率决议",
    "FOMC_DOTS": "FOMC点阵图",
    "FOMC_SPEECH": "FOMC讲话",
    "CN_CPI": "中国CPI",
    "CN_PPI": "中国PPI",
    "CN_PMI": "中国官方PMI",
    "CN_SF": "中国社融",
    "CN_M2": "中国M2",
    "CN_GDP": "中国GDP",
    "CN_LPR": "LPR",
    "CN_RRR": "降准",
    "CN_RATE": "降息",
    "CN_MEETING": "重要会议",
    "ECB": "欧央行决议",
    "BOJ": "日央行决议",
    "OPEC": "OPEC+会议",
    "EIA": "EIA原油库存",
}

# 数值高于预期偏鹰的数据；失业率等反向。
HAWKISH_IF_HIGHER = frozenset(
    {
        "US_NFP",
        "US_CPI",
        "US_CORE_CPI",
        "US_PCE",
        "US_PPI",
        "US_GDP",
        "US_ISM_MFG",
        "US_ISM_SVC",
        "US_RETAIL",
        "CN_CPI",
        "CN_PPI",
        "CN_PMI",
        "CN_SF",
        "CN_M2",
        "CN_GDP",
        "EIA",  # 库存增加偏空油价，不直接映射鹰鸽；脚本里单独处理
    }
)
HAWKISH_IF_LOWER = frozenset({"US_UNEMP"})
RATE_EVENTS = frozenset({"FOMC", "ECB", "BOJ", "CN_LPR", "CN_RATE", "CN_RRR"})
SCENARIO_CN: dict[str, str] = {"hawkish": "偏鹰", "inline": "符合预期", "dovish": "偏鸽"}


class HorizonMove(BaseModel):
    model_config = ConfigDict(extra="ignore")

    t0: float | None = None
    t1: float | None = None
    t5: float | None = None
    missing: str | None = None


class MacroEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    event_type: str
    name: str = ""
    region: str = ""
    importance: int = 3
    scheduled_at: datetime | None = None
    previous: float | None = None
    consensus: float | None = None
    actual: float | None = None
    unit: str = ""
    sigma: float | None = None
    surprise_z: float | None = None
    surprise_missing: str | None = None
    scenario: Scenario | None = None
    decision: RateDecision | None = None
    tone: Scenario | None = None
    sessions: list[str] = Field(default_factory=list)
    reactions: dict[str, HorizonMove] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name or EVENT_TYPE_CN.get(self.event_type, self.event_type)


class ScriptReaction(BaseModel):
    asset: str
    label: str
    avg_t0: float | None = None
    avg_t1: float | None = None
    avg_t5: float | None = None
    winrate_t0: float | None = None
    n: int = 0


class ScenarioPlaybook(BaseModel):
    scenario: Scenario
    n: int = 0
    assets: list[ScriptReaction] = Field(default_factory=list)
    winners: list[str] = Field(default_factory=list)
    losers: list[str] = Field(default_factory=list)
    holding_impact: list[str] = Field(default_factory=list)


class EventScript(BaseModel):
    event_id: str
    event_type: str
    name: str
    scheduled_at: datetime | None = None
    playbooks: list[ScenarioPlaybook] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EventWindow(BaseModel):
    event_id: str
    event_type: str
    name: str
    start: date
    end: date
    freeze_adds: bool = False
    hint: str = "事件窗口内，新开仓降仓位/止损放宽或等落地"
    importance: int = 3

    def covers(self, on: date) -> bool:
        return self.start <= on <= self.end


class RateExpectation(BaseModel):
    meeting: str = ""
    asof: str = ""
    implied_rate: float | None = None
    current_rate: float | None = None
    hike_25_prob: float | None = None
    cut_25_prob: float | None = None
    hold_prob: float | None = None
    source: str = "缺数据"
    note: str = ""


class AnchorSnapshot(BaseModel):
    asset: str
    label: str
    price: float | None = None
    chg: float | None = None
    missing: str | None = None


class TransmissionChain(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    keywords: list[str] = Field(default_factory=list)
    commodities: list[str] = Field(default_factory=list)
    a_share: list[str] = Field(default_factory=list)
    h_share: list[str] = Field(default_factory=list)
    us: list[str] = Field(default_factory=list)
    etfs: list[str] = Field(default_factory=list)
    risk_on_off: str = "risk_off"
    hist_path_note: str = ""

    @field_validator("etfs", "keywords", "commodities", "a_share", "h_share", "us", mode="before")
    @classmethod
    def _stringify(cls, value):
        if not value:
            return []
        return [str(x) for x in value]


class GeoEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    on: date | None = None
    headline: str
    chain_id: str | None = None
    source: str = "fixture"


class GeoHeat(BaseModel):
    asof: str = ""
    gpr: float | None = None
    gpr_percentile: float | None = None
    temperature: GeoTemp = "缺数据"
    active_chains: list[str] = Field(default_factory=list)
    recent_events: list[GeoEvent] = Field(default_factory=list)
    missing: str | None = None


class ScriptHit(BaseModel):
    event_id: str
    event_type: str
    name: str
    scenario: Scenario | None = None
    matched: int = 0
    total: int = 0
    hit: bool | None = None
    missing: str | None = None


class HistorySample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    scenario: Scenario
    released: date | None = None
    surprise_z: float | None = None
    reactions: dict[str, HorizonMove] = Field(default_factory=dict)
    winners: list[str] = Field(default_factory=list)
    losers: list[str] = Field(default_factory=list)
