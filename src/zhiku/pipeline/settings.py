"""流水线 YAML 配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from zhiku.models import Market


class UniverseSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exclude_st: bool = True
    exclude_etf: bool = True
    min_mktcap_yi: float = 80.0
    min_amount_cny: dict[str, float] = Field(
        default_factory=lambda: {"CN": 3e8, "HK": 1e8, "US": 1e8}
    )

    def min_amount(self, market: Market) -> float:
        return float(self.min_amount_cny.get(market.value, 1e8))


class QualitySettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    roe_3y_min: float = 0.12
    ocf_positive: bool = True
    max_debt_asset: float = 0.70
    pe_percentile_max: float = 0.80
    pb_percentile_max: float = 0.80
    require_leader_or_roe: bool = True


class GreenblattSettings(BaseModel):
    top_n: int = 30
    min_score: float = 0.0


class PiotroskiSettings(BaseModel):
    min_score: int = 7
    top_n: int = 30


class DividendSettings(BaseModel):
    pe_percentile_max: float = 0.40
    pb_percentile_max: float = 0.40
    min_yield: float = 0.03
    min_consecutive_years: int = 3
    top_n: int = 30


class MultifactorSettings(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "roe": 0.25,
            "gm_stability": 0.15,
            "ocf_ni": 0.20,
            "momentum_12_1": 0.25,
            "low_vol": 0.15,
        }
    )
    top_n: int = 30


class MergeSettings(BaseModel):
    formula: float = 0.6
    qlib: float = 0.4


class BacktestSettings(BaseModel):
    rebalance: str = "monthly"
    equal_weight: bool = True
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    rf: float = 0.02
    benchmark: str = "510300"
    min_sharpe: float = 0.0
    require_beat_benchmark: bool = False
    top_n: int = 3


class AISettings(BaseModel):
    top_n: int = 15
    masters: list[str] = Field(default_factory=lambda: ["buffett", "munger", "graham", "lynch"])


class NewsScoreWeights(BaseModel):
    catalyst: float = 0.22
    hist_winrate: float = 0.18
    price_confirm: float = 0.18
    quality: float = 0.15
    correlation: float = 0.09
    source: float = 0.18


class NewsSourceFlags(BaseModel):
    model_config = ConfigDict(extra="ignore")

    eastmoney_news: bool = True
    cls: bool = True
    cninfo: bool = True
    eastmoney_hot: bool = True
    lhb: bool = True
    ratings: bool = True
    yfinance_news: bool = True
    finnhub: bool = True
    rss: bool = True
    apewisdom: bool = True
    twitter: bool = True
    official_rss: bool = True
    edgar: bool = True


class NewsSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    sources: NewsSourceFlags = Field(default_factory=NewsSourceFlags)
    rss_feeds: list[str] = Field(
        default_factory=lambda: [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.reuters.com/reuters/businessNews",
        ]
    )
    official_feeds: dict[str, str] = Field(
        default_factory=lambda: {
            "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
            "bls": "https://www.bls.gov/feed/bls_latest.rss",
            "ecb": "https://www.ecb.europa.eu/rss/press.html",
        }
    )
    stop_pct: float = 0.03
    max_hold_days: dict[str, int] = Field(
        default_factory=lambda: {"质量档": 20, "交易档": 5, "只观察": 3}
    )
    max_quality: int = 5
    max_trade: int = 5
    max_watch: int = 8
    max_notional_cny: float = 50000
    low_corr_bonus: float = 0.08
    poll_seconds: int = 300
    score_weights: NewsScoreWeights = Field(default_factory=NewsScoreWeights)
    unused_thr: float = 0.01
    over_thr: float = 0.04


class EventWindowRule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    before: int = 1
    after: int = 1


class MacroSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    freeze_adds_in_window: bool = False
    window_hint: str = "事件窗口内，新开仓降仓位/止损放宽或等落地"
    windows: dict[str, EventWindowRule] = Field(
        default_factory=lambda: {
            "FOMC": EventWindowRule(before=1, after=1),
            "US_NFP": EventWindowRule(before=1, after=1),
            "US_CPI": EventWindowRule(before=1, after=1),
            "US_CORE_CPI": EventWindowRule(before=1, after=1),
            "US_PCE": EventWindowRule(before=1, after=1),
            "default": EventWindowRule(before=1, after=1),
        }
    )
    min_window_importance: int = 4
    gdelt_enabled: bool = False
    fedwatch_approx: bool = True
    horizon_days: int = 7

    def window_for(self, event_type: str) -> dict:
        rule = self.windows.get(event_type) or self.windows.get("default") or EventWindowRule()
        return {"before": rule.before, "after": rule.after}


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    greenblatt: GreenblattSettings = Field(default_factory=GreenblattSettings)
    piotroski: PiotroskiSettings = Field(default_factory=PiotroskiSettings)
    dividend: DividendSettings = Field(default_factory=DividendSettings)
    multifactor: MultifactorSettings = Field(default_factory=MultifactorSettings)
    merge: MergeSettings = Field(default_factory=MergeSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    ai: AISettings = Field(default_factory=AISettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    macro: MacroSettings = Field(default_factory=MacroSettings)


DEFAULT_YAML = Path(__file__).resolve().parent / "default.yaml"


def load_pipeline_settings(path: Path | None = None) -> PipelineSettings:
    data: dict[str, Any] = {}
    if DEFAULT_YAML.is_file():
        loaded = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    if path is not None and path.is_file():
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(override, dict):
            data = _deep_merge(data, override)
    return PipelineSettings.model_validate(data)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
