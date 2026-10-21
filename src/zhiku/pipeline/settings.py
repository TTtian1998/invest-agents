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
    require_quality: bool = False


class PiotroskiSettings(BaseModel):
    min_score: int = 7
    top_n: int = 30
    require_quality: bool = False


class DividendSettings(BaseModel):
    pe_percentile_max: float = 0.40
    pb_percentile_max: float = 0.40
    min_yield: float = 0.03
    min_consecutive_years: int = 3
    top_n: int = 30
    require_quality: bool = False


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
    require_quality: bool = True


class ClassicScreenSettings(BaseModel):
    top_n: int = 30
    min_score: float = 0.0
    require_quality: bool = False


class SizingSettings(BaseModel):
    enabled: bool = False
    name_cap: float = 0.10
    stop_pct: float = 0.08


class MergeSettings(BaseModel):
    formula: float = 0.90
    qlib: float = 0.10
    qlib_floor: float = 0.10
    qlib_cap: float = 0.40
    rank_ic_threshold: float = 0.03
    icir_threshold: float = 0.20
    formula_first: bool = True


class BacktestSettings(BaseModel):
    rebalance: str = "monthly"
    equal_weight: bool = True
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    rf: float = 0.02
    benchmark: str = "universe_ew"
    min_sharpe: float = 0.0
    require_beat_benchmark: bool = True
    require_sharpe_ge_benchmark: bool = True
    top_n: int = 3
    benchmark_symbols: list[str] = Field(default_factory=lambda: ["510300", "000300", "000906"])
    sanity_max_ann_gap: float = 0.25
    pit_universe: str = "auto"
    require_half_excess: bool = True
    fill: str = "next_open"
    n_trials: int = 9
    min_coverage: float = 0.80


class AISettings(BaseModel):
    top_n: int = 15
    judge_n: int = 10
    backend: str = "auto"
    masters: list[str] = Field(
        default_factory=lambda: [
            "buffett",
            "munger",
            "graham",
            "lynch",
            "fisher",
            "livermore",
            "oneil",
            "marks",
        ]
    )


class NewsScoreWeights(BaseModel):
    """综合打分权重。预期差驱动，不按消息「好坏」打分。"""

    surprise: float = 0.16
    novelty: float = 0.10
    decay: float = 0.08
    hist_winrate: float = 0.12
    crowding: float = 0.10
    source: float = 0.12
    price_confirm: float = 0.10
    quality: float = 0.12
    correlation: float = 0.08
    catalyst: float = 0.02  # 残留项，不代表利好/利空

    def total(self) -> float:
        return (
            self.surprise
            + self.novelty
            + self.decay
            + self.hist_winrate
            + self.crowding
            + self.source
            + self.price_confirm
            + self.quality
            + self.correlation
            + self.catalyst
        )


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
            "https://www.caixin.com/rss/latest.xml",
            "https://www.yicai.com/rss/news.xml",
        ]
    )
    official_feeds: dict[str, str] = Field(
        default_factory=lambda: {
            "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
            "bls": "https://www.bls.gov/feed/bls_latest.rss",
            "bea": "https://www.bea.gov/rss.xml",
            "ecb": "https://www.ecb.europa.eu/rss/press.html",
            "pbc": "http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/11040/index1.html",
            "stats_cn": "https://www.stats.gov.cn/sj/zxfb/rss.xml",
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
    x_accounts_file: str = "config/x_accounts.yaml"
    x_max_posts: int = 8
    x_lookback_hours: int = 36
    x_throttle_seconds: float = 0.8
    x_resonance_min: int = 3
    x_extreme_share: float = 0.80
    x_fomc_boost: float = 1.35


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
    window_event_types: list[str] = Field(
        default_factory=lambda: [
            "US_NFP",
            "US_UNEMP",
            "US_CPI",
            "US_CORE_CPI",
            "US_PCE",
            "US_PPI",
            "US_GDP",
            "US_RETAIL",
            "US_ISM_MFG",
            "FOMC",
            "FOMC_DOTS",
            "ECB",
            "BOJ",
            "BOE",
            "SNB",
            "RBA",
            "PBOC",
            "CN_CPI",
            "CN_PPI",
            "CN_GDP",
            "CN_SF",
            "CN_LPR",
            "CN_M2",
            "CN_PMI",
            "CN_RRR",
            "CN_RATE",
        ]
    )

    def window_for(self, event_type: str) -> dict:
        rule = self.windows.get(event_type) or self.windows.get("default") or EventWindowRule()
        return {"before": rule.before, "after": rule.after}


class DataSettings(BaseModel):
    """真实数据层：股票池、同步窗口、港美名单。"""

    model_config = ConfigDict(extra="ignore")

    universe: str = "csi800"  # all / csi800 / csi300 / ...
    years: int = 3
    sleep_seconds: float = 0.2
    retries: int = 3
    include_global: bool = True
    hk_lists: list[str] = Field(default_factory=lambda: ["hsi", "hstech"])
    us_lists: list[str] = Field(default_factory=lambda: ["sp500", "ndx"])
    extra_symbols: list[str] = Field(default_factory=lambda: ["510300"])
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    incremental_lookback_days: int = 2
    detail_financials: bool = True
    hist_financials: bool = True


class EventSettings(BaseModel):
    """第 6/7 块事件驱动层：预期差、时效、拥挤、传导、止损、应急。"""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    half_life_hours: dict[str, float] = Field(
        default_factory=lambda: {
            "earnings_beat": 48.0,
            "earnings_preview": 72.0,
            "guidance": 48.0,
            "buyback": 72.0,
            "policy": 24.0,
            "sentiment_hot": 6.0,
            "lhb": 12.0,
            "rating_up": 36.0,
            "rating_down": 36.0,
            "default": 36.0,
        }
    )
    time_stop_days: dict[str, int] = Field(
        default_factory=lambda: {
            "earnings_beat": 5,
            "earnings_preview": 10,
            "policy": 3,
            "buyback": 10,
            "sentiment_hot": 2,
            "lhb": 3,
            "rating_up": 5,
            "rating_down": 5,
            "default": 5,
        }
    )
    crowding_high: float = 0.70
    crowding_low: float = 0.30
    vix_spike: float = 0.20
    index_intraday_drop: float = 0.03
    emergency_keywords: list[str] = Field(
        default_factory=lambda: [
            "冲突",
            "制裁",
            "监管",
            "袭击",
            "战争",
            "熔断",
            "禁运",
            "sanction",
            "war",
            "halt",
        ]
    )
    emergency_min_sources: int = 2
    require_price_confirm: bool = True
    freeze_adds_on_emergency: bool = True
    alert_on_emergency: bool = True
    playbook: list[str] = Field(
        default_factory=lambda: [
            "先降风险，不扩大暴露",
            "不抄底",
            "等官方表态",
            "列出受影响持仓代码（不输出数量/成本）和对冲工具",
        ]
    )
    stale_after_halflives: float = 1.0


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    greenblatt: GreenblattSettings = Field(default_factory=GreenblattSettings)
    piotroski: PiotroskiSettings = Field(default_factory=PiotroskiSettings)
    dividend: DividendSettings = Field(default_factory=DividendSettings)
    multifactor: MultifactorSettings = Field(default_factory=MultifactorSettings)
    graham: ClassicScreenSettings = Field(default_factory=ClassicScreenSettings)
    lynch_peg: ClassicScreenSettings = Field(default_factory=ClassicScreenSettings)
    canslim: ClassicScreenSettings = Field(default_factory=ClassicScreenSettings)
    livermore: ClassicScreenSettings = Field(default_factory=ClassicScreenSettings)
    fisher: ClassicScreenSettings = Field(default_factory=ClassicScreenSettings)
    sizing: SizingSettings = Field(default_factory=SizingSettings)
    merge: MergeSettings = Field(default_factory=MergeSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    ai: AISettings = Field(default_factory=AISettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    macro: MacroSettings = Field(default_factory=MacroSettings)
    events: EventSettings = Field(default_factory=EventSettings)
    data: DataSettings = Field(default_factory=DataSettings)


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
