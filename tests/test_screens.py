"""选股公式：用固定的虚构小样本，不访问网络。"""

from __future__ import annotations

from zhiku.models import Market
from zhiku.pipeline.settings import (
    DividendSettings,
    GreenblattSettings,
    MultifactorSettings,
    PipelineSettings,
    PiotroskiSettings,
    QualitySettings,
    UniverseSettings,
)
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import earnings_yield, roc, screen_greenblatt
from zhiku.screens.piotroski import fscore, screen_piotroski
from zhiku.screens.quality import apply_quality
from zhiku.screens.dividend import screen_dividend
from zhiku.screens.multifactor import screen_multifactor
from zhiku.screens.runner import run_screens


def _row(**kwargs) -> Fundamentals:
    base = dict(
        symbol="X",
        market=Market.CN,
        name="虚构",
        instrument="stock",
        industry_leader=True,
        st=False,
        mktcap=1e11,
        amount_avg=5e8,
        pe_percentile=0.3,
        pb_percentile=0.3,
        ebit=10.0,
        nwc=4.0,
        nfa=6.0,
        ev=50.0,
        ocf=12.0,
        net_income=8.0,
        debt_asset=0.4,
        roe=0.2,
        roe_3y=[0.18, 0.19, 0.2],
        roa=0.1,
        roa_prev=0.08,
        gross_margin=0.4,
        gross_margin_prev=0.35,
        gross_margin_3y=[0.33, 0.35, 0.4],
        lt_debt_asset=0.2,
        lt_debt_asset_prev=0.25,
        current_ratio=1.5,
        current_ratio_prev=1.2,
        shares_issued=False,
        asset_turnover=0.8,
        asset_turnover_prev=0.7,
        dividend_yield=0.04,
        dividend_years=5,
        momentum_12_1=0.1,
        volatility=0.2,
    )
    base.update(kwargs)
    return Fundamentals.model_validate(base)


def test_greenblatt_ranks_higher_roc_and_ey_first() -> None:
    a = _row(symbol="A", ebit=20, nwc=5, nfa=5, ev=80)   # ROC=2.0 EY=0.25
    b = _row(symbol="B", ebit=10, nwc=5, nfa=5, ev=80)   # ROC=1.0 EY=0.125
    c = _row(symbol="C", ebit=10, nwc=20, nfa=20, ev=200)  # ROC=0.25 EY=0.05
    assert roc(a) == 2.0
    assert abs(float(earnings_yield(a)) - 0.25) < 1e-9
    ranked = screen_greenblatt([a, b, c], GreenblattSettings(top_n=10))
    assert [r.symbol for r in ranked] == ["A", "B", "C"]
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_piotroski_nine_and_zero() -> None:
    perfect = _row(symbol="P")
    raw, items, missing = fscore(perfect)
    assert missing == []
    assert raw == 9
    assert all(v == 1 for v in items.values())

    dud = _row(
        symbol="D",
        roa=-0.1,
        roa_prev=0.1,
        ocf=-1,
        net_income=5,
        lt_debt_asset=0.4,
        lt_debt_asset_prev=0.2,
        current_ratio=0.8,
        current_ratio_prev=1.2,
        shares_issued=True,
        gross_margin=0.2,
        gross_margin_prev=0.3,
        asset_turnover=0.4,
        asset_turnover_prev=0.5,
    )
    raw2, _, _ = fscore(dud)
    assert raw2 == 0
    kept = screen_piotroski([perfect, dud], PiotroskiSettings(min_score=7))
    assert [r.symbol for r in kept] == ["P"]
    assert abs(kept[0].score - 100.0) < 1e-9


def test_quality_rejects_st_smallcap_and_high_pe() -> None:
    q, u = QualitySettings(), UniverseSettings()
    st = apply_quality(_row(st=True), q, u)
    assert not st.passed
    small = apply_quality(_row(mktcap=1e9, industry_leader=True), q, u)  # 10 亿 < 80 亿
    assert not small.passed
    high = apply_quality(_row(pe_percentile=0.95), q, u)
    assert not high.passed
    ok = apply_quality(_row(), q, u)
    assert ok.passed


def test_dividend_requires_yield_and_low_percentile() -> None:
    cheap = _row(symbol="CHEAP", pe_percentile=0.2, pb_percentile=0.2, dividend_yield=0.05, dividend_years=8)
    expensive = _row(symbol="EXP", pe_percentile=0.7, pb_percentile=0.7, dividend_yield=0.05, dividend_years=8)
    lowy = _row(symbol="LOWY", pe_percentile=0.2, pb_percentile=0.2, dividend_yield=0.01, dividend_years=8)
    out = screen_dividend([cheap, expensive, lowy], DividendSettings())
    assert [r.symbol for r in out] == ["CHEAP"]


def test_multifactor_prefers_quality_and_low_vol() -> None:
    good = _row(symbol="G", roe=0.3, momentum_12_1=0.2, volatility=0.1)
    bad = _row(symbol="B", roe=0.05, momentum_12_1=-0.1, volatility=0.5)
    out = screen_multifactor([good, bad], MultifactorSettings())
    assert out[0].symbol == "G"


def test_runner_excludes_etf() -> None:
    stock = _row(symbol="000333")
    etf = _row(symbol="510300", instrument="etf")
    bundle = run_screens([stock, etf], PipelineSettings(), names=["greenblatt"])
    assert "000333" in bundle.quality_passed
    assert "510300" not in bundle.quality_passed
