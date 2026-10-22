"""经典公式：虚构样本，缺字段不估算。"""

from __future__ import annotations

from zhiku.pipeline.settings import ClassicScreenSettings, PipelineSettings
from zhiku.screens.classics import graham_number_ratio, peg, screen_graham, screen_lynch_peg, screen_canslim, screen_livermore, screen_fisher
from zhiku.screens.runner import SCREEN_FUNCS, run_screens
from tests.test_screens import _row


def test_graham_number_and_screen_order() -> None:
    cheap = _row(symbol="C", pe=8, pb=1.0, ncav=90, mktcap=100)
    dear = _row(symbol="D", pe=30, pb=8, ncav=5, mktcap=100)
    assert graham_number_ratio(cheap) is not None and graham_number_ratio(cheap) < 1
    assert graham_number_ratio(dear) > 1
    ranked = screen_graham([cheap, dear], ClassicScreenSettings(top_n=10))
    assert ranked[0].symbol == "C"


def test_peg_requires_positive_growth() -> None:
    ok = _row(symbol="G", pe=12, eps_growth=0.20)
    no = _row(symbol="N", pe=12, eps_growth=None)
    assert peg(ok) == 12 / 20
    assert peg(no) is None
    out = screen_lynch_peg([ok, no], ClassicScreenSettings())
    assert [r.symbol for r in out] == ["G"]


def test_canslim_missing_letters_stay_missing() -> None:
    row = _row(symbol="S", roa=0.12, roa_prev=0.08, eps_growth=0.2, listed_years=5, shares_issued=False, momentum_12_1=0.1)
    out = screen_canslim([row], ClassicScreenSettings())
    assert out and "I" in out[0].missing and "M" in out[0].missing
    assert out[0].breakdown["C"] == 1


def test_livermore_prefers_uptrend_near_high() -> None:
    up = _row(symbol="U", momentum_12_1=0.3, dist_to_high=-0.04)
    down = _row(symbol="D", momentum_12_1=-0.2, dist_to_high=-0.4)
    out = screen_livermore([up, down], ClassicScreenSettings())
    assert out[0].symbol == "U"


def test_fisher_uses_roe_and_margin_stability() -> None:
    stable = _row(symbol="F", roe=0.25, roe_3y=[0.24, 0.25, 0.26], gross_margin_3y=[0.40, 0.41, 0.40], rd_to_sales=0.05)
    choppy = _row(symbol="X", roe=0.08, roe_3y=[0.02, 0.20, 0.03], gross_margin_3y=[0.10, 0.40, 0.05], rd_to_sales=None)
    out = screen_fisher([stable, choppy], ClassicScreenSettings())
    assert out[0].symbol == "F"
    assert out[0].breakdown["rd_to_sales"] == 0.05


def test_runner_registers_new_screens() -> None:
    for name in ("graham", "lynch_peg", "canslim", "livermore", "fisher"):
        assert name in SCREEN_FUNCS
    bundle = run_screens(
        [_row(symbol="000333", pe=10, pb=1.2, eps_growth=0.2, ncav=8e10, dist_to_high=-0.05)],
        PipelineSettings(),
        names=["graham", "lynch_peg"],
    )
    assert bundle.screens["graham"] and bundle.screens["lynch_peg"]
