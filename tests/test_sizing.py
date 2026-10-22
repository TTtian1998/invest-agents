"""半凯利与止损：用合成价格，不读真实持仓。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhiku.portfolio.sizing import compare_sizing, half_kelly


def test_half_kelly_zero_without_edge() -> None:
    assert half_kelly(0.5, 1.0) == 0.0
    assert 0 < half_kelly(0.6, 2.0) < 0.5


def test_compare_sizing_returns_three_books() -> None:
    idx = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        {
            "AAA": 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, len(idx)))),
            "BBB": 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, len(idx)))),
            "CCC": 15 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx)))),
        },
        index=idx,
    )
    out = compare_sizing(px, ["AAA", "BBB", "CCC"], name_cap=0.1, stop_pct=0.08)
    assert "equal_weight" in out and "half_kelly_cap" in out and "half_kelly_stop" in out
    assert out["equal_weight"].annualized is not None
