"""联网冒烟：默认 pytest 因 -m 'not network' 跳过。"""

from __future__ import annotations

import pytest


@pytest.mark.network
def test_akshare_one_daily_bar() -> None:
    import akshare as ak

    df = ak.fund_etf_hist_em(
        symbol="510300",
        period="daily",
        start_date="20260105",
        end_date="20260109",
        adjust="",
    )
    assert df is not None and len(df) >= 1


@pytest.mark.network
def test_yfinance_one_daily_bar() -> None:
    import yfinance as yf

    hist = yf.Ticker("AAPL").history(start="2026-01-05", end="2026-01-09")
    assert hist is not None and len(hist) >= 1
