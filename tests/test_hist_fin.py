"""历史季报/分红/预告映射：公告日 PIT，不插值。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from zhiku.data.detail_fin import (
    map_ak_dividend_history,
    map_baostock_dividend,
    map_baostock_forecast,
    map_baostock_statement,
)
from zhiku.data.store import MarketStore
from zhiku.factors.dataset import load_revision_features


def test_map_baostock_profit_uses_pubdate() -> None:
    row = {
        "statDate": "2023-03-31",
        "pubDate": "2023-04-28",
        "netProfit": "1.2e9",
        "epsTTM": "1.5",
        "totalShare": "8e8",
        "MBRevenue": "5e9",
        "roeAvg": "12.5",
    }
    mapped = map_baostock_statement("600519", "income", row)
    assert mapped is not None
    assert mapped["announced"] == date(2023, 4, 28)
    assert mapped["period"] == date(2023, 3, 31)
    assert mapped["announced_is_period"] is False
    assert mapped["payload"]["每股收益"] == 1.5
    assert mapped["payload"]["总股本"] == 8e8


def test_map_baostock_missing_pubdate_marks_period() -> None:
    row = {"statDate": "2022-12-31", "netProfit": "10", "epsTTM": "0.3"}
    mapped = map_baostock_statement("000333", "income", row)
    assert mapped is not None
    assert mapped["announced_is_period"] is True
    assert mapped["announced"] == date(2022, 12, 31)


def test_map_baostock_forecast_and_dividend() -> None:
    fc = map_baostock_forecast(
        "000333",
        {
            "profitForcastExpStatDate": "2024-12-31",
            "profitForcastExpPubDate": "2025-01-15",
            "profitForcastChgPctUp": "20",
            "profitForcastChgPctDwn": "10",
            "profitForcastType": "预增",
        },
    )
    assert fc is not None
    assert fc["kind"] == "forecast"
    assert abs(fc["payload"]["increment"] - 0.15) < 1e-9
    div = map_baostock_dividend(
        "000333",
        {
            "dividOperateDate": "2023-06-10",
            "dividPlanAnnounceDate": "2023-04-01",
            "dividCashPsBeforeTax": "0.8",
        },
    )
    assert div is not None
    assert div["ex_date"] == date(2023, 6, 10)
    assert div["announced"] == date(2023, 4, 1)
    assert div["amount"] == 0.8


def test_map_ak_dividend_history() -> None:
    df = pd.DataFrame(
        [
            {"公告日期": "2022-05-01", "除权除息日": "2022-06-01", "派息": 3.0, "股息率": 2.5},
            {"公告日期": "2021-05-01", "除权除息日": "2021-06-01", "派息": 2.0},
        ]
    )
    rows = map_ak_dividend_history("600519", df)
    assert not isinstance(rows, Exception)
    assert len(rows) == 2
    assert rows[0]["announced"] == date(2022, 5, 1)
    assert rows[1]["ex_date"] == date(2021, 6, 1)


def test_revision_features_ignore_future_forecast(tmp_path) -> None:
    store = MarketStore(tmp_path / "data")
    store.upsert_listings([{"symbol": "000333", "market": "CN", "instrument": "stock"}])
    store.upsert_statements(
        [
            {
                "symbol": "000333",
                "kind": "forecast",
                "period": date(2024, 12, 31),
                "announced": date(2025, 1, 10),
                "payload": {"profit_forecast": 10.0},
            },
            {
                "symbol": "000333",
                "kind": "forecast",
                "period": date(2025, 6, 30),
                "announced": date(2025, 7, 20),
                "payload": {"profit_forecast": 15.0},
            },
            {
                "symbol": "000333",
                "kind": "income",
                "period": date(2024, 12, 31),
                "announced": date(2025, 3, 31),
                "payload": {"净利润": 12.0},
            },
        ]
    )
    cal = pd.to_datetime(["2025-02-01", "2025-08-01"])
    panel = load_revision_features(store, ["000333"], cal)
    early = panel.xs(pd.Timestamp("2025-02-01"), level="date").iloc[0]
    late = panel.xs(pd.Timestamp("2025-08-01"), level="date").iloc[0]
    assert pd.isna(early["rev_fy1"])
    assert abs(float(late["rev_fy1"]) - 0.5) < 1e-9
    assert abs(float(late["rev_surprise"]) - (12.0 - 15.0) / 15.0) < 1e-9
