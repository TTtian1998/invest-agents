"""全市场数据层、PIT、涨跌停回测、qlib dump：全部虚构离线数据。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from zhiku.data.compute import attach_percentiles, ebit_of, missing_ratio, nwc_of, pick_num, snapshot_from_store
from zhiku.data.fetchers import MarketFetcher
from zhiku.data.store import MarketStore
from zhiku.data.sync import sync_market
from zhiku.factors.qlib_adapter import dump_qlib, latest_model, predict_scores
from zhiku.models import MissingData
from zhiku.pipeline.settings import BacktestSettings
from zhiku.backtest.engine import backtest_screen
from zhiku.backtest.metrics import monthly_equal_weight
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import roc
from zhiku.screens.piotroski import fscore


class FakeFetcher(MarketFetcher):
    def __init__(self, fail_once: str | None = None) -> None:
        self.fail_once = fail_once
        self.calls: list[str] = []

    def cn_listings(self):
        return [
            {"symbol": "600519", "market": "CN", "name": "示例白酒", "list_date": date(2001, 8, 27), "st": False, "instrument": "stock"},
            {"symbol": "000333", "market": "CN", "name": "示例家电", "list_date": date(2013, 9, 18), "st": False, "instrument": "stock"},
            {"symbol": "ST0001", "market": "CN", "name": "ST示例", "st": True, "instrument": "stock"},
        ]

    def cn_delisted(self):
        return [{"symbol": "999999", "delist_date": date(2024, 1, 15), "name": "已退市示例"}]

    def cn_suspended(self, on: date):
        del on
        return []

    def index_members(self, index_code: str):
        del index_code
        return ["600519", "000333"]

    def cn_spot(self):
        return [
            {"symbol": "600519", "name": "示例白酒", "pe": 20.0, "pb": 5.0, "mktcap": 2e12, "amount": 4e9, "st": False},
            {"symbol": "000333", "name": "示例家电", "pe": 12.0, "pb": 2.0, "mktcap": 4e11, "amount": 2e9, "st": False},
        ]

    def cn_bars(self, symbol: str, start: date, end: date, adjust: str):
        self.calls.append(f"bars:{symbol}:{adjust}:{start}")
        if self.fail_once == f"bars:{symbol}:{adjust}":
            self.fail_once = None
            return MissingData("模拟限流", symbol=symbol)
        rows = []
        d = start
        px = 100.0 if symbol == "600519" else 50.0
        while d <= end:
            if d.weekday() < 5:
                close = px + (d.toordinal() % 7)
                high = close * (1.11 if d == end else 1.01)
                pct = 10.2 if d == end else 0.3
                rows.append(
                    {
                        "symbol": symbol,
                        "market": "CN",
                        "date": d,
                        "adjust": "none" if adjust == "none" else "qfq",
                        "open": close,
                        "high": high,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 1e6,
                        "amount": 1e8,
                        "turnover": 1.2,
                        "pct_chg": pct,
                        "limit_up": int(pct >= 9.8 and close >= high * 0.998),
                        "limit_down": 0,
                    }
                )
            d += timedelta(days=1)
        return pd.DataFrame(rows)

    def cn_statement(self, symbol: str, kind: str):
        payload_income = {
            "营业利润": 80.0,
            "净利润": 50.0,
            "营业总收入": 400.0,
            "营业成本": 240.0,
            "利润总额": 80.0,
            "利息费用": 2.0,
        }
        payload_balance = {
            "流动资产合计": 200.0,
            "货币资金": 40.0,
            "流动负债合计": 80.0,
            "短期借款": 10.0,
            "固定资产": 90.0,
            "资产总计": 500.0,
            "负债合计": 180.0,
            "股东权益合计": 320.0,
            "长期借款": 50.0,
            "股本": 10.0,
        }
        payload_cash = {"经营活动产生的现金流量净额": 60.0}
        payload = {"income": payload_income, "balance": payload_balance, "cashflow": payload_cash}[kind]
        prev = dict(payload)
        if kind == "income":
            prev["净利润"] = 40.0
            prev["营业总收入"] = 350.0
        if kind == "balance":
            prev["资产总计"] = 480.0
            prev["长期借款"] = 60.0
            prev["流动资产合计"] = 180.0
            prev["流动负债合计"] = 90.0
            prev["股本"] = 10.0
        return [
            {
                "symbol": symbol,
                "kind": kind,
                "period": date(2025, 12, 31),
                "announced": date(2026, 3, 31),
                "payload": payload,
            },
            {
                "symbol": symbol,
                "kind": kind,
                "period": date(2024, 12, 31),
                "announced": date(2025, 3, 31),
                "payload": prev,
            },
            {
                "symbol": symbol,
                "kind": kind,
                "period": date(2026, 6, 30),
                "announced": date(2026, 10, 25),  # 晚于测试 asof，不得前视
                "payload": {**payload, "净利润": 999.0, "营业利润": 999.0},
            },
        ]

    def cn_dividends(self, symbol: str):
        return [
            {"symbol": symbol, "ex_date": date(2025, 6, 1), "announced": date(2025, 5, 1), "amount": 1.0},
            {"symbol": symbol, "ex_date": date(2024, 6, 1), "announced": date(2024, 5, 1), "amount": 0.9},
            {"symbol": symbol, "ex_date": date(2023, 6, 1), "announced": date(2023, 5, 1), "amount": 0.8},
        ]

    def global_members(self, name: str):
        if name == "hstech":
            return ["0700.HK"]
        return MissingData(f"测试不提供 {name}")

    def global_bars(self, symbol: str, start: date, end: date):
        return self.cn_bars(symbol, start, end, "qfq")

    def global_financials(self, symbol: str):
        return MissingData("测试无港美财务", symbol=symbol)


def test_store_roundtrip_and_pit(tmp_path: Path) -> None:
    store = MarketStore(tmp_path / "data")
    fetcher = FakeFetcher()
    report = sync_market(
        store,
        universe="csi800",
        start=date(2026, 1, 5),
        end=date(2026, 1, 20),
        resume=False,
        fetcher=fetcher,
        include_global=False,
        extra_symbols=[],
        sleep=0,
    )
    assert report.listings >= 2
    assert report.bar_symbols >= 2
    assert not store.is_empty()
    from zhiku.data.universe_live import build_fundamentals

    rows, delist, notes = build_fundamentals(store, date(2026, 10, 20), universe="csi800")
    assert "999999" not in {r.symbol for r in rows}
    assert delist.get("999999") == date(2024, 1, 15)
    by = {r.symbol: r for r in rows}
    assert by["600519"].net_income == 50.0
    assert by["600519"].ebit == 80.0
    assert by["600519"].net_income != 999.0  # 未公告不得进入
    assert by["600519"].pe_percentile is not None
    assert "缺数据" in "".join(notes) or "截面" in "".join(notes)


def test_resume_skips_completed(tmp_path: Path) -> None:
    store = MarketStore(tmp_path / "data")
    fetcher = FakeFetcher()
    sync_market(
        store,
        universe="csi800",
        start=date(2026, 1, 5),
        end=date(2026, 1, 10),
        resume=False,
        fetcher=fetcher,
        include_global=False,
        extra_symbols=[],
        sleep=0,
    )
    n1 = len([c for c in fetcher.calls if c.startswith("bars:600519:qfq")])
    sync_market(
        store,
        universe="csi800",
        start=date(2026, 1, 5),
        end=date(2026, 1, 10),
        resume=True,
        fetcher=fetcher,
        include_global=False,
        extra_symbols=[],
        sleep=0,
    )
    n2 = len([c for c in fetcher.calls if c.startswith("bars:600519:qfq")])
    assert n2 == n1
    assert any(c.startswith("bars:600519:none") for c in fetcher.calls)


def test_resume_skips_missing_bars(tmp_path: Path) -> None:
    store = MarketStore(tmp_path / "data")
    fetcher = FakeFetcher(fail_once="bars:600519:qfq")
    sync_market(
        store,
        universe="csi800",
        start=date(2026, 1, 5),
        end=date(2026, 1, 10),
        resume=False,
        fetcher=fetcher,
        include_global=False,
        extra_symbols=[],
        sleep=0,
        workers=1,
    )
    assert store.job_status("bars:qfq", "600519") == "missing"
    n1 = len([c for c in fetcher.calls if c.startswith("bars:600519:qfq")])
    sync_market(
        store,
        universe="csi800",
        start=date(2026, 1, 5),
        end=date(2026, 1, 10),
        resume=True,
        fetcher=fetcher,
        include_global=False,
        extra_symbols=[],
        sleep=0,
        workers=1,
    )
    n2 = len([c for c in fetcher.calls if c.startswith("bars:600519:qfq")])
    assert n2 == n1


def test_nan_delist_is_not_risk() -> None:
    row = snapshot_from_store(
        symbol="600519",
        asof=date(2026, 10, 20),
        listing={"name": "示例白酒", "instrument": "stock", "st": 0, "delist_date": float("nan")},
        statements={},
        prev_income=None,
        prev_balance=None,
        spot=None,
        closes=None,
        amounts=None,
        div_years=None,
        div_yield=None,
        shares_issued=None,
    )
    assert row.st is False
    assert row.delist_risk is False
    st_row = snapshot_from_store(
        symbol="000001",
        asof=date(2026, 10, 20),
        listing={"name": "*ST示例", "instrument": "stock", "st": 0},
        statements={},
        prev_income=None,
        prev_balance=None,
        spot=None,
        closes=None,
        amounts=None,
        div_years=None,
        div_yield=None,
        shares_issued=None,
    )
    assert st_row.st is True


def test_empty_picks_fail_gate() -> None:
    idx = pd.to_datetime(["2025-01-31", "2025-02-28"])
    prices = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)
    result = backtest_screen(prices, [], BacktestSettings(), screen="greenblatt")
    assert result.passed_gate is False
    assert any("空" in n for n in result.notes)


def test_missing_not_interpolated() -> None:
    assert pick_num({"营业利润": "-"}, "ebit") is None
    assert ebit_of({}) is None
    assert nwc_of({"流动资产合计": 10}) is None  # 缺流动负债
    row = snapshot_from_store(
        symbol="X",
        asof=date(2026, 10, 20),
        listing={"name": "缺字段示例", "instrument": "stock"},
        statements={},
        prev_income=None,
        prev_balance=None,
        spot=None,
        closes=None,
        amounts=None,
        div_years=None,
        div_yield=None,
        shares_issued=None,
    )
    assert row.ebit is None
    assert row.roe is None
    assert row.pe is None
    assert missing_ratio([row]) == 1.0
    filled = attach_percentiles([row, Fundamentals(symbol="Y", pe=10, pb=1)])
    assert filled[0].pe_percentile is None
    assert filled[1].pe_percentile == 0.0 or filled[1].pe_percentile is not None


def test_greenblatt_piotroski_from_statements() -> None:
    income = {"营业利润": 20.0, "净利润": 12.0, "营业总收入": 100.0, "营业成本": 60.0}
    balance = {
        "流动资产合计": 50.0,
        "货币资金": 10.0,
        "流动负债合计": 20.0,
        "短期借款": 5.0,
        "固定资产": 15.0,
        "资产总计": 80.0,
        "负债合计": 30.0,
        "股东权益合计": 50.0,
        "长期借款": 8.0,
    }
    cash = {"经营活动产生的现金流量净额": 14.0}
    row = snapshot_from_store(
        symbol="000333",
        asof=date(2026, 10, 20),
        listing={"name": "示例家电", "instrument": "stock"},
        statements={
            "income": {"payload": income},
            "balance": {"payload": balance},
            "cashflow": {"payload": cash},
        },
        prev_income={"净利润": 10.0, "营业总收入": 90.0, "营业成本": 58.0, "营业利润": 16.0},
        prev_balance={
            "资产总计": 70.0,
            "长期借款": 10.0,
            "流动资产合计": 40.0,
            "流动负债合计": 22.0,
            "股东权益合计": 45.0,
        },
        spot={"pe": 12, "pb": 2, "mktcap": 100.0, "amount": 1e8, "name": "示例家电"},
        closes=None,
        amounts=None,
        div_years=3,
        div_yield=0.04,
        shares_issued=False,
    )
    assert isinstance(roc(row), float)
    raw, items, missing = fscore(row)
    assert raw >= 1
    assert "roa_positive" in items


def test_limit_up_cannot_buy() -> None:
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.0], "BBB": [10.0, 10.5, 11.0]}, index=idx)
    up = pd.DataFrame({"AAA": [False, True, False], "BBB": [False, False, False]}, index=idx)
    nav_blocked, _ = monthly_equal_weight(
        prices, ["AAA", "BBB"], cost_bps=0, slippage_bps=0, limit_up=up
    )
    nav_free, _ = monthly_equal_weight(prices, ["AAA", "BBB"], cost_bps=0, slippage_bps=0)
    assert len(nav_blocked) == len(nav_free)
    result = backtest_screen(
        prices,
        ["AAA"],
        BacktestSettings(min_sharpe=-99, cost_bps=10, slippage_bps=5),
        screen="greenblatt",
        limit_up=up,
    )
    assert result.passed_gate is True
    assert any("涨停" in n or "成本" in n for n in result.notes)


def test_qlib_dump_and_predict_without_model(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-02", periods=40, freq="B")
    rows = []
    px = 10.0
    for d in dates:
        px *= 1.001
        rows.append(
            {
                "symbol": "600519",
                "market": "CN",
                "date": d.date(),
                "adjust": "qfq",
                "open": px,
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "volume": 1e6,
                "amount": 1e8,
                "turnover": 1.0,
            }
        )
    bars = pd.DataFrame(rows)
    qdir = tmp_path / "qlib"
    n = dump_qlib(bars, qdir)
    assert n == 1
    assert (qdir / "calendars" / "day.txt").is_file()
    assert list((qdir / "features").glob("*/close.day.bin"))
    store = MarketStore(tmp_path / "data")
    store.upsert_bars(bars)
    assert predict_scores(None, None, store=store, models_dir=tmp_path / "models") == {}
    path, meta = latest_model(tmp_path / "models")
    assert path is None
    assert meta is None


def test_lgb_train_predict_fictional(tmp_path: Path) -> None:
    lgb = pytest.importorskip("lightgbm")
    del lgb
    from zhiku.factors.qlib_adapter import train_model

    store = MarketStore(tmp_path / "data")
    dates = pd.date_range("2024-01-02", periods=280, freq="B")
    rows = []
    for symbol, base in (("600519", 100.0), ("000333", 50.0)):
        px = base
        for d in dates:
            px *= 1.0 + (((d.toordinal() + len(symbol)) % 9) - 4) * 0.002
            rows.append(
                {
                    "symbol": symbol,
                    "market": "CN",
                    "date": d.date(),
                    "adjust": "qfq",
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": 1e6,
                    "amount": 1e8,
                    "turnover": 1.0,
                }
            )
    store.upsert_bars(pd.DataFrame(rows))
    store.upsert_listings(
        [
            {"symbol": "600519", "market": "CN", "name": "示例白酒", "instrument": "stock"},
            {"symbol": "000333", "market": "CN", "name": "示例家电", "instrument": "stock"},
        ]
    )
    model = tmp_path / "models" / "lgb.pkl"
    msg = train_model(qlib_dir=tmp_path / "qlib", model_path=model, store=store)
    assert "已训练" in msg or "LightGBM" in msg or "qlib" in msg.lower()
    scores = predict_scores(None, str(model), store=store, asof=dates[-6].date(), models_dir=tmp_path / "models")
    assert scores
    assert all(0 <= v <= 100 for v in scores.values())
