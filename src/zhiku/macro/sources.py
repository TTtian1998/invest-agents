"""宏观数据源：离线 fixture 优先；live 走 akshare / FRED / yfinance，失败记缺数据。"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import yaml

from zhiku.calendar import SHANGHAI
from zhiku.macro.models import (
    GeoEvent,
    GeoHeat,
    HistorySample,
    HorizonMove,
    MacroEvent,
)
from zhiku.models import MissingData


def _parse_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=SHANGHAI)
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SHANGHAI)
    return dt


def _parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


def _reactions(raw: dict | None) -> dict[str, HorizonMove]:
    out: dict[str, HorizonMove] = {}
    for asset, mv in (raw or {}).items():
        if isinstance(mv, HorizonMove):
            out[str(asset)] = mv
            continue
        payload = dict(mv or {})
        out[str(asset)] = HorizonMove.model_validate(payload)
    return out


def load_calendar(path: Path) -> list[MacroEvent]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    events: list[MacroEvent] = []
    for row in raw.get("events") or []:
        row = dict(row)
        row["scheduled_at"] = _parse_dt(row.get("scheduled_at"))
        row["reactions"] = _reactions(row.get("reactions"))
        events.append(MacroEvent.model_validate(row))
    return events


def load_history(path: Path) -> list[HistorySample]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    samples: list[HistorySample] = []
    for row in raw.get("samples") or []:
        row = dict(row)
        row["released"] = _parse_date(row.get("released"))
        row["reactions"] = _reactions(row.get("reactions"))
        samples.append(HistorySample.model_validate(row))
    return samples


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def load_geo(path: Path) -> GeoHeat:
    raw = load_yaml(path)
    if not raw:
        return GeoHeat(missing="缺数据：无地缘风险样本", temperature="缺数据")
    events: list[GeoEvent] = []
    for row in raw.get("recent_events") or []:
        payload = dict(row)
        if True in payload and "on" not in payload:
            payload["on"] = payload.pop(True)
        if "on" not in payload and payload.get("date") is not None:
            payload["on"] = payload.get("date")
        payload["on"] = _parse_date(payload.get("on") or payload.get("date"))
        events.append(GeoEvent.model_validate(payload))
    gpr = raw.get("gpr") or {}
    value = gpr.get("value")
    pct = gpr.get("history_percentile")
    temp: str = "缺数据"
    if pct is None and value is None:
        temp = "缺数据"
    elif pct is not None:
        if pct >= 0.8:
            temp = "高"
        elif pct >= 0.5:
            temp = "中"
        else:
            temp = "低"
    return GeoHeat(
        asof=str(gpr.get("asof") or raw.get("asof") or ""),
        gpr=value,
        gpr_percentile=pct,
        temperature=temp,  # type: ignore[arg-type]
        active_chains=list(raw.get("active_chains") or []),
        recent_events=events,
        missing=None if value is not None or events else "缺数据：GPR 与事件均为空",
    )


def fred_series(series_id: str, start: date | None = None) -> object | MissingData:
    key = os.environ.get("FRED_API_KEY") or os.environ.get("FREDAPI_KEY")
    if not key:
        return MissingData("未配置 FRED_API_KEY")
    try:
        from fredapi import Fred

        fred = Fred(api_key=key)
        return fred.get_series(series_id, observation_start=start)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"FRED 拉取失败: {exc}")
    try:
        import pandas_datareader.data as pdr

        return pdr.DataReader(series_id, "fred", start=start, api_key=key)
    except ImportError:
        return MissingData("未安装 fredapi/pandas-datareader")
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"pandas-datareader FRED 失败: {exc}")


def _pick_row(row, *names: str):
    for name in names:
        if hasattr(row, "get") and name in row:
            return row[name]
        if hasattr(row, "index") and name in row.index:
            return row[name]
    return None


def _num_cell(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "--", "None", "nan", "—"}:
        return None
    try:
        return float(text.replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def _importance(raw) -> int:
    if raw is None:
        return 3
    text = str(raw).strip().lower()
    if text in {"高", "high", "3", "★★★", "***", "红"}:
        return 5
    if text in {"中", "medium", "2", "★★", "**", "橙"}:
        return 4
    if text in {"低", "low", "1", "★", "*", "黄"}:
        return 2
    n = _num_cell(raw)
    if n is not None:
        if n >= 3:
            return 5
        if n >= 2:
            return 4
        return 2
    return 3


CONFERENCE_KW = (
    "大会",
    "展会",
    "论坛",
    "博览会",
    "博览",
    "药交会",
    "车展",
    "开幕式",
    "研讨会",
    "智能网联",
    "产业峰会",
    "展览",
)
CB_KW = ("央行", "联储", "FOMC", "ECB", "日银", "欧央行", "议息", "利率决议", "PBOC")


def _is_conference_noise(name: str) -> bool:
    if any(k in name for k in CB_KW):
        return False
    return any(k in name for k in CONFERENCE_KW)


def official_core_events(on: date, horizon_days: int = 7) -> list[MacroEvent]:
    """公开日程补充：美联储例会等。不是预测，只填日历缺口。"""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    meetings = (
        (date(2026, 1, 28), "FOMC 利率决议"),
        (date(2026, 3, 18), "FOMC 利率决议"),
        (date(2026, 4, 29), "FOMC 利率决议"),
        (date(2026, 6, 17), "FOMC 利率决议"),
        (date(2026, 7, 29), "FOMC 利率决议"),
        (date(2026, 9, 16), "FOMC 利率决议"),
        (date(2026, 10, 27), "FOMC 会议（首日）"),
        (date(2026, 10, 28), "FOMC 利率决议"),
        (date(2026, 12, 9), "FOMC 利率决议"),
    )
    end = on + timedelta(days=horizon_days)
    out: list[MacroEvent] = []
    for day, name in meetings:
        if not (on <= day <= end):
            continue
        out.append(
            MacroEvent(
                id=f"official-fomc-{day.isoformat()}",
                event_type="FOMC",
                name=name,
                region="美国",
                importance=5,
                scheduled_at=datetime(day.year, day.month, day.day, 14, 0, tzinfo=et),
                notes=["公开日程补充（美联储例会）"],
            )
        )
    return out


def _infer_event_type(name: str, region: str) -> str:
    text = f"{region} {name}"
    if "核心CPI" in text or "核心消费" in text:
        return "US_CORE_CPI" if ("美" in text or "美国" in text) else "CN_CPI"
    rules = (
        ("非农", "US_NFP"),
        ("失业率", "US_UNEMP"),
        ("PCE", "US_PCE"),
        ("CPI", "US_CPI" if ("美" in text or "美国" in text) else "CN_CPI"),
        ("PPI", "US_PPI" if ("美" in text or "美国" in text) else "CN_PPI"),
        ("GDP", "US_GDP" if ("美" in text or "美国" in text) else "CN_GDP"),
        ("零售", "US_RETAIL"),
        ("FOMC", "FOMC"),
        ("联储", "FOMC"),
        ("利率决议", "FOMC" if ("美" in text or "联储" in text) else "ECB"),
        ("LPR", "CN_LPR"),
        ("社融", "CN_SF"),
        ("M2", "CN_M2"),
        ("PMI", "CN_PMI" if ("中" in text or "中国" in text) else "US_ISM_MFG"),
        ("欧央行", "ECB"),
        ("日央行", "BOJ"),
        ("OPEC", "OPEC"),
        ("EIA", "EIA"),
    )
    for needle, etype in rules:
        if needle.lower() in text.lower() or needle in text:
            return etype
    return "OTHER"


def parse_ws_calendar_row(row, *, asof: date, index: int) -> MacroEvent | None:
    """华尔街见闻 macro_info_ws 一行 → MacroEvent（含前值/预期/今值）。"""
    name = str(_pick_row(row, "事件", "event", "指标", "name") or "").strip()
    if not name:
        return None
    raw_time = _pick_row(row, "时间", "日期", "datetime", "date")
    scheduled = _parse_dt(raw_time)
    when = scheduled.date() if scheduled is not None else _parse_date(raw_time) or asof
    region = str(_pick_row(row, "地区", "国家", "region") or "").strip()
    etype = _infer_event_type(name, region)
    importance = _importance(_pick_row(row, "重要性", "重要", "importance"))
    if _is_conference_noise(name):
        if etype == "OTHER":
            etype = "CONFERENCE"
        importance = min(importance, 2)
    ev = MacroEvent(
        id=f"ws-{when.isoformat()}-{index}-{etype}",
        event_type=etype,
        name=name[:80],
        region=region,
        importance=importance,
        scheduled_at=scheduled,
        previous=_num_cell(_pick_row(row, "前值", "前", "previous")),
        consensus=_num_cell(_pick_row(row, "预期", "共识", "consensus")),
        actual=_num_cell(_pick_row(row, "今值", "公布", "actual", "今")),
    )
    if ev.consensus is None and ev.actual is not None:
        ev.notes.append("缺数据：无市场一致预期，不计算预期差")
    return ev


def live_calendar(on: date, horizon_days: int = 7) -> tuple[list[MacroEvent], list[str]]:
    """akshare 华尔街见闻日历（含前值/预期/今值）。金十/investing 能用则补。"""
    notes: list[str] = []
    events: list[MacroEvent] = []
    if not (os.environ.get("FRED_API_KEY") or os.environ.get("FREDAPI_KEY")):
        notes.append("未配置 FRED_API_KEY，跳过 FRED 宏观序列")
    try:
        import akshare as ak
    except ImportError:
        extra = official_core_events(on, horizon_days)
        return extra, ["缺数据：未安装 akshare，跳过 live 宏观日历；已补公开核心日程"]
    from datetime import timedelta

    days = [on + timedelta(days=i) for i in range(max(1, horizon_days) + 1)]
    for day in days:
        fn = getattr(ak, "macro_info_ws", None)
        if fn is None:
            notes.append("缺数据：akshare 无 macro_info_ws")
            break
        try:
            df = fn(date=day.strftime("%Y%m%d"))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"macro_info_ws {day} 失败：{exc}")
            continue
        if df is None or getattr(df, "empty", True):
            notes.append(f"缺数据：macro_info_ws {day} 空表")
            continue
        for i, (_, row) in enumerate(df.iterrows()):
            ev = parse_ws_calendar_row(row, asof=day, index=i)
            if ev is not None:
                events.append(ev)
    extra_fns = ("macro_info_js", "news_economic_baidu")
    for fn_name in extra_fns:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn() if fn_name != "news_economic_baidu" else fn()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{fn_name} 失败：{exc}")
            continue
        if df is None or getattr(df, "empty", True):
            continue
        notes.append(f"{fn_name} 返回 {len(df)} 行（已解析华尔街见闻为主）")
    have_fomc = {e.scheduled_at.date() for e in events if e.event_type == "FOMC" and e.scheduled_at}
    for ev in official_core_events(on, horizon_days):
        meet_day = ev.scheduled_at.date() if ev.scheduled_at is not None else None
        if meet_day is not None and meet_day in have_fomc:
            continue
        events.append(ev)
        notes.append(f"公开日程补充：{ev.name} {meet_day}")
    return events, notes


def live_anchors() -> tuple[list, list[str]]:
    """yfinance 锚定资产现价。失败记缺数据，不编造。"""
    from zhiku.macro.anchors import snapshots_from_live
    from zhiku.macro.models import ANCHOR_YF

    notes: list[str] = []
    try:
        import yfinance as yf
    except ImportError:
        return snapshots_from_live({}), ["缺数据：未安装 yfinance，锚定资产为空"]
    raw: dict[str, dict] = {}
    for asset, ticker in ANCHOR_YF.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist is None or getattr(hist, "empty", True):
                notes.append(f"缺数据：锚定 {asset} 空行情")
                continue
            close = hist["Close"].dropna()
            if close.empty:
                notes.append(f"缺数据：锚定 {asset} 无收盘")
                continue
            price = float(close.iloc[-1])
            chg = None
            if len(close) >= 2 and float(close.iloc[-2]) != 0:
                chg = float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0
            raw[asset] = {"price": price, "chg": chg}
        except Exception as exc:  # noqa: BLE001
            notes.append(f"锚定 {asset} 失败：{exc}")
    return snapshots_from_live(raw), notes
