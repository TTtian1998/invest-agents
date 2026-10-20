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


def live_calendar(on: date) -> tuple[list[MacroEvent], list[str]]:
    """akshare 宏观/财经日历。任一接口失败不阻塞。"""
    notes: list[str] = []
    events: list[MacroEvent] = []
    try:
        import akshare as ak
    except ImportError:
        return [], ["缺数据：未安装 akshare，跳过 live 宏观日历"]
    fetchers = [
        ("macro_usa_non_farm", "US_NFP"),
        ("macro_china_cpi_monthly", "CN_CPI"),
        ("macro_china_ppi_yearly", "CN_PPI"),
        ("macro_china_gdp_yearly", "CN_GDP"),
        ("macro_china_lpr", "CN_LPR"),
        ("macro_bank_usa_interest_rate", "FOMC"),
    ]
    for fn_name, etype in fetchers:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            notes.append(f"缺数据：akshare 无 {fn_name}")
            continue
        try:
            df = fn()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"akshare {fn_name} 失败：{exc}")
            continue
        if df is None or getattr(df, "empty", True):
            notes.append(f"缺数据：{fn_name} 空表")
            continue
        events.append(
            MacroEvent(
                id=f"live-{etype}-{on.isoformat()}",
                event_type=etype,
                importance=3,
                notes=[f"live 源 {fn_name} 仅作存在性探测，数值未解析进预期差"],
            )
        )
    return events, notes
