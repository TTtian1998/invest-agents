"""事件研究：各类催化 T+1/T+5/T+20 平均超额与胜率，并反哺权重。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict


class EventSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    catalyst: str
    source: str = "unknown"
    t1: float | None = None
    t5: float | None = None
    t20: float | None = None
    crowding_bucket: str = ""
    surprise_z: float | None = None


class CatalystStat(BaseModel):
    catalyst: str
    n: int
    avg_t1: float | None = None
    avg_t5: float | None = None
    avg_t20: float | None = None
    winrate_t1: float | None = None
    winrate_t5: float | None = None
    winrate_t20: float | None = None


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _winrate(xs: list[float]) -> float | None:
    return sum(1 for x in xs if x > 0) / len(xs) if xs else None


def summarize_samples(samples: list[EventSample]) -> dict[str, CatalystStat]:
    by: dict[str, list[EventSample]] = {}
    for s in samples:
        by.setdefault(s.catalyst, []).append(s)
    out: dict[str, CatalystStat] = {}
    for cat, rows in by.items():
        t1 = [r.t1 for r in rows if r.t1 is not None]
        t5 = [r.t5 for r in rows if r.t5 is not None]
        t20 = [r.t20 for r in rows if r.t20 is not None]
        out[cat] = CatalystStat(
            catalyst=cat,
            n=len(rows),
            avg_t1=_avg(t1),
            avg_t5=_avg(t5),
            avg_t20=_avg(t20),
            winrate_t1=_winrate(t1),
            winrate_t5=_winrate(t5),
            winrate_t20=_winrate(t20),
        )
    return out


def load_event_history(path: Path) -> list[EventSample]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = raw.get("samples") if isinstance(raw, dict) else raw
    return [EventSample.model_validate(r) for r in (rows or [])]


def event_study_from_prices(
    prices: pd.DataFrame,
    events: list[tuple[str, pd.Timestamp, str]],
    horizons: tuple[int, ...] = (1, 5, 20),
    benchmark: str | None = "510300",
) -> list[EventSample]:
    """events: (symbol, event_date, catalyst)。价格不足则该样本缺对应期限，不插值。"""
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    samples: list[EventSample] = []
    for symbol, when, catalyst in events:
        if symbol not in px.columns:
            samples.append(EventSample(catalyst=catalyst, t1=None, t5=None, t20=None))
            continue
        series = px[symbol].dropna()
        bench = px[benchmark].dropna() if benchmark and benchmark in px.columns else None
        after = series[series.index > when]
        payload: dict[str, float | None] = {"t1": None, "t5": None, "t20": None}
        for h, key in zip(horizons, ("t1", "t5", "t20"), strict=False):
            if len(after) < h:
                continue
            p0 = series[series.index <= when]
            if p0.empty or float(p0.iloc[-1]) == 0:
                continue
            r = float(after.iloc[h - 1] / p0.iloc[-1] - 1.0)
            if bench is not None:
                b0 = bench[bench.index <= when]
                b1 = bench[bench.index > when]
                if len(b0) and len(b1) >= h and float(b0.iloc[-1]) != 0:
                    r -= float(b1.iloc[h - 1] / b0.iloc[-1] - 1.0)
            payload[key] = r
        samples.append(
            EventSample(
                catalyst=catalyst,
                t1=payload["t1"],
                t5=payload["t5"],
                t20=payload["t20"],
            )
        )
    return samples


def default_weights() -> dict[str, dict[str, float]]:
    return {"catalyst": {}, "source": {}, "x_handle": {}}


def load_weights(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return default_weights()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_weights()
    return {
        "catalyst": {str(k): float(v) for k, v in (data.get("catalyst") or {}).items()},
        "source": {str(k): float(v) for k, v in (data.get("source") or {}).items()},
        "x_handle": {str(k).lower(): float(v) for k, v in (data.get("x_handle") or {}).items()},
    }


def save_weights(path: Path, weights: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_weights(
    weights: dict[str, dict[str, float]],
    stats: dict[str, CatalystStat],
    *,
    low: float = 0.45,
    high: float = 0.55,
) -> dict[str, dict[str, float]]:
    """胜率差的催化降权，好的略升，夹在 [0.3, 1.5]。"""
    cat = dict(weights.get("catalyst") or {})
    for name, st in stats.items():
        wr = st.winrate_t5 if st.winrate_t5 is not None else st.winrate_t1
        if wr is None:
            continue
        cur = cat.get(name, 1.0)
        if wr < low:
            cur *= 0.9
        elif wr > high:
            cur *= 1.05
        cat[name] = max(0.3, min(1.5, cur))
    out = dict(weights)
    out["catalyst"] = cat
    return out
