"""Point-in-time signals, with explicit missing-data exclusions and provenance.

Rows represent a supplied trading-session calendar. No calendar is inferred, prices
are never filled, and signals at a close must be executed on a later session.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from typing import Literal

import pandas as pd

from .models import StrategyDecision, StrategySpec


def validate_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """Validate structural integrity; individual invalid prices are excluded later."""
    if not isinstance(closes, pd.DataFrame):
        raise ValueError("closes must be a pandas DataFrame")
    if closes.columns.has_duplicates:
        raise ValueError("duplicate symbol columns")
    if not all(isinstance(c, str) for c in closes.columns):
        raise ValueError("symbol columns must be strings")
    if isinstance(closes.index, pd.RangeIndex):
        raise ValueError("closes index must contain session dates, not row numbers")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(closes.index, errors="raise"))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid session dates") from exc
    if index.hasnans or index.tz is not None:
        raise ValueError("session dates must be nonmissing and timezone-naive")
    if not index.equals(index.normalize()):
        raise ValueError("session dates must be normalized daily dates")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("session dates must be unique and increasing")
    result = closes.copy()
    result.index = index
    return result


def generate_decision(
    closes: pd.DataFrame,
    asof: date | str,
    spec: StrategySpec,
    method: Literal["momentum", "equal_weight"] = "momentum",
) -> StrategyDecision:
    """Use exactly lookback+1 rows ending at asof; future prices cannot affect output.

    Equal-weight uses all complete-window eligible symbols, without the momentum
    filter. Momentum takes top_n nonnegative (by default) signals. Weight caps
    leave unallocated funds in cash; no optimization or leverage is attempted.
    """
    if method not in ("momentum", "equal_weight"):
        raise ValueError("method must be momentum or equal_weight")
    stamp = pd.Timestamp(asof)
    if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize():
        raise ValueError("asof must be a timezone-naive daily date")
    day = stamp.date()
    if spec.universe_known_at > day:
        raise ValueError("universe_known_at must be no later than asof")
    frame = validate_closes(closes)
    window = frame.loc[frame.index <= stamp].tail(spec.lookback + 1)
    exclusions: dict[str, str] = {}
    signals: dict[str, float] = {}
    used: dict[str, list[float | None | str]] = {}
    anchored = len(window) > 0 and window.index[-1] == stamp
    for symbol in spec.universe:
        if symbol not in frame.columns:
            exclusions[symbol] = "missing_column"
            used[symbol] = []
            continue
        values: list[float | None | str] = []
        for raw in window[symbol]:
            try:
                number = float(raw)
                values.append(number if math.isfinite(number) else None)
            except (TypeError, ValueError):
                values.append("invalid_non_numeric")
        used[symbol] = values
        if not anchored:
            exclusions[symbol] = "stale_or_missing_asof_session"
        elif len(window) != spec.lookback + 1:
            exclusions[symbol] = "insufficient_window"
        elif any(not isinstance(v, float) or v <= 0 for v in values):
            exclusions[symbol] = "missing_nonfinite_or_nonpositive_price"
        else:
            score = values[-1] / values[0] - 1  # type: ignore[operator]
            if not math.isfinite(score):
                exclusions[symbol] = "nonfinite_momentum"
            else:
                signals[symbol] = score
    eligible = sorted(signals)
    if method == "momentum":
        selected = sorted(
            (s for s in eligible if signals[s] >= spec.min_momentum),
            key=lambda s: (-signals[s], s),
        )[:spec.top_n]
    else:
        selected = eligible
    weight = min(spec.max_gross / len(selected), spec.max_weight) if selected else 0.0
    weights = {s: weight for s in sorted(selected)}
    evidence = {
        "source_mode": spec.source_mode,
        "universe_known_at": spec.universe_known_at.isoformat(),
        "window_dates": [d.date().isoformat() for d in window.index],
        "required_observations": spec.lookback + 1,
        "eligible_symbols": eligible,
        "exclusions": exclusions,
        "below_momentum_threshold": sorted(s for s in eligible if signals[s] < spec.min_momentum)
        if method == "momentum" else [],
        "execution_rule": "signal at asof close; earliest fill next supplied session",
        "calendar_assumption": "input rows are the declared trading-session calendar",
    }
    payload = {"spec": spec.model_dump(mode="json"), "asof": day.isoformat(),
               "method": method, "dates": evidence["window_dates"], "closes": used}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return StrategyDecision(
        asof=day, method=method, signals=signals, target_weights=weights,
        cash_weight=max(0.0, 1 - sum(weights.values())),
        rebalance_schedule={"every_trading_sessions": spec.rebalance_every,
                            "signal_asof": day.isoformat(), "execution": "next_session"},
        evidence=evidence, fingerprint=digest, spec_fingerprint=spec.fingerprint(),
    )


def schedule_decisions(
    closes: pd.DataFrame, spec: StrategySpec, start: date | str, end: date | str,
    method: Literal["momentum", "equal_weight"] = "momentum",
) -> list[StrategyDecision]:
    """Anchor at the first supplied session >= start, then count supplied sessions.

    Warmup rows before start are retained for signals, not used to alter the anchor.
    """
    frame = validate_closes(closes)
    start_stamp, end_stamp = pd.Timestamp(start), pd.Timestamp(end)
    if start_stamp > end_stamp:
        raise ValueError("start must be no later than end")
    dates = frame.index[(frame.index >= start_stamp) & (frame.index <= end_stamp)]
    return [generate_decision(frame, d.date(), spec, method)
            for d in dates[::spec.rebalance_every]]
