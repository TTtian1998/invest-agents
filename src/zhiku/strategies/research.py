"""Offline, immutable ETF experiments. Nothing here publishes into daily execution."""
from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
from datetime import date
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zhiku.strategies.backtest import backtest_targets
from zhiku.strategies.models import StrategySpec
from zhiku.strategies.signals import schedule_decisions


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_bars(path: Path, universe: Sequence[str], *, raw_bytes: bytes | None = None) -> dict[str, pd.DataFrame]:
    """A supplied session grid, with explicit instrument-level execution masks."""
    frame = pd.read_csv(io.BytesIO(path.read_bytes() if raw_bytes is None else raw_bytes), dtype={"symbol": str})
    required = {"date", "symbol", "open", "close", "tradable", "limit_up", "limit_down"}
    if not required.issubset(frame.columns):
        raise ValueError(f"bars missing columns: {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty or frame["date"].isna().any() or frame["symbol"].isna().any():
        raise ValueError("bars must contain dated instrument records")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate date/symbol bars")
    if set(frame["symbol"]) != set(universe):
        raise ValueError("bars symbols must exactly match the declared fixed universe")
    for name in ("tradable", "limit_up", "limit_down"):
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"{name} must explicitly contain 0 or 1")
    for name in ("open", "close"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        present = frame[name].dropna()
        if not (np.isfinite(present).all() and (present > 0).all()):
            raise ValueError(f"{name} prices must be finite and positive or missing")
    panels = {name: frame.pivot(index="date", columns="symbol", values=name).sort_index()
              .reindex(columns=universe) for name in ("open", "close", "tradable", "limit_up", "limit_down")}
    # Absent symbol sessions are NOT inferred tradable from another symbol's calendar.
    for name in ("tradable", "limit_up", "limit_down"):
        panels[name] = panels[name].fillna(0).astype(bool)
    return panels


def run_research(spec_path: Path, bars_path: Path, split: date, output_dir: Path) -> Path:
    """Fixed parameters, chronological development/holdout, no optimizer or promotion."""
    spec_bytes, bars_bytes = spec_path.read_bytes(), bars_path.read_bytes()
    raw_spec = yaml.safe_load(spec_bytes.decode("utf-8"))
    spec = StrategySpec.model_validate(raw_spec)
    panels = load_bars(bars_path, spec.universe, raw_bytes=bars_bytes)
    closes = panels["close"]
    dates = closes.index
    cut = pd.Timestamp(split)
    if cut not in dates or cut <= dates[0] or cut >= dates[-1]:
        raise ValueError("split must be an interior observed session (holdout starts on split)")
    if spec.universe_known_at > dates[0].date():
        raise ValueError("universe_known_at must precede all research sessions")
    periods = {"development": dates[dates < cut], "holdout": dates[dates >= cut]}
    results: dict = {}
    for label, sessions in periods.items():
        if len(sessions) < 2:
            raise ValueError(f"{label} requires at least two sessions")
        strategies: dict = {}
        for method in ("momentum", "equal_weight"):
            # History is warmup only. Both portfolios restart in cash at each split;
            # no development holdings, signals or pre-split returns leak into holdout.
            decisions = schedule_decisions(closes.loc[:sessions[-1]], spec,
                                           sessions[0].date(), sessions[-1].date(), method=method)
            targets = {item.asof: item.target_weights for item in decisions}
            kwargs = {"cost_bps": spec.cost_bps, "slippage_bps": spec.slippage_bps,
                      "initial_cash": spec.initial_cash, "lot_size": spec.lot_size,
                      "tradable": panels["tradable"].loc[sessions],
                      "limit_up": panels["limit_up"].loc[sessions],
                      "limit_down": panels["limit_down"].loc[sessions]}
            result = backtest_targets(closes.loc[sessions], panels["open"].loc[sessions], targets, **kwargs)
            strategies[method] = {"decisions": [d.model_dump(mode="json") for d in decisions],
                                  "backtest": result.as_dict()}
        results[label] = {"start": sessions[0].date().isoformat(), "end": sessions[-1].date().isoformat(),
                          "sessions": len(sessions), "strategies": strategies}
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    code_files.append(Path(__file__).parent.parent / "backtest" / "metrics.py")
    payload = {
        "schema": "zhiku.etf_research.v1", "status": "RESEARCH_ONLY", "source_mode": spec.source_mode,
        "spec": spec.model_dump(mode="json"), "split": split.isoformat(),
        "provenance": {"spec_sha256": _digest(spec_bytes), "bars_sha256": _digest(bars_bytes),
                       "code_sha256": _digest(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in code_files))},
        "dependencies": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pydantic")},
        "periods": results,
        "acceptance": {"production_eligible": False, "reason": "Research artifact only; no daily publication adapter or verified real-data provenance in this pilot.",
                       "holdout_sessions": len(periods["holdout"]), "minimum_recommended_sessions": 60,
                       "sample_sufficient": len(periods["holdout"]) >= 60},
        "limitations": ["Fixed, user-declared universe and calendar: not a survivorship-free universe proof.",
                        "Only same-currency cash ETFs; no FX, corporate actions or dividend engine.",
                        "Open and close must share a consistent price basis; raw prices spanning splits/dividends require preprocessing and independent validation.",
                        "Execution masks are supplied evidence, not verified exchange facts. No real orders, no market impact or depth model.",
                        "Holdout is a chronological diagnostic, not a sealed independent test once inspected. No tuning or automatic acceptance.",
                        "Equal-weight comparator uses the same eligible pool, cadence, exposure caps and cost engine; it is rebalanced, not buy-and-hold.",
                        "No report is accepted by daily, publish-gates or factor publish. Demo returns are synthetic, not evidence of profitability."]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ident = _digest(encoded.encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"etf-research-{ident}.json"
    try:
        with target.open("x", encoding="utf-8") as out:
            out.write(encoded)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing research artifact does not match its content hash")
    return target
