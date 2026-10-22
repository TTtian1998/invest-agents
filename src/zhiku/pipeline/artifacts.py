"""Small, content-addressed acceptance artifacts shared by research and daily.

Publication is an explicit local action, not a side effect of daily execution.
The stored metrics are revalidated; a claimed ``passed_gate`` flag is never authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from zhiku.pipeline.settings import PipelineSettings
from zhiku.pipeline.sources import resolve_source_mode
from zhiku.screens.runner import SCREEN_FUNCS

GATE_SCHEMA = "zhiku.gate_acceptance.v1"
STRATEGY_VERSION = "daily-research-boundaries.v1"


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = sorted([*root.glob("screens/*.py"), *root.glob("backtest/*.py"),
                      *root.glob("factors/*.py"), root / "pipeline" / "artifacts.py",
                      root / "pipeline" / "research.py", root / "pipeline" / "settings.py",
                      root / "data" / "compute.py", root / "data" / "universe_live.py"])
    return fingerprint({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def config_fingerprint(settings: PipelineSettings) -> str:
    """Only settings affecting selection/validation/model context invalidate approval."""
    keys = ["universe", "quality", "backtest", "data", "merge", *SCREEN_FUNCS]
    raw = settings.model_dump(mode="json")
    return fingerprint({"schema": GATE_SCHEMA, "strategy_version": STRATEGY_VERSION, "implementation": _implementation_fingerprint(), "settings": {key: raw[key] for key in keys}})


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validated_gate_status(result: dict, settings: PipelineSettings) -> str:
    """Strict acceptance: missing evidence is insufficient, not an implicit pass."""
    if not isinstance(result, dict):
        return "insufficient"
    cfg = settings.backtest
    counts = [result.get("n_months"), result.get("n_covered")]
    if any(type(value) is not int or value < 4 for value in counts):
        return "insufficient"
    benchmarks = result.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return "insufficient"
    equity = result.get("equity")
    if not isinstance(equity, list) or len(equity) < 4:
        return "insufficient"
    for pair in equity:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not _finite(pair[1]) or pair[1] <= 0:
            return "insufficient"
        try:
            date.fromisoformat(pair[0])
        except (ValueError, TypeError):
            return "insufficient"
    required = ["annualized", "sharpe", "coverage_ratio", "n_months", "n_covered"]
    if cfg.require_beat_benchmark:
        required += ["excess_annualized", "benchmark_annualized"]
    if cfg.require_half_excess:
        required += ["excess_first_half", "excess_second_half"]
    if any(not _finite(result.get(key)) for key in required):
        return "insufficient"
    if result.get("insufficient_data") or result["n_months"] < 4 or result["n_covered"] < 4:
        return "insufficient"
    coverage = result["n_covered"] / result["n_months"]
    if not 0 <= coverage <= 1 or abs(coverage - result["coverage_ratio"]) > 1e-9 or coverage < cfg.min_coverage:
        return "insufficient"
    if cfg.fill == "next_open" and (type(result.get("close_fallback_fills")) is not int or result["close_fallback_fills"] != 0):
        return "insufficient"
    if result.get("engine") != "pandas-pit" or result.get("fill") != cfg.fill:
        return "insufficient"
    if len(result.get("equity") or []) < 4:
        return "insufficient"
    if cfg.require_beat_benchmark or cfg.require_half_excess or cfg.require_sharpe_ge_benchmark:
        bench = benchmarks.get("universe_ew")
        if not isinstance(bench, dict) or bench.get("engine") != "pandas-pit" or bench.get("fill") != cfg.fill:
            return "insufficient"
        if cfg.fill == "next_open" and (type(bench.get("close_fallback_fills")) is not int or bench["close_fallback_fills"] != 0):
            return "insufficient"
    if result["sharpe"] < cfg.min_sharpe or result.get("sanity_ok") is False:
        return "failed"
    if cfg.require_beat_benchmark and result["excess_annualized"] <= 0:
        return "failed"
    if cfg.require_half_excess and (result["excess_first_half"] <= 0 or result["excess_second_half"] <= 0):
        return "failed"
    if cfg.require_sharpe_ge_benchmark:
        benchmark = benchmarks.get("universe_ew")
        if not isinstance(benchmark, dict) or not _finite(benchmark.get("sharpe")):
            return "insufficient"
        if result["sharpe"] < benchmark["sharpe"]:
            return "failed"
    return "passed"


def _clean_json(value: Any) -> Any:
    # Financial metrics can have NaN for an empty sample. Persist explicit absence.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


def save_gate_research(paths, on: date, provider_kind: str, settings: PipelineSettings, *, backtests: dict, provenance: dict, notes: list[str]) -> Path:
    payload = _clean_json({
        "schema": GATE_SCHEMA,
        "asof": on.isoformat(),
        "provider_kind": resolve_source_mode(provider_kind),
        "config_fingerprint": config_fingerprint(settings),
        "backtests": backtests,
        "provenance": provenance,
        "notes": notes,
    })
    version = fingerprint(payload)
    path = Path(paths.models_dir) / "gate-research" / f"{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps({**payload, "version": version}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _version(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_research(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Gate artifact must be an object")
    version = data.pop("version", None)
    if not _version(version) or data.get("schema") != GATE_SCHEMA or fingerprint(data) != version:
        raise ValueError("Gate artifact schema or content hash is invalid")
    if data.get("provider_kind") not in {"live", "demo"} or not _version(data.get("config_fingerprint")):
        raise ValueError("Invalid gate context")
    evidence_asof = date.fromisoformat(data["asof"])
    provenance = data.get("provenance")
    if not isinstance(provenance, dict) or not _version(provenance.get("input_hash")) or not isinstance(provenance.get("method"), str):
        raise ValueError("Missing gate validation provenance")
    cutoff = date.fromisoformat(provenance["data_asof"])
    if cutoff > evidence_asof:
        raise ValueError("Gate artifact contains future evidence")
    backtests = data.get("backtests")
    if not isinstance(backtests, dict) or not backtests or any(not isinstance(v, dict) for v in backtests.values()):
        raise ValueError("Gate backtests must contain evidence objects")
    for value in backtests.values():
        previous_day = date.min
        equity = value.get("equity", [])
        if not isinstance(equity, list):
            raise ValueError("Invalid equity evidence")
        for pair in equity:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not _finite(pair[1]):
                raise ValueError("Invalid equity evidence")
            day = date.fromisoformat(pair[0])
            if day > cutoff:
                raise ValueError("Future equity evidence")
            if day <= previous_day:
                raise ValueError("Equity evidence dates must be increasing")
            previous_day = day
    return {**data, "version": version}


def _gate_statuses(data: dict, settings: PipelineSettings, on: date) -> dict[str, str]:
    statuses = {}
    for name, value in data["backtests"].items():
        if name not in SCREEN_FUNCS:
            continue
        status = validated_gate_status(value, settings)
        equity = value.get("equity") or []
        if status != "insufficient" and (on - date.fromisoformat(equity[-1][0])).days > settings.artifacts.gate_max_age_days:
            status = "insufficient"
        statuses[name] = status
    return statuses


def publish_gate_research(paths, research_path: Path, *, on: date, provider_kind: str, settings: PipelineSettings) -> dict:
    data = _read_research(research_path)
    mode = resolve_source_mode(provider_kind)
    if mode == "disabled" or data["provider_kind"] != mode:
        raise ValueError("Gate artifact provider mismatch")
    if data["config_fingerprint"] != config_fingerprint(settings):
        raise ValueError("Gate artifact config mismatch; rerun research")
    evidence_asof = date.fromisoformat(data["asof"])
    provenance = data.get("provenance") or {}
    if not provenance.get("input_hash") or not provenance.get("method"):
        raise ValueError("Missing gate validation provenance")
    cutoff = date.fromisoformat(provenance.get("data_asof") or data["asof"])
    if cutoff > evidence_asof or evidence_asof > on:
        raise ValueError("Gate artifact contains future evidence")
    if (on - min(evidence_asof, cutoff)).days > settings.artifacts.gate_max_age_days:
        raise ValueError("Gate validation has expired; rerun research")
    # Insufficient/empty evaluations are never publishable, and cannot clobber a
    # previously useful approval. Valid negative evidence may publish an empty pool.
    statuses = _gate_statuses(data, settings, on)
    if not statuses or all(s == "insufficient" for s in statuses.values()):
        raise ValueError("No complete gate validation evidence; previous publication preserved")
    folder = Path(paths.models_dir) / "approved-gates"
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / f"{data['version']}.json"
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if artifact.exists() and artifact.read_text(encoding="utf-8") != text:
        raise ValueError("Immutable gate version already exists with different content")
    artifact.write_text(text, encoding="utf-8")
    manifest = {
        "version": data["version"], "published_asof": on.isoformat(),
        "published_at": datetime.now(timezone.utc).isoformat(), "statuses": statuses,
    }
    temp = folder / "approved.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(folder / "approved.json")
    return manifest


def load_approved_gates(paths, *, on: date, provider_kind: str, settings: PipelineSettings) -> tuple[dict, dict]:
    """No fallback to old filenames, fixture validation, or unapproved research."""
    info: dict[str, Any] = {"status": "unavailable", "reason": "not_published", "approved_screens": []}
    folder = Path(paths.models_dir) / "approved-gates"
    if not (folder / "approved.json").is_file():
        return {}, info
    try:
        manifest = json.loads((folder / "approved.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid gate manifest")
        version = manifest["version"]
        if not _version(version):
            raise ValueError("Invalid gate version")
        data = _read_research(folder / f"{version}.json")
        if data["version"] != version:
            raise ValueError("Gate manifest/version mismatch")
        info["version"] = version
        published = date.fromisoformat(manifest["published_asof"])
        evaluated = date.fromisoformat(data["asof"])
        cutoff = date.fromisoformat(data["provenance"]["data_asof"])
        from zhiku.calendar import SHANGHAI
        stamp = datetime.fromisoformat(manifest["published_at"])
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Publication time must include timezone")
        recorded = stamp.astimezone(SHANGHAI).date()
        effective_publication = max(published, recorded) if data["provider_kind"] == "live" else published
        if max(effective_publication, evaluated, cutoff) > on or cutoff > evaluated:
            info["reason"] = "future_artifact"
        elif data["provider_kind"] != resolve_source_mode(provider_kind):
            info["reason"] = "provider_mismatch"
        elif data["config_fingerprint"] != config_fingerprint(settings):
            info["reason"] = "config_mismatch"
        elif (on - min(evaluated, cutoff)).days > settings.artifacts.gate_max_age_days:
            info["reason"] = "expired"
        else:
            statuses = _gate_statuses(data, settings, on)
            info.update(status="approved", reason="validated", evaluated_asof=evaluated.isoformat(), statuses=statuses, approved_screens=[n for n, s in statuses.items() if s == "passed"])
            return data, info
    except (OSError, ValueError, TypeError, KeyError):
        info["reason"] = "invalid_artifact"
    return {}, info
