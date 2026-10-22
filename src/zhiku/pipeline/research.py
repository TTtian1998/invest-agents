"""Explicit historical formula evaluation. Never called by daily execution."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from zhiku.backtest.engine import attach_benchmarks, load_price_panel, load_price_panel_from_store, sanitize_mixed_adjust
from zhiku.backtest.metrics import BacktestResult
from zhiku.backtest.pit_gate import evaluate_screens, load_statement_cache
from zhiku.backtest.universe import build_pit_universe
from zhiku.config import Paths
from zhiku.data.store import MarketStore
from zhiku.events.pit import drop_survivors
from zhiku.pipeline.artifacts import fingerprint, save_gate_research, validated_gate_status
from zhiku.pipeline.settings import PipelineSettings, load_pipeline_settings
from zhiku.pipeline.sources import resolve_source_mode
from zhiku.screens.runner import SCREEN_FUNCS


def evaluate_gates(paths: Paths, on: date, *, provider_kind: str, settings: PipelineSettings | None = None) -> Path:
    """Evaluate cached PIT history and save evidence, without promoting it.

    The demo dataset has only sparse monthly closes, not a validated PIT/open
    history. It honestly produces insufficient evidence while retaining useful
    formula observations. Live evaluation never downloads history implicitly.
    """
    settings = settings or load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    mode = resolve_source_mode(provider_kind)
    notes: list[str] = []
    results: dict[str, BacktestResult] = {}
    provenance: dict = {"method": "pandas-pit", "data_asof": on.isoformat()}
    if mode == "live":
        store = MarketStore(paths.data_dir)
        pit = build_pit_universe(store, asof=on, years=settings.data.years, universe=settings.data.universe, allow_fetch=False)
        notes.extend(pit.notes)
        provenance["universe"] = pit.to_dict()
        prices = load_price_panel_from_store(store, pit.all_symbols)
        prices = prices.loc[:on.isoformat()]
        if not prices.empty:
            prices = drop_survivors(prices, store.delist_map(), on)
            opens = store.price_panel(list(prices.columns), field="open").loc[:on.isoformat()]
            raw = store.price_panel(list(prices.columns), adjust="none", field="close").loc[:on.isoformat()]
            amounts = store.price_panel(list(prices.columns), field="amount").loc[:on.isoformat()]
            limit_up, limit_down = store.limit_panels(list(prices.columns))
            if not opens.empty:
                opens = sanitize_mixed_adjust(opens, raw).reindex(prices.index)
            cache = load_statement_cache(store, pit.all_symbols)
            # Hash inputs, not just output flags, so reviewers can identify evidence.
            provenance["input_hash"] = fingerprint({
                "close": prices.to_json(), "open": opens.to_json(), "raw": raw.to_json(),
                "amount": amounts.to_json(), "membership": {str(k): v for k, v in pit.membership.items()},
                "statements": str(cache.by_sym), "listings": str(cache.listings),
            })
            provenance["data_asof"] = prices.index.max().date().isoformat()
            if opens.empty or opens.reindex(prices.index).dropna(how="all").empty:
                notes.append("缺 T+1 开盘数据，拒绝用收盘价替代验收。")
            else:
                results = evaluate_screens(settings.model_copy(deep=True), close=prices, membership=pit.membership, opens=opens, raw_close=raw, amounts=amounts, limit_up=limit_up, limit_down=limit_down, cache=cache)
                for name, result in results.items():
                    if name in SCREEN_FUNCS:
                        attach_benchmarks(result, results, sanity_max_ann_gap=settings.backtest.sanity_max_ann_gap)
    elif mode == "demo":
        price_path = paths.fixtures / "monthly_prices.csv"
        prices = load_price_panel(price_path).loc[:on.isoformat()] if price_path.is_file() else None
        provenance["input_hash"] = fingerprint({"demo_prices": prices.to_json() if prices is not None else None})
        provenance["method"] = "demo-insufficient-pit"
        notes.append("离线样本只有月度收盘价，不能伪装成时点成分/T+1 开盘验收；仅展示研究观察。")
    else:
        notes.append("数据源 disabled，无法进行历史验收。")
    provenance.setdefault("input_hash", fingerprint({"empty": True, "mode": mode, "asof": on.isoformat()}))
    for name in SCREEN_FUNCS:
        if name not in results:
            results[name] = BacktestResult(name, None, None, None, None, None, engine="pandas-pit", insufficient_data=True, passed_gate=False, gate_status="数据不足", notes=["无完整时点验证数据"])
        status = validated_gate_status(results[name].to_dict(), settings)
        results[name].passed_gate = status == "passed"
        results[name].insufficient_data = status == "insufficient"
        results[name].gate_status = {"passed": "通过", "failed": "未过闸", "insufficient": "数据不足"}[status]
    return save_gate_research(paths, on, mode, settings, backtests={name: value.to_dict() for name, value in results.items()}, provenance=provenance, notes=notes)
