"""zk daily：选股 → 打分 → 回测闸门 → AI 审核 → 报告。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from zhiku.backtest.engine import (
    attach_benchmarks,
    backtest_screen,
    compute_benchmarks,
    crosscheck_public_index,
    finalize_gate,
    load_price_panel,
    load_price_panel_from_store,
)
from zhiku.config import Paths
from zhiku.events.pit import drop_survivors, load_delist_map
from zhiku.factors.merge import merge_scores, scale_merge_settings
from zhiku.factors.qlib_adapter import model_status, predict_scores, qlib_available
from zhiku.notify.render import render, write_report
from zhiku.pipeline.settings import load_pipeline_settings
from zhiku.pipeline.universe import fundamentals_path, load_pipeline_universe
from zhiku.review.audit import audit_top
from zhiku.screens.fundamentals import Fundamentals, by_symbol
from zhiku.screens.runner import ScreenBundle, average_formula_score, formula_scores, run_screens


@dataclass
class DailyResult:
    asof: str
    notes: list[str] = field(default_factory=list)
    quality_passed: list[str] = field(default_factory=list)
    quality_rejected: list[dict] = field(default_factory=list)
    screens: dict[str, list[dict]] = field(default_factory=dict)
    backtests: dict[str, dict] = field(default_factory=dict)
    gated_out: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    audits: list[dict] = field(default_factory=list)
    news: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    qlib: dict = field(default_factory=dict)
    fictional: bool = True
    universe_size: int = 0
    missing_ratio: float | None = None
    pit_universe: dict = field(default_factory=dict)
    merge_weights: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _fundamentals_path(paths: Paths) -> Path:
    return fundamentals_path(paths)


def run_daily(
    paths: Paths,
    on: date,
    *,
    provider_kind: str = "fixture",
    screen_names: list[str] | None = None,
) -> DailyResult:
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    rows, delist, uni_notes = load_pipeline_universe(
        paths, on, provider_kind=provider_kind, settings=settings
    )
    names = {r.symbol: r.name for r in rows}
    bundle = run_screens(rows, settings, names=screen_names)
    notes: list[str] = list(uni_notes)
    qlib_meta = model_status(paths.models_dir)
    if qlib_meta.get("status") == "未训练":
        notes.append("qlib 因子：未训练。总分仅使用公式分。")
    elif not qlib_available() and qlib_meta.get("kind") not in {"lgb_lite", "lgb_panel", "zscore_panel"}:
        notes.append("未安装 [qlib]，若本地已有 LightGBM / 面板模型仍会出分。")
    store = None
    if provider_kind == "live":
        from zhiku.data.store import MarketStore

        store = MarketStore(paths.data_dir)
    qlib_scores = predict_scores(
        None,
        None,
        store=store,
        asof=on,
        symbols=[r.symbol for r in rows],
        models_dir=paths.models_dir,
    )
    if qlib_scores:
        notes.append(
            f"qlib 模型 {qlib_meta.get('trained_on') or '未知日期'}；"
            f"IC={qlib_meta.get('ic')} RankIC={qlib_meta.get('rank_ic')}"
        )

    result = DailyResult(asof=on.isoformat(), fictional=provider_kind != "live")
    result.universe_size = len(rows)
    cmp_path = paths.models_dir / "comparison.json"
    if cmp_path.is_file():
        try:
            qlib_meta = dict(qlib_meta)
            qlib_meta["comparison"] = json.loads(cmp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    result.qlib = qlib_meta
    result.quality_passed = bundle.quality_passed
    result.quality_rejected = [
        {"symbol": q.symbol, "reasons": q.reasons} for q in bundle.quality if not q.passed
    ]
    from zhiku.data.compute import missing_ratio as _miss

    result.missing_ratio = _miss(rows) if rows else None
    if result.missing_ratio is not None:
        notes.append(f"截面缺数据比例 {result.missing_ratio:.1%}")

    prices = None
    limit_up = limit_down = None
    live_prices: dict[str, float] = {}
    pit = None
    if provider_kind == "live" and store is not None and not store.is_empty():
        bench_syms = [
            s
            for s in dict.fromkeys(
                [settings.backtest.benchmark, *list(settings.backtest.benchmark_symbols or [])]
            )
            if s and s not in {"universe_ew", "csi300", "csi800"}
        ]
        from zhiku.data.fetchers import AkshareFetcher
        from zhiku.backtest.universe import build_pit_universe

        pit = build_pit_universe(
            store,
            asof=on,
            years=settings.data.years,
            universe=settings.data.universe,
            fetcher=AkshareFetcher(),
            allow_fetch=True,
        )
        result.pit_universe = pit.to_dict()
        notes.extend(pit.notes)
        want = list(dict.fromkeys([r.symbol for r in rows] + pit.all_symbols + bench_syms))
        prices = load_price_panel_from_store(store, want)
        limit_up, limit_down = store.limit_panels(list(prices.columns) if not prices.empty else [])
        if delist and prices is not None and not prices.empty:
            prices = drop_survivors(prices, delist, on)
            notes.append("价格面板来自本地库（前复权），已按退市日截断；末日复权混入已清洗。")
        if prices is None or prices.empty:
            notes.append("本地库缺日线，跳过回测闸门。请先 zk data sync。")
            prices = None
        notes.extend(crosscheck_public_index(store, on))
        for r in rows:
            bar = store.last_bar(r.symbol, "none") or store.last_bar(r.symbol, "qfq")
            if bar and bar.get("close") is not None:
                live_prices[r.symbol] = float(bar["close"])
    else:
        if not delist:
            delist = load_delist_map(paths.fixtures / "delisted.yaml")
        price_file = paths.fixtures / "monthly_prices.csv"
        if price_file.is_file():
            prices = load_price_panel(price_file)
            if delist:
                prices = drop_survivors(prices, delist, on)
                notes.append("价格面板已按退市日截断，避免幸存者偏差。")

    active_screens = dict(bundle.screens)
    gated: list[str] = []
    backtests: dict[str, dict] = {}
    benches: dict = {}
    membership = pit.membership if pit is not None else None
    universe_note = pit.label if pit is not None else None
    if prices is not None:
        benches = compute_benchmarks(
            prices,
            settings.backtest,
            membership=membership,
            universe_note=universe_note,
        )
        for bname, bres in benches.items():
            backtests[bname] = bres.to_dict()
        if benches.get("universe_ew") and (benches.get("csi300") or benches.get("csi800")):
            probe = attach_benchmarks(
                benches["universe_ew"],
                benches,
                sanity_max_ann_gap=settings.backtest.sanity_max_ann_gap,
            )
            if probe.sanity_note:
                notes.append(probe.sanity_note)
            if probe.sanity_ok is False:
                notes.append("同池等权与指数年化差超过阈值，请对照时点池口径，勿与静态今日成分混读。")
        pit_results = None
        if provider_kind == "live" and store is not None and membership is not None:
            try:
                from zhiku.backtest.pit_gate import evaluate_screens, load_statement_cache

                opens = store.price_panel(list(prices.columns), field="open")
                raw_px = store.price_panel(list(prices.columns), adjust="none", field="close")
                amt_px = store.price_panel(list(prices.columns), field="amount")
                cache = load_statement_cache(store, list(pit.all_symbols) if pit is not None else None)
                pit_results = evaluate_screens(
                    settings,
                    close=prices,
                    membership=membership,
                    opens=opens,
                    raw_close=raw_px,
                    amounts=amt_px,
                    limit_up=limit_up,
                    limit_down=limit_down,
                    cache=cache,
                    names=list(bundle.screens),
                )
                notes.append(
                    "闸门：每月时点重算名单（T 日及之前数据），T+1 开盘成交，涨停买不进；"
                    "对同池等权超额>0、Sharpe≥等权、前后半段超额都>0。"
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"时点闸门失败，回退静态今日名单（该口径可能前视）：{exc}")
                pit_results = None
        gate_benches = dict(benches)
        if pit_results and pit_results.get("universe_ew") is not None:
            gate_benches["universe_ew"] = pit_results["universe_ew"]
        for name, srows in bundle.screens.items():
            if pit_results and name in pit_results:
                bt = pit_results[name]
            else:
                picks = [r.symbol for r in srows[: settings.backtest.top_n]]
                bt = backtest_screen(
                    prices,
                    picks,
                    settings.backtest,
                    screen=name,
                    delist=delist,
                    asof=on,
                    limit_up=limit_up,
                    limit_down=limit_down,
                    membership=membership,
                    universe_note=universe_note,
                )
            attach_benchmarks(bt, gate_benches, sanity_max_ann_gap=settings.backtest.sanity_max_ann_gap)
            finalize_gate(bt, settings.backtest)
            backtests[name] = bt.to_dict()
            if not bt.passed_gate:
                gated.append(name)
                notes.append(
                    f"公式 {name} 回测未达标（对同池等权超额须>0、Sharpe 不低于等权、"
                    "前后半段超额都>0），公式分不计入总分，仅单票详情展示。"
                )
        for name in gated:
            active_screens.pop(name, None)
    else:
        if provider_kind == "live":
            notes.append("缺本地日线，跳过回测闸门。请先 zk data sync。")
        else:
            notes.append("缺 monthly_prices.csv，跳过回测闸门。")

    formula_all = formula_scores(
        ScreenBundle(bundle.universe_size, bundle.quality_passed, bundle.quality, bundle.screens)
    )
    merge_formula = formula_scores(
        ScreenBundle(bundle.universe_size, bundle.quality_passed, bundle.quality, active_screens)
    )
    if gated:
        notes.append("未过闸公式分不计入总分，只在单票详情（screens_all）展示。")
    if not merge_formula and qlib_scores:
        notes.append("全部公式回测未过闸，总分仅用量化分。")
    elif not merge_formula and not qlib_scores:
        notes.append("全部公式回测未过闸且无量化分，候选为空。")
    rank_icir = qlib_meta.get("rank_icir")
    if rank_icir is None:
        rank_icir = qlib_meta.get("icir")
    try:
        rank_icir_f = float(rank_icir) if rank_icir is not None else None
    except (TypeError, ValueError):
        rank_icir_f = None
    merge_cfg, merge_info = scale_merge_settings(settings.merge, rank_icir_f)
    result.merge_weights = merge_info
    notes.append(merge_info["note"])
    merged = merge_scores(names, merge_formula, qlib_scores, merge_cfg)
    result.screens = {
        name: [
            {"symbol": r.symbol, "name": r.name, "score": r.score, "breakdown": r.breakdown}
            for r in srows
        ]
        for name, srows in bundle.screens.items()
    }
    result.backtests = backtests
    result.gated_out = gated
    result.candidates = [
        {
            "symbol": c.symbol,
            "name": c.name,
            "formula_score": c.formula_score,
            "formula_score_all": average_formula_score(formula_all.get(c.symbol, {})),
            "qlib_score": c.qlib_score,
            "total": c.total,
            "screens": c.screens,
            "screens_all": formula_all.get(c.symbol, {}),
            "source": c.source,
        }
        for c in merged
    ]

    fund_map = by_symbol(rows)
    top_rows: list[Fundamentals] = [
        fund_map[c.symbol] for c in merged[: settings.ai.top_n] if c.symbol in fund_map
    ]
    audits = audit_top(top_rows, settings.ai.masters, top_n=settings.ai.top_n)
    if audits and all(a.skipped for a in audits):
        notes.append(audits[0].skip_reason or "AI 审核已跳过")
    result.audits = [a.to_dict() for a in audits]

    from zhiku.macro.runner import run_macro
    from zhiku.news.runner import run_news

    formula_map = {c.symbol: c.screens for c in merged}
    macro = run_macro(paths, on, provider_kind=provider_kind, settings=settings)
    name_map = {r.name: r.symbol for r in rows if r.name}
    news = run_news(
        paths,
        on,
        provider_kind=provider_kind,
        screen_scores=formula_map,
        qlib_scores=qlib_scores,
        quality_passed=set(bundle.quality_passed),
        settings=settings,
        macro_bundle=macro,
        funds=rows if provider_kind == "live" else None,
        prices=live_prices if provider_kind == "live" else None,
        name_map=name_map if provider_kind == "live" else None,
    )
    result.macro = json.loads(macro.to_json())
    result.news = json.loads(news.to_json())
    notes.extend(macro.notes)
    notes.extend(news.notes)
    result.notes = list(dict.fromkeys(notes))
    return result


def write_daily_reports(paths: Paths, on: date, daily: DailyResult) -> list[Path]:
    paths.ensure_runtime_dirs()
    md_path = write_report(paths.reports, on, "daily", render("daily.md.j2", daily=daily))
    folder = md_path.parent
    html_path = folder / f"{on.isoformat()}-zk-daily.html"
    html_path.write_text(render("daily.html.j2", daily=daily), encoding="utf-8")
    json_path = folder / f"{on.isoformat()}-zk-daily.json"
    payload = daily.to_json() + "\n"
    json_path.write_text(payload, encoding="utf-8")
    latest = paths.reports / "latest-daily.json"
    latest.write_text(payload, encoding="utf-8")
    written = [md_path, html_path, json_path, latest]
    brief_md = render("daily_brief.md.j2", daily=daily)
    lines = brief_md.splitlines()
    if len(lines) > 150:
        brief_md = "\n".join(lines[:150]) + "\n"
    brief_path = write_report(paths.reports, on, "brief", brief_md)
    latest_brief = paths.reports / "latest-brief.md"
    latest_brief.write_text(brief_md, encoding="utf-8")
    written.extend([brief_path, latest_brief])
    from zhiku.macro.runner import MacroBundle, write_macro_reports
    from zhiku.news.runner import NewsBundle, write_news_reports

    if daily.macro:
        written.extend(write_macro_reports(paths, on, MacroBundle(**daily.macro)))
    if daily.news:
        written.extend(write_news_reports(paths, on, NewsBundle(**daily.news)))
    return written
