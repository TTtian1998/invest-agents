"""zk daily：选股 → 打分 → 回测闸门 → AI 审核 → 报告。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from zhiku.backtest.engine import backtest_screen, load_price_panel, load_price_panel_from_store
from zhiku.config import Paths
from zhiku.events.pit import drop_survivors, load_delist_map
from zhiku.factors.merge import merge_scores
from zhiku.factors.qlib_adapter import model_status, predict_scores, qlib_available
from zhiku.notify.render import render, write_report
from zhiku.pipeline.settings import load_pipeline_settings
from zhiku.pipeline.universe import fundamentals_path, load_pipeline_universe
from zhiku.review.audit import audit_top
from zhiku.screens.fundamentals import Fundamentals, by_symbol
from zhiku.screens.runner import ScreenBundle, formula_scores, run_screens


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
    elif not qlib_available() and qlib_meta.get("kind") != "lgb_lite":
        notes.append("未安装 [qlib]，若本地已有 LightGBM 模型仍会出分。")
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
    if provider_kind == "live" and store is not None and not store.is_empty():
        prices = load_price_panel_from_store(store, [r.symbol for r in rows] + [settings.backtest.benchmark])
        limit_up, limit_down = store.limit_panels(list(prices.columns) if not prices.empty else [])
        if delist and prices is not None and not prices.empty:
            prices = drop_survivors(prices, delist, on)
            notes.append("价格面板来自本地库（前复权），已按退市日截断。")
        if prices is None or prices.empty:
            notes.append("本地库缺日线，跳过回测闸门。请先 zk data sync。")
            prices = None
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
    if prices is not None:
        for name, srows in bundle.screens.items():
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
            )
            backtests[name] = bt.to_dict()
            if not bt.passed_gate:
                gated.append(name)
                notes.append(f"公式 {name} 回测未达标，不进入每日候选。")
        for name in gated:
            active_screens.pop(name, None)
    else:
        if provider_kind == "live":
            notes.append("缺本地日线，跳过回测闸门。请先 zk data sync。")
        else:
            notes.append("缺 monthly_prices.csv，跳过回测闸门。")

    formula = formula_scores(
        ScreenBundle(bundle.universe_size, bundle.quality_passed, bundle.quality, active_screens)
    )
    merged = merge_scores(names, formula, qlib_scores, settings.merge)
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
            "qlib_score": c.qlib_score,
            "total": c.total,
            "screens": c.screens,
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
    news = run_news(
        paths,
        on,
        provider_kind=provider_kind,
        screen_scores=formula_map,
        qlib_scores=qlib_scores,
        quality_passed=set(bundle.quality_passed),
        settings=settings,
        macro_bundle=macro,
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
    from zhiku.macro.runner import MacroBundle, write_macro_reports
    from zhiku.news.runner import NewsBundle, write_news_reports

    if daily.macro:
        written.extend(write_macro_reports(paths, on, MacroBundle(**daily.macro)))
    if daily.news:
        written.extend(write_news_reports(paths, on, NewsBundle(**daily.news)))
    return written
