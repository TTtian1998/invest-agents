"""Daily execution: current data + approved artifacts, never historical research."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from zhiku.config import Paths
from zhiku.factors.merge import merge_scores, scale_merge_settings
from zhiku.factors.qlib_adapter import predict_published_scores
from zhiku.pipeline.artifacts import config_fingerprint, fingerprint, load_approved_gates
from zhiku.pipeline.sources import claim_source_mode
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
    insufficient: list[str] = field(default_factory=list)
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
    ai_layer: dict = field(default_factory=dict)
    review_pack: str | None = None
    source_mode: str = "demo"
    snapshot_id: str = ""
    gate_artifact: dict = field(default_factory=dict)
    observations: list[dict] = field(default_factory=list)

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
    mode = claim_source_mode(paths.kb, provider_kind)
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    rows, delist, uni_notes = load_pipeline_universe(
        paths, on, provider_kind=provider_kind, settings=settings
    )
    names = {r.symbol: r.name for r in rows}
    bundle = run_screens(rows, settings, names=screen_names)
    notes: list[str] = list(uni_notes)
    if mode != "live" and settings.ai.backend not in {"external", "mock"}:
        settings.ai.backend = "external"
        notes.append("离线/停用模式：AI 只导出任务包，不自动调用网络后端。")
    context = config_fingerprint(settings)
    store = None
    if mode == "live":
        from zhiku.data.store import MarketStore
        store = MarketStore(paths.data_dir)
    qlib_scores, qlib_meta = predict_published_scores(paths.models_dir, store=store, asof=on,
        symbols=[r.symbol for r in rows],
        provider_kind=mode, config_fingerprint=context,
        max_age_days=settings.artifacts.model_max_age_days)
    qlib_meta["inference_available"] = bool(qlib_scores)
    if not qlib_scores and qlib_meta.get("available"):
        qlib_meta["inference_reason"] = "no_current_features_or_scores"
    if not qlib_scores:
        notes.append(f"量化辅助不可用：{qlib_meta.get('inference_reason') or qlib_meta.get('reason') or qlib_meta.get('status')}；不自动训练或回退其他模型。")
    result = DailyResult(asof=on.isoformat(), fictional=mode != "live", source_mode=mode)
    result.universe_size = len(rows)
    result.qlib = qlib_meta
    result.quality_passed = bundle.quality_passed
    result.quality_rejected = [{"symbol": q.symbol, "reasons": q.reasons} for q in bundle.quality if not q.passed]
    from zhiku.data.compute import missing_ratio as _miss
    result.missing_ratio = _miss(rows) if rows else None
    if result.missing_ratio is not None:
        notes.append(f"截面缺数据比例 {result.missing_ratio:.1%}")

    live_prices: dict[str, float] = {}
    fresh_spots: set[str] = set()
    if store is not None and rows:
        from datetime import timedelta
        from zhiku.data.universe_live import _spot_map
        fresh_spots = set(_spot_map(store, on, settings.artifacts.price_max_age_days))
        bars = store.bars([r.symbol for r in rows], start=on - timedelta(days=settings.artifacts.price_max_age_days), end=on, adjust="none")
        if not bars.empty:
            for symbol, part in bars.groupby("symbol"):
                close = part.sort_values("date").iloc[-1].get("close")
                try:
                    valid_close = float(close)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(valid_close) and valid_close > 0:
                    live_prices[str(symbol)] = valid_close

    acceptance, gate_info = load_approved_gates(paths, on=on, provider_kind=mode, settings=settings)
    result.gate_artifact = gate_info
    backtests = acceptance.get("backtests", {})
    result.pit_universe = acceptance.get("provenance", {}).get("universe", {})
    statuses = gate_info.get("statuses", {})
    approved = set(gate_info["approved_screens"])
    active_screens = {name: values for name, values in bundle.screens.items() if name in approved}
    gated = [name for name in bundle.screens if statuses.get(name) == "failed"]
    insufficient = [name for name in bundle.screens if name not in approved and name not in gated]
    if not approved:
        notes.append(f"正式候选关闭：公式验收 {gate_info['reason']}。请运行 zk research gates 后显式 publish-gates；研究观察不代表过闸。")
    else:
        notes.append(f"只使用已发布公式验收 {gate_info.get('version')}（{gate_info.get('evaluated_asof')}）；daily 不重跑历史闸门。")
    formula_all = formula_scores(bundle)
    merge_formula = formula_scores(ScreenBundle(bundle.universe_size, bundle.quality_passed, bundle.quality, active_screens))
    # Unversioned live_rank_ic.json is research evidence, not publication authority.
    merge_cfg, merge_info = scale_merge_settings(settings.merge, None)
    merge_cfg = merge_cfg.model_copy(update={"formula_first": True})
    merge_info["formula_first"] = True
    result.merge_weights = merge_info
    notes.append(merge_info["note"])
    if mode == "live":
        stale = set(merge_formula) - (set(live_prices) & fresh_spots)
        merge_formula = {symbol: scores for symbol, scores in merge_formula.items() if symbol in live_prices and symbol in fresh_spots}
        if stale:
            notes.append(f"{len(stale)} 只标的缺 {settings.artifacts.price_max_age_days} 天内有效正数价格或新鲜估值截面，不进入正式候选。")
    if mode == "live":
        from zhiku.calendar import now_shanghai
        if on != now_shanghai().date():
            merge_formula = {}
            notes.append("历史/未来 live 截面完整 PIT 来源尚未验证，仅输出研究观察，不生成正式候选。")
    merged = merge_scores(names, merge_formula, qlib_scores, merge_cfg)
    observation_scores = merge_scores(names, formula_all, qlib_scores, merge_cfg.model_copy(update={"formula_first": False}))
    result.observations = [{"symbol": c.symbol, "name": c.name, "formula_score": c.formula_score,
        "qlib_score": c.qlib_score, "screens": c.screens, "source": c.source,
        "eligible": c.symbol in merge_formula, "status": "research_only"} for c in observation_scores]

    result.screens = {
        name: [
            {"symbol": r.symbol, "name": r.name, "score": r.score, "breakdown": r.breakdown}
            for r in srows
        ]
        for name, srows in bundle.screens.items()
    }
    result.backtests = backtests
    result.gated_out = gated
    result.insufficient = insufficient
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
    from zhiku.ai.runner import run_ai_layer

    ai_result, pack_path, ai_notes = run_ai_layer(
        asof=on,
        paths=paths,
        settings=settings,
        candidates=result.candidates,
        rows=rows,
        news=result.news,
        macro=result.macro,
        trailing=None,
    )
    notes.extend(ai_notes)
    if pack_path is not None:
        result.review_pack = str(pack_path)
    if ai_result is not None:
        result.ai_layer = ai_result.to_dict()
    result.notes = list(dict.fromkeys(notes))
    # Snapshot identifies data/decisions, not transient filesystem paths or report rendering.
    result.snapshot_id = fingerprint({"asof": result.asof, "mode": mode, "config": context,
        "gate": gate_info, "model": qlib_meta, "candidates": result.candidates,
        "observations": result.observations, "review_pack": Path(result.review_pack).name if result.review_pack else None})
    return result


def write_daily_reports(paths: Paths, on: date, daily: DailyResult, *, write_latest: bool = True, write_sources: bool = True) -> list[Path]:
    paths.ensure_runtime_dirs()
    md_path = write_report(paths.reports, on, "daily", render("daily.md.j2", daily=daily))
    folder = md_path.parent
    html_path = folder / f"{on.isoformat()}-zk-daily.html"
    html_path.write_text(render("daily.html.j2", daily=daily), encoding="utf-8")
    json_path = folder / f"{on.isoformat()}-zk-daily.json"
    payload = daily.to_json() + "\n"
    json_path.write_text(payload, encoding="utf-8")
    snapshots = paths.reports / "daily-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    version = fingerprint(json.loads(payload))
    snapshot = snapshots / f"{on.isoformat()}-{version}.json"
    if not snapshot.exists():
        snapshot.write_text(payload, encoding="utf-8")
    latest = paths.reports / "latest-daily.json"
    written = [md_path, html_path, json_path, snapshot]
    if write_latest:
        latest.write_text(payload, encoding="utf-8")
        written.append(latest)
    brief_md = render("daily_brief.md.j2", daily=daily)
    lines = brief_md.splitlines()
    if len(lines) > 150:
        brief_md = "\n".join(lines[:150]) + "\n"
    brief_path = write_report(paths.reports, on, "brief", brief_md)
    latest_brief = paths.reports / "latest-brief.md"
    written.append(brief_path)
    if write_latest:
        latest_brief.write_text(brief_md, encoding="utf-8")
        written.append(latest_brief)
    from zhiku.macro.runner import MacroBundle, write_macro_reports
    from zhiku.news.runner import NewsBundle, write_news_reports

    if write_sources and daily.macro:
        written.extend(write_macro_reports(paths, on, MacroBundle(**daily.macro)))
    if write_sources and daily.news:
        written.extend(write_news_reports(paths, on, NewsBundle(**daily.news)))
    return written


def apply_review_to_daily(paths: Paths, result) -> bool:
    """Render an ingested decision against its saved daily, without collecting again.

    A newer daily may already exist. Never attach an old decision to that newer
    evidence; it remains available in the journal and immutable review snapshot.
    """
    on = date.fromisoformat(result.asof)
    dated = paths.reports / on.strftime("%Y-%m") / f"{on.isoformat()}-zk-daily.json"
    if not dated.is_file():
        return False
    raw = json.loads(dated.read_text(encoding="utf-8"))
    if raw.get("asof") != result.asof or Path(raw.get("review_pack") or "").stem != result.snapshot_id:
        return False
    daily = DailyResult(**raw)
    daily.ai_layer = result.to_dict()
    # Updating the exact dated report must not move latest backwards in time.
    latest = paths.reports / "latest-daily.json"
    update_latest = True
    if latest.is_file():
        current = json.loads(latest.read_text(encoding="utf-8"))
        update_latest = current.get("asof", "") <= result.asof
    write_daily_reports(paths, on, daily, write_latest=update_latest, write_sources=False)
    return True
