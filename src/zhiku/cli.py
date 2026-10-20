"""zhiku CLI：check / close / review / exposure / debate。输出只写本地 markdown 和 stdout。"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer

from zhiku.calendar import now_shanghai, parse_now
from zhiku.config import ENV_NOW, Paths
from zhiku.debate.tradingagents_adapter import DebateNotInstalled, run_debate
from zhiku.engine.exposure import ExposureGuard
from zhiku.engine.rules import RulesSchemaError, load_portfolio, load_symbol_books
from zhiku.engine.triggers import EvalMode, TriggerEvaluator
from zhiku.notify.render import render, write_report
from zhiku.review.events import EventLog
from zhiku.review.hitrate import review_week

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="投资智库 zhiku：选股打分、回测、AI 审核、看板，以及触发线纪律。不接券商、不下单。",
)
factor_app = typer.Typer(add_completion=False, help="qlib 因子准备 / 训练 / 预测（可选 extra）。")


def _die(msg: str, code: int = 1) -> None:
    typer.echo(msg, err=True)
    raise typer.Exit(code)


def _parse_date(text: str | None) -> date:
    if not text:
        return now_shanghai().date()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter("日期格式应为 YYYY-MM-DD") from exc


def _make_provider(kind: str, paths: Paths):
    kind = kind.lower()
    if kind == "fixture":
        from zhiku.data.fixture import FixtureProvider

        return FixtureProvider(paths.fixtures)
    if kind == "live":
        from zhiku.data.ashare import AShareProvider
        from zhiku.data.base import LiveProvider
        from zhiku.data.cache import SQLiteCache
        from zhiku.data.global_ import GlobalProvider

        cache = SQLiteCache(paths.cache)
        return LiveProvider(AShareProvider(cache=cache), GlobalProvider())
    _die("未知 --provider，可选 live 或 fixture")


def _latest_quotes(provider, symbols: list[str], on: date | None):
    from zhiku.data.fixture import FixtureProvider
    from zhiku.models import MissingData

    if isinstance(provider, FixtureProvider) and on is None:
        return provider.latest_quotes(symbols)
    quotes = {}
    day = on or now_shanghai().date()
    for symbol in symbols:
        quotes[symbol] = provider.get_quote(symbol, day)
        if isinstance(quotes[symbol], MissingData) and isinstance(provider, FixtureProvider):
            # 指定日无 K 线时退回该标的最后一根，仍不估算中间值
            fallback = provider.latest_quotes([symbol]).get(symbol)
            if fallback is not None:
                quotes[symbol] = fallback
    return quotes


@app.callback()
def main(
    ctx: typer.Context,
    provider: str = typer.Option(
        "live",
        "--provider",
        help="行情源：live=akshare/yfinance；fixture=examples/fixtures 离线 CSV",
    ),
    kb: Optional[Path] = typer.Option(None, "--kb", help="覆盖 ZHIKU_KB"),
    fixtures: Optional[Path] = typer.Option(None, "--fixtures", help="覆盖 ZHIKU_FIXTURES"),
) -> None:
    ctx.ensure_object(dict)
    paths = Paths.load(kb=kb, fixtures=fixtures)
    ctx.obj["paths"] = paths
    ctx.obj["provider_kind"] = provider
    ctx.obj["provider"] = None
    if provider == "live":
        os.environ.setdefault("ZHIKU_LIVE_CALENDAR", "1")


def _provider(ctx: typer.Context):
    if ctx.obj.get("provider") is None:
        ctx.obj["provider"] = _make_provider(ctx.obj["provider_kind"], ctx.obj["paths"])
    return ctx.obj["provider"]


def _clock(now_opt: str | None) -> datetime:
    raw = now_opt if now_opt is not None else os.environ.get(ENV_NOW)
    parsed = parse_now(raw)
    return parsed if parsed is not None else now_shanghai()


def _load_books(paths: Paths):
    try:
        return load_symbol_books(paths.rules)
    except RulesSchemaError as exc:
        _die(str(exc))


def _apply_macro_windows(paths: Paths, on: date, provider_kind: str, evaluations) -> list[str]:
    banners: list[str] = []
    try:
        from zhiku.macro.models import EventWindow
        from zhiku.macro.runner import run_macro
        from zhiku.macro.windows import apply_to_evaluations

        bundle = run_macro(paths, on, provider_kind=provider_kind)
        wins = [EventWindow.model_validate(w) for w in bundle.windows]
        banners.extend(apply_to_evaluations(evaluations, wins, on))
    except Exception:  # noqa: BLE001 — 宏观失败不得阻断触发线
        pass
    try:
        from zhiku.events.invalidation import apply_event_discipline, expiry_alerts
        from zhiku.events.models import Emergency
        from zhiku.news.runner import run_news
        from zhiku.pipeline.settings import load_pipeline_settings

        settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
        news = run_news(paths, on, provider_kind=provider_kind, settings=settings)
        emergency = Emergency.model_validate(news.emergency or {})
        expiries = expiry_alerts(paths.journal / "opportunities.jsonl", on)
        banners.extend(apply_event_discipline(evaluations, emergency=emergency, expiries=expiries))
        from zhiku.events.log import DecisionLog

        DecisionLog(paths.journal / "decisions.jsonl").append(
            "trigger_judgment",
            {
                "asof": on.isoformat(),
                "freeze_adds": bool(emergency.freeze_adds and emergency.active),
                "expiry": [e.reason + " " + e.symbol for e in expiries],
                "symbols": [getattr(ev.book, "symbol", "") for ev in (evaluations or [])],
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return banners


@app.command("check")
def check_cmd(
    ctx: typer.Context,
    at: int = typer.Option(..., "--at", help="目前仅支持 1455（盘中判定）"),
    date_text: Optional[str] = typer.Option(None, "--date", help="评估日 YYYY-MM-DD"),
    now_text: Optional[str] = typer.Option(None, "--now", help="注入当前时间，测试用"),
) -> None:
    """14:55 盘中检查：price_1455 规则正式判定；close 规则仅预警（PREVIEW）。"""
    if at != 1455:
        _die("--at 目前只接受 1455")
    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    books = _load_books(paths)
    replay = ctx.obj["provider_kind"] == "fixture"
    evaluator = TriggerEvaluator(_provider(ctx), now=_clock(now_text), replay=replay)
    evaluations = evaluator.evaluate(books, on, EvalMode.CHECK_1455)
    banners = _apply_macro_windows(paths, on, ctx.obj["provider_kind"], evaluations)
    events = evaluator.events_from(evaluations, on)
    written = EventLog(paths.events).append_many(events)
    md = render(
        "check.md.j2",
        on=on.isoformat(),
        provider=ctx.obj["provider_kind"],
        kb=str(paths.kb),
        evaluations=evaluations,
        warnings=[],
        event_windows=banners,
    )
    out = write_report(paths.reports, on, "check", md)
    typer.echo(md)
    typer.echo(f"已写入 {out}；新事件 {written} 条", err=True)


@app.command("close")
def close_cmd(
    ctx: typer.Context,
    date_text: Optional[str] = typer.Option(None, "--date", help="评估日 YYYY-MM-DD"),
    now_text: Optional[str] = typer.Option(None, "--now", help="注入当前时间，测试用"),
) -> None:
    """收盘确认：评估 close/open/low。fixture + --date 视为回放已收盘日。"""
    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    books = _load_books(paths)
    clock = _clock(now_text)
    # 显式 --date 的 fixture 回放视为该日已收盘；live 或注入 --now 时尊重 16:10。
    replay = ctx.obj["provider_kind"] == "fixture" and now_text is None
    evaluator = TriggerEvaluator(_provider(ctx), now=clock, replay=replay)
    evaluations = evaluator.evaluate(books, on, EvalMode.CLOSE)
    banners = _apply_macro_windows(paths, on, ctx.obj["provider_kind"], evaluations)
    events = evaluator.events_from(evaluations, on)
    written = EventLog(paths.events).append_many(events)
    md = render(
        "close.md.j2",
        on=on.isoformat(),
        provider=ctx.obj["provider_kind"],
        kb=str(paths.kb),
        evaluations=evaluations,
        warnings=[],
        event_windows=banners,
    )
    out = write_report(paths.reports, on, "close", md)
    typer.echo(md)
    typer.echo(f"已写入 {out}；新事件 {written} 条", err=True)


@app.command("exposure")
def exposure_cmd(
    ctx: typer.Context,
    date_text: Optional[str] = typer.Option(None, "--date", help="估值日 YYYY-MM-DD"),
) -> None:
    """成长仓占比闸门（带滞回）。对 add_blocked 标的输出「禁加」。"""
    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    try:
        portfolio = load_portfolio(paths.positions)
    except RulesSchemaError as exc:
        _die(str(exc))
    on = _parse_date(date_text) if date_text else None
    symbols = [p.symbol for p in portfolio.positions]
    quotes = _latest_quotes(_provider(ctx), symbols, on)
    guard = ExposureGuard(paths.exposure_state)
    asof = on or now_shanghai().date()
    snap = guard.evaluate(portfolio, quotes, persist=True, asof=asof)
    try:
        from zhiku.events.models import Emergency
        from zhiku.news.runner import run_news
        from zhiku.pipeline.settings import load_pipeline_settings

        settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
        news = run_news(paths, asof, provider_kind=ctx.obj["provider_kind"], settings=settings)
        emergency = Emergency.model_validate(news.emergency or {})
        if emergency.active and emergency.freeze_adds:
            snap.frozen = True
            snap.status = "FROZEN"
            snap.notes.append("应急模式：禁加")
    except Exception:  # noqa: BLE001
        pass
    md = render("exposure.md.j2", on=asof.isoformat(), snap=snap)
    out = write_report(paths.reports, asof, "exposure", md)
    typer.echo(md)
    blocked = [p.symbol for p in portfolio.positions if p.add_blocked]
    if blocked:
        typer.echo("禁加: " + ", ".join(blocked), err=True)
    typer.echo(f"已写入 {out}", err=True)


@app.command("review")
def review_cmd(
    ctx: typer.Context,
    week: bool = typer.Option(False, "--week", help="统计本周一到周五"),
    end_text: Optional[str] = typer.Option(None, "--end", help="周末截止日 YYYY-MM-DD"),
    rebound_pct: float = typer.Option(0.01, "--rebound-pct"),
) -> None:
    """周五命中率复盘。候选教训只打印，不写入 CIO_LESSONS。"""
    if not week:
        _die("MVP 仅支持 zk review --week")
    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    end = _parse_date(end_text)
    events = EventLog(paths.events).read()
    result = review_week(events, _provider(ctx), end, rebound_pct=rebound_pct)
    script_hits: list[dict] = []
    source_hits: dict = {}
    x_hits: dict = {}
    crowding_stats: dict = {}
    try:
        from zhiku.macro.runner import run_macro
        from zhiku.news.grades import load_followups, source_hitrate
        from zhiku.news.runner import run_news

        macro = run_macro(paths, end, provider_kind=ctx.obj["provider_kind"])
        script_hits = macro.script_hits
        news = run_news(paths, end, provider_kind=ctx.obj["provider_kind"], macro_bundle=macro)
        source_hits = news.source_hits or source_hitrate(
            load_followups(paths.fixtures / "news.yaml")
        )
        x_hits = news.x_hits or {}
        crowding_stats = news.crowding_stats or {}
    except Exception:  # noqa: BLE001
        pass
    md = render(
        "weekly.md.j2",
        review=result,
        provider=ctx.obj["provider_kind"],
        kb=str(paths.kb),
        script_hits=script_hits,
        source_hits=source_hits,
        x_hits=x_hits,
        crowding_stats=crowding_stats,
    )
    out = write_report(paths.reports, result.end, "weekly", md)
    typer.echo(md)
    typer.echo("候选教训（不会自动写入 lessons 文件）：", err=True)
    for lesson in result.lessons:
        typer.echo(f"- {lesson}", err=True)
    typer.echo(f"已写入 {out}", err=True)


@app.command("screen")
def screen_cmd(
    ctx: typer.Context,
    name: str = typer.Argument("all", help="greenblatt / piotroski / dividend / multifactor / all"),
) -> None:
    """运行经典选股公式，输出带分项的候选表。"""
    from zhiku.events.pit import load_asof_universe
    from zhiku.pipeline.daily import _fundamentals_path
    from zhiku.pipeline.settings import load_pipeline_settings
    from zhiku.screens.runner import SCREEN_FUNCS, run_screens

    paths: Paths = ctx.obj["paths"]
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    rows = load_asof_universe(
        fundamentals_path=_fundamentals_path(paths),
        pit_path=paths.fixtures / "pit_fundamentals.yaml",
        delist_path=paths.fixtures / "delisted.yaml",
        asof=now_shanghai().date(),
    )
    names = None if name == "all" else [name]
    if names and names[0] not in SCREEN_FUNCS:
        _die(f"未知公式 {name}，可选：{', '.join(SCREEN_FUNCS)} / all")
    bundle = run_screens(rows, settings, names=names)
    for q in bundle.quality:
        if not q.passed:
            typer.echo(f"[门槛] {q.symbol} 未通过：{'；'.join(q.reasons)}", err=True)
    for sname, srows in bundle.screens.items():
        typer.echo(f"## {sname}")
        for row in srows:
            typer.echo(f"{row.symbol}\t{row.name}\t{row.score:.1f}\t{row.breakdown}")


@app.command("backtest")
def backtest_cmd(
    ctx: typer.Context,
    screen: str = typer.Argument(..., help="公式名，或 trigger 回测触发线"),
) -> None:
    """对选股公式做月度等权回测；`trigger` 回测跌破价位线。"""
    from zhiku.backtest.engine import backtest_screen, backtest_trigger_line, load_price_panel
    from zhiku.events.pit import drop_survivors, load_asof_universe, load_delist_map
    from zhiku.pipeline.daily import _fundamentals_path
    from zhiku.pipeline.settings import load_pipeline_settings
    from zhiku.screens.runner import SCREEN_FUNCS, run_screens

    paths: Paths = ctx.obj["paths"]
    price_file = paths.fixtures / "monthly_prices.csv"
    if not price_file.is_file():
        _die(f"找不到 {price_file}")
    prices = load_price_panel(price_file)
    asof = now_shanghai().date()
    delist = load_delist_map(paths.fixtures / "delisted.yaml")
    if delist:
        prices = drop_survivors(prices, delist, asof)
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    if screen == "trigger":
        if "510300" not in prices.columns:
            _die("触发线示例需要 510300 价格列")
        result = backtest_trigger_line(prices["510300"], 3.80)
    else:
        if screen not in SCREEN_FUNCS:
            _die(f"未知公式 {screen}")
        rows = load_asof_universe(
            fundamentals_path=_fundamentals_path(paths),
            pit_path=paths.fixtures / "pit_fundamentals.yaml",
            delist_path=paths.fixtures / "delisted.yaml",
            asof=asof,
        )
        bundle = run_screens(rows, settings, names=[screen])
        picks = [r.symbol for r in bundle.screens.get(screen, [])[: settings.backtest.top_n]]
        result = backtest_screen(
            prices, picks, settings.backtest, screen=screen, delist=delist, asof=asof
        )
    typer.echo(
        f"{result.screen} 年化={result.annualized} 回撤={result.max_drawdown} "
        f"夏普={result.sharpe} 胜率={result.win_rate} 换手={result.turnover} "
        f"闸门={'通过' if result.passed_gate else '未通过'} 引擎={result.engine}"
    )
    for note in result.notes:
        typer.echo(note, err=True)


@app.command("daily")
def daily_cmd(
    ctx: typer.Context,
    date_text: Optional[str] = typer.Option(None, "--date", help="截面日 YYYY-MM-DD"),
) -> None:
    """一键：选股 → 打分 → 回测闸门 → AI 审核 → markdown/HTML/JSON。可配 cron。"""
    from zhiku.pipeline.daily import run_daily, write_daily_reports

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    daily = run_daily(paths, on, provider_kind=ctx.obj["provider_kind"])
    written = write_daily_reports(paths, on, daily)
    md = next(p for p in written if p.suffix == ".md")
    typer.echo(md.read_text(encoding="utf-8"))
    for path in written:
        typer.echo(f"已写入 {path}", err=True)


@app.command("ui")
def ui_cmd(
    port: int = typer.Option(8501, "--port"),
) -> None:
    """启动本地 Streamlit 看板。"""
    import subprocess
    import sys
    from pathlib import Path as P

    try:
        import streamlit  # noqa: F401
    except ImportError:
        _die('未安装看板 extra。请执行：uv pip install -e ".[ui]"')
    app_path = P(__file__).resolve().parent / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    raise typer.Exit(subprocess.call(cmd))


@factor_app.command("prepare")
def factor_prepare(
    ctx: typer.Context,
    qlib_dir: Path = typer.Option(Path("./qlib_data"), "--qlib-dir"),
) -> None:
    """把本地 CSV 转为 qlib 格式（需 [qlib] extra）。"""
    from zhiku.factors.qlib_adapter import QlibNotInstalled, prepare_from_csv

    paths: Paths = ctx.obj["paths"]
    try:
        msg = prepare_from_csv(paths.fixtures, qlib_dir)
    except QlibNotInstalled as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(0) from None
    typer.echo(msg)


@factor_app.command("train")
def factor_train(
    qlib_dir: Path = typer.Option(Path("./qlib_data"), "--qlib-dir"),
    model: Path = typer.Option(Path("./kb/models/lgb_alpha158.pkl"), "--model"),
) -> None:
    """训练 Alpha158 + LightGBM 基线。"""
    from zhiku.factors.qlib_adapter import QlibNotInstalled, train_alpha158

    try:
        msg = train_alpha158(qlib_dir, model)
    except QlibNotInstalled as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(msg)


@factor_app.command("predict")
def factor_predict() -> None:
    """输出 qlib 分数；未安装则空结果并提示降级。"""
    from zhiku.factors.qlib_adapter import QLIB_HINT, predict_scores, qlib_available

    scores = predict_scores(None, None)
    if not qlib_available():
        typer.echo(QLIB_HINT.strip(), err=True)
        typer.echo("已降级：qlib 分数为空，流水线将只用公式分。")
        raise typer.Exit(0)
    typer.echo(scores or "{}")


@app.command("news")
def news_cmd(
    ctx: typer.Context,
    kind: str = typer.Option("all", "--kind", help="all / premarket-ah / premarket-us / midday"),
    date_text: Optional[str] = typer.Option(None, "--date"),
) -> None:
    """新闻雷达：分析为主体，原文链接附后。无 LLM key 时规则模板降级。"""
    from zhiku.news.runner import run_news, write_news_reports

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    bundle = run_news(
        paths, on, kind=kind.replace("-", "_"), provider_kind=ctx.obj["provider_kind"]
    )
    written = write_news_reports(paths, on, bundle)
    md = next((p for p in written if p.suffix == ".md"), None)
    if md is not None:
        typer.echo(md.read_text(encoding="utf-8"))
    for path in written:
        typer.echo(f"已写入 {path}", err=True)


@app.command("macro")
def macro_cmd(
    ctx: typer.Context,
    date_text: Optional[str] = typer.Option(None, "--date"),
) -> None:
    """宏观事件锚点：日历、预期差、剧本、传导链、事件窗口。"""
    from zhiku.macro.runner import run_macro, write_macro_reports

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    bundle = run_macro(paths, on, provider_kind=ctx.obj["provider_kind"])
    written = write_macro_reports(paths, on, bundle)
    md = next((p for p in written if p.suffix == ".md"), None)
    if md is not None:
        typer.echo(md.read_text(encoding="utf-8"))
    for path in written:
        typer.echo(f"已写入 {path}", err=True)


@app.command("debate")
def debate_cmd(symbol: str) -> None:
    """可选 TradingAgents 辩论。未安装时友好提示并以非零码退出。"""
    try:
        message = run_debate(symbol)
    except DebateNotInstalled as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except Exception as exc:  # noqa: BLE001 — 辩论层失败也不得把 traceback 甩给用户
        typer.echo(f"辩论层调用失败: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(message)


app.add_typer(factor_app, name="factor")


def run() -> None:
    app()


if __name__ == "__main__":
    app()
