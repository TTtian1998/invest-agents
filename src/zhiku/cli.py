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
research_app = typer.Typer(add_completion=False, help="历史验收 / 显式发布，与 daily 分离。")
data_app = typer.Typer(add_completion=False, help="全市场数据同步到 ZHIKU_KB/data（不进 git）。")


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
    if kind in {"fixture", "demo"}:
        from zhiku.data.fixture import FixtureProvider

        return FixtureProvider(paths.fixtures)
    if kind == "live":
        from zhiku.data.ashare import AShareProvider
        from zhiku.data.base import LiveProvider
        from zhiku.data.cache import SQLiteCache
        from zhiku.data.global_ import GlobalProvider

        cache = SQLiteCache(paths.cache)
        return LiveProvider(AShareProvider(cache=cache), GlobalProvider())
    _die("此命令需要行情源 live 或 demo（fixture 是兼容别名）")


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
        help="数据源：live / demo（离线）；fixture=demo 兼容别名；disabled=停用流水线来源",
    ),
    kb: Optional[Path] = typer.Option(None, "--kb", help="覆盖 ZHIKU_KB"),
    fixtures: Optional[Path] = typer.Option(None, "--fixtures", help="覆盖 ZHIKU_FIXTURES"),
) -> None:
    ctx.ensure_object(dict)
    paths = Paths.load(kb=kb, fixtures=fixtures)
    ctx.obj["paths"] = paths
    from zhiku.pipeline.sources import resolve_source_mode
    try:
        mode = resolve_source_mode(provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    from zhiku.pipeline.sources import claim_source_mode
    from zhiku.calendar import calendar_network
    if ctx.invoked_subcommand != "ui":
        try:
            claim_source_mode(paths.kb, mode)
        except (ValueError, OSError) as exc:
            _die(str(exc))
    ctx.with_resource(calendar_network(enabled=mode == "live"))
    ctx.obj["provider_kind"] = "fixture" if mode == "demo" else mode
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
        source_hits = news.source_hits or {}
        if ctx.obj["provider_kind"] == "fixture" and not source_hits:
            source_hits = source_hitrate(load_followups(paths.fixtures / "news.yaml"))
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
    name: str = typer.Argument("all", help="greenblatt / piotroski / dividend / multifactor / graham / lynch_peg / canslim / livermore / fisher / all"),
) -> None:
    """运行经典选股公式，输出带分项的候选表。"""
    from zhiku.pipeline.settings import load_pipeline_settings
    from zhiku.pipeline.universe import load_pipeline_universe
    from zhiku.screens.runner import SCREEN_FUNCS, run_screens

    paths: Paths = ctx.obj["paths"]
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    rows, _, notes = load_pipeline_universe(
        paths, now_shanghai().date(), provider_kind=ctx.obj["provider_kind"], settings=settings
    )
    for n in notes:
        typer.echo(n, err=True)
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
    """对选股公式做月度等权回测；`trigger` 回测跌破价位线。live 用本地库价格。"""
    if ctx.obj["provider_kind"] == "disabled":
        _die("数据源 disabled，不能运行需要价格的回测；请显式选择 live 或 demo。")
    from zhiku.backtest.engine import (
        backtest_screen,
        backtest_trigger_line,
        load_price_panel,
        load_price_panel_from_store,
    )
    from zhiku.events.pit import drop_survivors, load_delist_map
    from zhiku.pipeline.settings import load_pipeline_settings
    from zhiku.pipeline.universe import load_pipeline_universe
    from zhiku.screens.runner import SCREEN_FUNCS, run_screens

    paths: Paths = ctx.obj["paths"]
    asof = now_shanghai().date()
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    limit_up = limit_down = None
    if ctx.obj["provider_kind"] == "live":
        from zhiku.data.store import MarketStore

        store = MarketStore(paths.data_dir)
        prices = load_price_panel_from_store(store)
        if prices.empty:
            _die("本地库无日线，请先 zk data sync")
        delist = store.delist_map()
        limit_up, limit_down = store.limit_panels(list(prices.columns))
    else:
        price_file = paths.fixtures / "monthly_prices.csv"
        if not price_file.is_file():
            _die(f"找不到 {price_file}")
        prices = load_price_panel(price_file)
        delist = load_delist_map(paths.fixtures / "delisted.yaml")
    if delist:
        prices = drop_survivors(prices, delist, asof)
    if screen == "trigger":
        col = "510300" if "510300" in prices.columns else None
        if col is None:
            _die("触发线回测需要 510300 价格列（fixture 示例或已同步 ETF）")
        result = backtest_trigger_line(prices[col], 3.80)
    else:
        if screen not in SCREEN_FUNCS:
            _die(f"未知公式 {screen}")
        rows, delist2, _notes = load_pipeline_universe(
            paths, asof, provider_kind=ctx.obj["provider_kind"], settings=settings
        )
        delist = delist or delist2
        bundle = run_screens(rows, settings, names=[screen])
        picks = [r.symbol for r in bundle.screens.get(screen, [])[: settings.backtest.top_n]]
        result = backtest_screen(
            prices,
            picks,
            settings.backtest,
            screen=screen,
            delist=delist,
            asof=asof,
            limit_up=limit_up,
            limit_down=limit_down,
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
    """每日执行：读已发布验收/模型 → 正式候选/研究观察 → 单次 AI 审核 → 报告。"""
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


@app.command("run-daily")
def run_daily_cmd(
    ctx: typer.Context,
    date_text: Optional[str] = typer.Option(None, "--date", help="截面日 YYYY-MM-DD"),
    force: bool = typer.Option(False, "--force", help="非交易日也跑"),
    skip_sync: bool = typer.Option(False, "--skip-sync"),
    skip_train: bool = typer.Option(False, "--skip-train", help="已弃用兼容参数；daily 永不自动训练"),
) -> None:
    """一条命令：增量同步 → 新闻宏观 → daily → 精简摘要。非交易日退出 0。"""
    import time

    from zhiku.calendar import is_trading_day
    from zhiku.data.store import MarketStore
    from zhiku.data.sync import sync_market
    from zhiku.models import Market
    from zhiku.pipeline.daily import run_daily, write_daily_reports
    from zhiku.pipeline.settings import load_pipeline_settings

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    if not force and not is_trading_day(Market.CN, on):
        typer.echo(f"非交易日 {on}，跳过。", err=True)
        raise typer.Exit(0)
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    timings: dict[str, float] = {}
    t_all = time.monotonic()
    store = MarketStore(paths.data_dir)
    if not skip_sync and ctx.obj["provider_kind"] == "live":
        t0 = time.monotonic()
        typer.echo("== 增量同步 ==", err=True)
        report = sync_market(
            store,
            universe=settings.data.universe,
            years=settings.data.years,
            resume=True,
            incremental=True,
            lookback_days=settings.data.incremental_lookback_days,
            include_global=False,
            extra_symbols=settings.data.extra_symbols,
            sleep=settings.data.sleep_seconds,
            retries=settings.data.retries,
            workers=4,
            detail_financials=settings.data.detail_financials,
            hist_financials=False,
            progress=lambda m: typer.echo(m, err=True),
        )
        timings["sync"] = time.monotonic() - t0
        typer.echo(report.summary(), err=True)
    if skip_train:
        typer.echo("--skip-train 已弃用：daily 不训练；请显式运行 factor train / factor publish。", err=True)
    if ctx.obj["provider_kind"] != "live":
        typer.echo("离线/停用模式：不做 live 数据同步。", err=True)
    t0 = time.monotonic()
    typer.echo("== daily ==", err=True)
    daily = run_daily(paths, on, provider_kind=ctx.obj["provider_kind"])
    timings["daily"] = time.monotonic() - t0
    timings["total"] = time.monotonic() - t_all
    for key, sec in timings.items():
        typer.echo(f"耗时 {key}={sec:.1f}s", err=True)
        daily.notes.append(f"耗时 {key}={sec:.1f}s")
    written = write_daily_reports(paths, on, daily)
    brief = next((p for p in written if p.name.endswith("zk-brief.md")), None)
    if brief is not None:
        typer.echo(brief.read_text(encoding="utf-8"))
    for path in written:
        typer.echo(f"已写入 {path}", err=True)
    raise typer.Exit(0)


@app.command("ui")
def ui_cmd(
    ctx: typer.Context,
    port: int = typer.Option(8501, "--port", min=1, max=65535),
    host: str = typer.Option("127.0.0.1", "--host", help="默认仅本机访问；此服务不提供身份认证"),
    allow_host: Optional[list[str]] = typer.Option(None, "--allow-host", help="显式允许反向代理 Host，可重复"),
    legacy: bool = typer.Option(False, "--legacy", help="启动旧版 Streamlit 看板"),
) -> None:
    """启动只读投研工作台，不采集、不训练、不调用 AI、不写报告。"""
    import sys

    paths: Paths = ctx.obj["paths"]
    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo("警告：工作台无身份认证，不应直接暴露到公网；请配置受保护的反向代理。", err=True)
    if legacy:
        import subprocess
        try:
            import streamlit  # noqa: F401
        except ImportError:
            _die('未安装旧版看板依赖。请执行：uv pip install -e ".[legacy-ui]"')
        app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
        env = {**os.environ, "ZHIKU_KB": str(paths.kb), "ZHIKU_FIXTURES": str(paths.fixtures)}
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
               "--server.port", str(port), "--server.address", host, "--server.headless", "true"]
        raise typer.Exit(subprocess.call(cmd, env=env))
    try:
        import uvicorn
        from zhiku.app.workspace import create_app
    except ImportError:
        _die('未安装 Web 工作台依赖。请执行：uv pip install -e ".[ui]"；旧版使用 zk ui --legacy')
    hosts = ["localhost", "127.0.0.1", "[::1]"]
    for value in allow_host or []:
        if not value or any(char in value for char in "/*:@ "):
            raise typer.BadParameter("--allow-host 仅接受明确的主机名，不接受通配符、协议或端口")
        hosts.append(value)
    if host not in {"0.0.0.0", "::", "::1"}:
        hosts.append(host)
    typer.echo(f"只读投研工作台：http://{host}:{port}（报告来源以页面标记为准）")
    uvicorn.run(create_app(paths, allowed_hosts=hosts), host=host, port=port, proxy_headers=False)


@factor_app.command("prepare")
def factor_prepare(
    ctx: typer.Context,
    qlib_dir: Optional[Path] = typer.Option(None, "--qlib-dir"),
) -> None:
    """把本地库日线转为 qlib bin；无库时尝试 fixtures CSV。"""
    from zhiku.data.store import MarketStore
    from zhiku.factors.qlib_adapter import prepare_from_csv, prepare_from_store

    paths: Paths = ctx.obj["paths"]
    dest = qlib_dir or (paths.data_dir / "qlib")
    store = MarketStore(paths.data_dir)
    if not store.is_empty():
        typer.echo(prepare_from_store(store, dest))
        return
    if ctx.obj["provider_kind"] != "fixture":
        _die("本地库无日线；live/disabled 不回退 fixture，请先 data sync。")
    try:
        msg = prepare_from_csv(paths.fixtures, dest)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(msg)


@factor_app.command("train")
def factor_train(
    ctx: typer.Context,
    qlib_dir: Optional[Path] = typer.Option(None, "--qlib-dir"),
    model: Optional[Path] = typer.Option(None, "--model"),
) -> None:
    """生成研究模型并明确实际实现；不会发布到 daily。"""
    from zhiku.data.store import MarketStore
    from zhiku.factors.qlib_adapter import QlibNotInstalled, train_and_select

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    dest = qlib_dir or (paths.data_dir / "qlib")
    store = MarketStore(paths.data_dir)
    if model is not None:
        _die("--model 不再被忽略：train 输出研究候选；请用 factor publish PATH 显式发布。")
    try:

        from zhiku.pipeline.artifacts import config_fingerprint
        from zhiku.pipeline.settings import load_pipeline_settings
        settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
        winner = train_and_select(qlib_dir=dest, models_dir=paths.models_dir, store=store,
            provider_kind=ctx.obj["provider_kind"], config_fingerprint=config_fingerprint(settings))
        msg = (
            f"研究训练完成（尚未发布）：{winner.get('kind')}；"
            f"IC={winner.get('ic')} RankIC={winner.get('rank_ic')} ICIR={winner.get('icir')}；产物={winner.get('path')}"
        )
    except QlibNotInstalled as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"训练失败: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(msg)


@factor_app.command("research")
def factor_research(
    ctx: typer.Context,
    out_dir: Optional[Path] = typer.Option(None, "--out", help="实验日志目录，默认 artifacts"),
    asof_text: Optional[str] = typer.Option(None, "--asof"),
) -> None:
    """单变量消融：验证集决定去留，测试集只看一次。产出不进 git。"""
    from zhiku.data.store import MarketStore
    from zhiku.factors.experiment import run_ablation

    paths: Paths = ctx.obj["paths"]
    store = MarketStore(paths.data_dir)
    dest = out_dir or Path("/opt/cursor/artifacts/zhiku-quant-research")
    asof = _parse_date(asof_text)
    payload = run_ablation(store, asof=asof, out_dir=dest, models_dir=paths.models_dir)
    kept = payload["payload"].get("kept") or []
    typer.echo(
        f"完成 {payload['payload'].get('n_experiments')} 个实验；"
        f"测试集查看次数={payload['payload'].get('test_looks')}；"
        f"保留={kept}；锁={payload['payload'].get('locked')}；日志={dest}"
    )


@factor_app.command("research-round2")
def factor_research_round2(
    ctx: typer.Context,
    out_dir: Optional[Path] = typer.Option(None, "--out", help="实验日志目录，默认 artifacts/round2"),
    asof_text: Optional[str] = typer.Option(None, "--asof"),
) -> None:
    """第二轮单变量：相对锁定 z-score；测试集只看一次。产出不进 git。"""
    from zhiku.data.store import MarketStore
    from zhiku.factors.experiment import run_round2

    paths: Paths = ctx.obj["paths"]
    store = MarketStore(paths.data_dir)
    dest = out_dir or Path("/opt/cursor/artifacts/zhiku-quant-round2")
    asof = _parse_date(asof_text)
    payload = run_round2(store, asof=asof, out_dir=dest, models_dir=paths.models_dir)
    kept = payload["payload"].get("kept") or []
    typer.echo(
        f"第二轮完成 {payload['payload'].get('n_experiments')} 个实验；"
        f"测试集查看次数={payload['payload'].get('test_looks')}；"
        f"保留={kept}；锁={payload['payload'].get('locked')}；日志={dest}"
    )


@factor_app.command("research-round3")
def factor_research_round3(
    ctx: typer.Context,
    out_dir: Optional[Path] = typer.Option(None, "--out", help="实验日志目录，默认 artifacts/round3"),
    asof_text: Optional[str] = typer.Option(None, "--asof"),
) -> None:
    """第三轮：T+20 行业中性超额 + walk-forward 多折。测试集只看一次。产出不进 git。"""
    from zhiku.data.store import MarketStore
    from zhiku.factors.experiment import run_round3

    paths: Paths = ctx.obj["paths"]
    store = MarketStore(paths.data_dir)
    dest = out_dir or Path("/opt/cursor/artifacts/zhiku-quant-round3")
    asof = _parse_date(asof_text)
    payload = run_round3(store, asof=asof, out_dir=dest, models_dir=paths.models_dir)
    kept = payload["payload"].get("kept") or []
    typer.echo(
        f"第三轮完成 {payload['payload'].get('n_experiments')} 个实验；"
        f"测试集查看次数={payload['payload'].get('test_looks')}；"
        f"保留={kept}；锁={payload['payload'].get('locked')}；日志={dest}"
    )


@factor_app.command("style-attrib")
def factor_style_attrib(
    ctx: typer.Context,
    out_dir: Optional[Path] = typer.Option(None, "--out", help="归因目录，默认 artifacts/style-attrib"),
    asof_text: Optional[str] = typer.Option(None, "--asof"),
) -> None:
    """dividend/graham 分年+行业超额，必要时试 EW 12-1 开关。测试集只看一次。"""
    from zhiku.data.store import MarketStore
    from zhiku.backtest.style_attrib import run_style_attribution
    from zhiku.pipeline.settings import load_pipeline_settings

    paths: Paths = ctx.obj["paths"]
    store = MarketStore(paths.data_dir)
    dest = out_dir or Path("/opt/cursor/artifacts/zhiku-style-attrib")
    asof = _parse_date(asof_text)
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    payload = run_style_attribution(store, settings, asof=asof, out_dir=dest)
    typer.echo(
        f"风格归因完成；轮动={payload.get('rotation')}；"
        f"开关测试查看={(payload.get('regime') or {}).get('test_looks')}；日志={dest}"
    )


@factor_app.command("publish")
def factor_publish(
    ctx: typer.Context,
    model_path: Path = typer.Argument(..., exists=True, readable=True),
    date_text: Optional[str] = typer.Option(None, "--date"),
) -> None:
    """验证并发布指定研究模型版本；失败保留上个有效发布。"""
    from zhiku.factors.qlib_adapter import publish_model
    from zhiku.pipeline.artifacts import config_fingerprint
    from zhiku.pipeline.settings import load_pipeline_settings
    paths: Paths = ctx.obj["paths"]
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    try:
        result = publish_model(paths.models_dir, model_path, asof=_parse_date(date_text),
            provider_kind=ctx.obj["provider_kind"], config_fingerprint=config_fingerprint(settings),
            max_age_days=settings.artifacts.model_max_age_days)
    except (ValueError, OSError, TypeError) as exc:
        _die(f"模型发布被拒绝：{exc}")
    typer.echo(result)


@factor_app.command("predict")
def factor_predict(ctx: typer.Context, date_text: Optional[str] = typer.Option(None, "--date")) -> None:
    """只用兼容且未过期的已发布模型预测，不自动选择研究文件。"""
    from zhiku.data.store import MarketStore
    from zhiku.factors.qlib_adapter import predict_published_scores
    from zhiku.pipeline.artifacts import config_fingerprint
    from zhiku.pipeline.settings import load_pipeline_settings
    paths: Paths = ctx.obj["paths"]
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    on = _parse_date(date_text)
    context = config_fingerprint(settings)
    store = MarketStore(paths.data_dir)
    scores, meta = predict_published_scores(paths.models_dir, store=store, asof=on, provider_kind=ctx.obj["provider_kind"], config_fingerprint=context, max_age_days=settings.artifacts.model_max_age_days)
    if not meta.get("available"):
        typer.echo(f"模型不可用：{meta.get('reason')}。先训练/验证，再 factor publish PATH。", err=True)
        raise typer.Exit(0)
    typer.echo(scores or "{}")


@research_app.command("gates")
def research_gates(ctx: typer.Context, date_text: Optional[str] = typer.Option(None, "--date")) -> None:
    """从本地时点历史计算公式验收证据，不联网、不发布。"""
    from zhiku.pipeline.research import evaluate_gates
    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    typer.echo(str(evaluate_gates(paths, _parse_date(date_text), provider_kind=ctx.obj["provider_kind"])))


@research_app.command("publish-gates")
def publish_gates(ctx: typer.Context, path: Path = typer.Argument(..., exists=True, readable=True), date_text: Optional[str] = typer.Option(None, "--date")) -> None:
    """重验完整证据并发布公式准入版本；不接受裸 passed=true。"""
    from zhiku.pipeline.artifacts import publish_gate_research
    from zhiku.pipeline.settings import load_pipeline_settings
    paths: Paths = ctx.obj["paths"]
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    try:
        result = publish_gate_research(paths, path, on=_parse_date(date_text), provider_kind=ctx.obj["provider_kind"], settings=settings)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        _die(f"公式发布被拒绝：{exc}")
    typer.echo(result)


@app.command("news")
def news_cmd(
    ctx: typer.Context,
    kind: str = typer.Option("all", "--kind", help="all / premarket-ah / premarket-us / midday"),
    date_text: Optional[str] = typer.Option(None, "--date"),
    research: bool = typer.Option(False, "--research", help="显式重算新闻/X 历史统计"),
) -> None:
    """新闻雷达：确定性分析；--research 显式更新历史统计/权重。"""
    from zhiku.news.runner import run_news, write_news_reports

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    on = _parse_date(date_text)
    bundle = run_news(
        paths, on, kind=kind.replace("-", "_"), provider_kind=ctx.obj["provider_kind"], include_research=research
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


@app.command("export-review")
def export_review_cmd(
    ctx: typer.Context,
    date_text: Optional[str] = typer.Option(None, "--date"),
) -> None:
    """导出外部研判任务包（候选、截面、新闻摘录、大师清单）。"""
    from zhiku.ai.pack import load_review_snapshot
    paths: Paths = ctx.obj["paths"]
    on = _parse_date(date_text)
    try:
        pack = load_review_snapshot(paths.reports, asof=on.isoformat())
    except (ValueError, OSError, KeyError) as exc:
        _die(f"没有可导出的有效快照，请先运行 daily：{exc}")
    typer.echo(str(paths.reports / "review-snapshots" / f"{pack['snapshot_id']}.json"))


@app.command("ingest-review")
def ingest_review_cmd(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, readable=True, help="zhiku.review_result.v1 JSON"),
) -> None:
    """把外部研判结果写入决策日志。"""
    from zhiku.ai.journal import DecisionJournal
    from zhiku.ai.pack import load_review_result, load_review_snapshot, validate_review_result

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    try:
        result = load_review_result(path)
        pack = load_review_snapshot(paths.reports, asof=result.asof, snapshot_id=result.snapshot_id)
        result = validate_review_result(result, pack)
    except (ValueError, OSError, KeyError) as exc:
        _die(f"研判结果被拒绝：{exc}")
    try:
        n = DecisionJournal(paths.journal / "ai_decisions.jsonl").append_result(
            result, pack=pack, available={"source": str(path)}
        )
    except ValueError as exc:
        _die(f"研判写入被拒绝：{exc}")
    from zhiku.pipeline.daily import apply_review_to_daily
    rendered = apply_review_to_daily(paths, result)
    typer.echo(f"写入 {n} 条决策日志（asof={result.asof} backend={result.backend}）；" +
               ("已更新匹配快照日报（未重跑 daily）" if rendered else "无当前匹配日报，保留历史决策"))


@app.command("review-scorecard")
def review_scorecard_cmd(
    ctx: typer.Context,
    formula_top_n: int = typer.Option(10, "--formula-top"),
) -> None:
    """AI 研究跟进/回避 在 T+5、T+20 对同池等权的超额，对照公式排名。"""
    from zhiku.ai.journal import DecisionJournal
    from zhiku.ai.scorecard import build_scorecard
    from zhiku.backtest.engine import load_price_panel, load_price_panel_from_store

    paths: Paths = ctx.obj["paths"]
    journal = DecisionJournal(paths.journal / "ai_decisions.jsonl")
    prices = None
    if ctx.obj["provider_kind"] == "live":
        from zhiku.data.store import MarketStore

        store = MarketStore(paths.data_dir)
        prices = load_price_panel_from_store(store)
    elif ctx.obj["provider_kind"] == "fixture":
        csv = paths.fixtures / "monthly_prices.csv"
        if csv.is_file():
            prices = load_price_panel(csv)
    if prices is None or prices.empty:
        _die("缺价格面板，无法做 scorecard")
    card = build_scorecard(journal, prices, formula_top_n=formula_top_n)
    typer.echo(f"日志截止 {card.asof_max or '空'}")
    for note in card.notes:
        typer.echo(note, err=True)
    for gname, hz in card.groups.items():
        for h, stat in hz.items():
            ex = "待定" if stat.excess is None else f"{stat.excess:.2%}"
            typer.echo(f"{gname} T+{h} 超额={ex} n={stat.n} 待定={stat.pending}")


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


@data_app.command("sync")
def data_sync(
    ctx: typer.Context,
    universe: Optional[str] = typer.Option(None, "--universe", help="all / csi800 / csi300 / ..."),
    years: Optional[int] = typer.Option(None, "--years"),
    start: Optional[str] = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, "--end", help="YYYY-MM-DD"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    sleep: Optional[float] = typer.Option(None, "--sleep", help="请求间隔秒"),
    max_symbols: Optional[int] = typer.Option(None, "--max-symbols", help="调试上限"),
    include_global: Optional[bool] = typer.Option(None, "--include-global/--skip-global"),
    workers: int = typer.Option(4, "--workers", help="日线并行数（baostock 子进程）"),
    incremental: bool = typer.Option(False, "--incremental/--full", help="只拉 last_ok 前几天"),
    hist_fin: Optional[bool] = typer.Option(None, "--hist-fin/--skip-hist-fin", help="补历史季报/分红/预告"),
) -> None:
    """增量同步 A 股列表/成分/日线/财报到 ZHIKU_KB/data。可断点续传。"""
    if ctx.obj["provider_kind"] != "live":
        _die("data sync 仅支持 live；demo/disabled 不连接实时行情源。")
    from zhiku.data.store import MarketStore
    from zhiku.data.sync import sync_market
    from zhiku.pipeline.settings import load_pipeline_settings

    paths: Paths = ctx.obj["paths"]
    paths.ensure_runtime_dirs()
    settings = load_pipeline_settings(paths.kb / "config" / "pipeline.yaml")
    ds = settings.data
    store = MarketStore(paths.data_dir)
    report = sync_market(
        store,
        universe=universe or ds.universe,
        start=_parse_date(start) if start else None,
        end=_parse_date(end) if end else None,
        years=years or ds.years,
        resume=resume,
        include_global=ds.include_global if include_global is None else include_global,
        hk_lists=ds.hk_lists,
        us_lists=ds.us_lists,
        extra_symbols=ds.extra_symbols,
        max_symbols=max_symbols,
        sleep=sleep if sleep is not None else ds.sleep_seconds,
        retries=ds.retries,
        workers=workers,
        incremental=incremental,
        lookback_days=ds.incremental_lookback_days,
        detail_financials=ds.detail_financials,
        hist_financials=hist_fin if hist_fin is not None else ds.hist_financials,
        progress=lambda m: typer.echo(m, err=True),
    )
    typer.echo(report.summary())


app.add_typer(factor_app, name="factor")
app.add_typer(data_app, name="data")
app.add_typer(research_app, name="research")


def run() -> None:
    app()


if __name__ == "__main__":
    app()
