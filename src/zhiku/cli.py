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
    help="投资智库 zhiku：触发线、14:55/收盘确认、成长仓闸门、周五命中率。不接券商、不下单。",
)


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
    events = evaluator.events_from(evaluations, on)
    written = EventLog(paths.events).append_many(events)
    md = render(
        "check.md.j2",
        on=on.isoformat(),
        provider=ctx.obj["provider_kind"],
        kb=str(paths.kb),
        evaluations=evaluations,
        warnings=[],
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
    events = evaluator.events_from(evaluations, on)
    written = EventLog(paths.events).append_many(events)
    md = render(
        "close.md.j2",
        on=on.isoformat(),
        provider=ctx.obj["provider_kind"],
        kb=str(paths.kb),
        evaluations=evaluations,
        warnings=[],
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
    md = render(
        "weekly.md.j2",
        review=result,
        provider=ctx.obj["provider_kind"],
        kb=str(paths.kb),
    )
    out = write_report(paths.reports, result.end, "weekly", md)
    typer.echo(md)
    typer.echo("候选教训（不会自动写入 lessons 文件）：", err=True)
    for lesson in result.lessons:
        typer.echo(f"- {lesson}", err=True)
    typer.echo(f"已写入 {out}", err=True)


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


def run() -> None:
    app()


if __name__ == "__main__":
    app()
