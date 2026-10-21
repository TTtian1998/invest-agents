"""zk data sync：增量、可断点、限速重试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from zhiku.calendar import now_shanghai
from zhiku.data.codes import is_cn_etf, normalize_cn
from zhiku.data.fetchers import (
    INDEX_CSI,
    AkshareFetcher,
    MarketFetcher,
    tushare_bars,
)
from zhiku.data.retry import timed_call
from zhiku.data.store import MarketStore
from zhiku.models import MissingData

Progress = Callable[[str], None]


@dataclass
class SyncReport:
    universe: str
    start: date
    end: date
    listings: int = 0
    bar_symbols: int = 0
    statement_symbols: int = 0
    missing: int = 0
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    used_fallback: str | None = None
    spot_rows: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"universe={self.universe} {self.start}～{self.end}",
            f"listings={self.listings} bar_symbols={self.bar_symbols} "
            f"statement_symbols={self.statement_symbols} missing={self.missing}",
            f"耗时 {self.elapsed_s:.1f}s",
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


def _log(progress: Progress | None, msg: str) -> None:
    if progress:
        progress(msg)


def _plus_one(d: date | None) -> date | None:
    if d is None:
        return None
    return d + timedelta(days=1)


def sync_market(
    store: MarketStore,
    *,
    universe: str = "csi800",
    start: date | None = None,
    end: date | None = None,
    years: int = 3,
    resume: bool = True,
    fetcher: MarketFetcher | None = None,
    include_global: bool = True,
    hk_lists: list[str] | None = None,
    us_lists: list[str] | None = None,
    extra_symbols: list[str] | None = None,
    max_symbols: int | None = None,
    sleep: float | None = None,
    retries: int = 3,
    workers: int = 1,
    incremental: bool = False,
    lookback_days: int = 2,
    detail_financials: bool = True,
    progress: Progress | None = None,
) -> SyncReport:
    import time

    t0 = time.monotonic()
    end = end or now_shanghai().date()
    start = start or date(end.year - years, end.month, end.day)
    fetcher = fetcher or AkshareFetcher(sleep=sleep, retries=retries)
    report = SyncReport(universe=universe, start=start, end=end)
    workers = max(1, int(workers))
    skip_bulk = False
    if incremental:
        last_bar = store.max_last_ok("bars:qfq")
        if last_bar is not None:
            start = max(start, last_bar - timedelta(days=max(0, lookback_days)))
            report.notes.append(f"增量同步：日线从 {start} 起（last_ok- {lookback_days}d）")
        last_bulk = store.max_last_ok("bulk:income:" + _last_quarter(end).isoformat())
        if last_bulk is not None:
            skip_bulk = True
            report.notes.append("增量同步：本季整表已在库，跳过整表财报")

    listings_ok = store.last_ok("listings", "*")
    if resume and listings_ok is not None and listings_ok >= end:
        _log(progress, "列表/成分已是目标日，跳过刷新")
    else:
        _sync_cn_meta(store, fetcher, end, report, progress)
    symbols = _resolve_cn_symbols(store, universe, extra_symbols)
    if max_symbols is not None:
        symbols = symbols[: max(0, max_symbols)]
        report.notes.append(f"max_symbols={max_symbols}，实际 CN {len(symbols)} 只")
    _log(progress, f"CN 同步标的 {len(symbols)} 只，workers={workers}")
    if incremental and report.spot_rows:
        n_spot = _upsert_spot_as_bars(store, report.spot_rows, end, symbols=symbols, only_existing=True)
        if n_spot:
            report.notes.append(f"截面写入当日日线 {n_spot} 只（增量不必逐只 hist）")

    # 先前复权（回测/qlib），再财报（选股 PIT），最后不复权
    _sync_cn_bars(store, fetcher, symbols, start, end, resume, report, "qfq", workers, progress)
    if skip_bulk:
        bulk_ok = True
    else:
        bulk_ok = _sync_bulk_financials(store, fetcher, start, end, resume, report, progress)
        if not bulk_ok:
            report.notes.append("整表财报不可用，回退到按代码拉三张报表（较慢）")
            for i, symbol in enumerate(symbols, 1):
                if i == 1 or i % 50 == 0 or i == len(symbols):
                    _log(progress, f"财报 {i}/{len(symbols)} {symbol}")
                _sync_cn_symbol_fin(store, fetcher, symbol, resume, report)
    if detail_financials:
        _sync_detail_financials(store, fetcher, symbols, resume, report, workers, progress)
    _sync_cn_bars(store, fetcher, symbols, start, end, resume, report, "none", workers, progress)
    cov = store.coverage(symbols, adjust="qfq")
    report.notes.append(
        f"日线覆盖 {cov['have']}/{cov['wanted']}（{cov['ratio']:.1%}），"
        f"未覆盖 {cov['missing']} 只"
    )

    if include_global:
        g_syms = _sync_global_lists(store, fetcher, end, hk_lists or ["hsi", "hstech"], us_lists or ["sp500", "ndx"], report, progress)
        if max_symbols is not None:
            g_syms = g_syms[: max(0, max_symbols)]
        for i, symbol in enumerate(g_syms, 1):
            if i == 1 or i % 20 == 0 or i == len(g_syms):
                _log(progress, f"港美 {i}/{len(g_syms)} {symbol}")
            _sync_global_symbol(store, fetcher, symbol, start, end, resume, report)

    try:
        n = len(store.export_parquet("qfq"))
        report.notes.append(f"已导出 {n} 个日线文件到 data/bars/qfq/")
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"parquet 导出跳过: {exc}")

    counts = store.counts()
    report.listings = counts["listings"]
    report.bar_symbols = counts["bar_symbols"]
    report.statement_symbols = counts["statement_symbols"]
    report.missing = counts["missing"]
    report.elapsed_s = time.monotonic() - t0
    return report


def _sync_cn_meta(
    store: MarketStore,
    fetcher: MarketFetcher,
    end: date,
    report: SyncReport,
    progress: Progress | None,
) -> None:
    _log(progress, "同步 A 股列表 / 退市 / 停牌 / 指数成分 / 截面")
    listings = timed_call(fetcher.cn_listings, 45, field="listings")
    if isinstance(listings, MissingData):
        store.log_missing(None, "listings", end, listings.reason)
        report.notes.append(f"缺数据：A 股列表 {listings.reason}")
        report.missing += 1
    else:
        store.upsert_listings(listings)
        store.set_state("listings", "*", last_ok=end, status="ok")

    delisted = timed_call(fetcher.cn_delisted, 45, field="delist")
    if isinstance(delisted, MissingData):
        store.log_missing(None, "delist", end, delisted.reason)
        report.notes.append(f"缺数据：退市列表 {delisted.reason}")
        report.missing += 1
    else:
        for row in delisted:
            if row.get("symbol") and row.get("delist_date"):
                store.mark_delisted(row["symbol"], row["delist_date"], row.get("name"))

    suspended = timed_call(fetcher.cn_suspended, 30, end, field="suspended")
    if isinstance(suspended, MissingData):
        store.log_missing(None, "suspended", end, suspended.reason)
        report.missing += 1
    else:
        store.upsert_listings(
            [{"symbol": s, "market": "CN", "suspended": True, "instrument": "stock"} for s in suspended]
        )

    for alias, code in INDEX_CSI.items():
        members = timed_call(fetcher.index_members, 45, code, field=f"index:{alias}")
        if isinstance(members, MissingData):
            store.log_missing(None, f"index:{alias}", end, members.reason)
            report.notes.append(f"缺数据：指数 {alias} {members.reason}")
            report.missing += 1
            continue
        store.upsert_index_members(code, end, members)
        store.upsert_listings(
            [{"symbol": s, "market": "CN", "instrument": "etf" if is_cn_etf(s) else "stock"} for s in members]
        )

    spot = timed_call(fetcher.cn_spot, 45, field="spot")
    if isinstance(spot, MissingData):
        store.log_missing(None, "spot", end, spot.reason)
        report.notes.append(f"缺数据：行情截面 {spot.reason}")
        report.missing += 1
    else:
        store.upsert_spot(end, spot)
        store.upsert_listings(
            [
                {
                    "symbol": r["symbol"],
                    "market": "CN",
                    "name": r.get("name"),
                    "st": r.get("st"),
                    "instrument": "etf" if is_cn_etf(r["symbol"]) else "stock",
                }
                for r in spot
            ]
        )
        report.spot_rows = spot


def _upsert_spot_as_bars(
    store: MarketStore,
    spot: list[dict],
    on: date,
    *,
    symbols: list[str] | None = None,
    only_existing: bool = True,
) -> int:
    """把当日东财截面写成一根日线。只更新已有历史的标的，避免把未拉满的新票标成完成。"""
    wanted = set(symbols) if symbols else None
    rows = []
    seen: list[str] = []
    for r in spot:
        symbol = r["symbol"]
        if wanted is not None and symbol not in wanted:
            continue
        close = r.get("close")
        if close is None:
            continue
        if only_existing:
            if store.last_ok("bars:qfq", symbol) is None and store.last_ok("bars:none", symbol) is None:
                continue
        for adjust in ("qfq", "none"):
            rows.append(
                {
                    "symbol": symbol,
                    "market": "CN",
                    "date": on,
                    "adjust": adjust,
                    "open": r.get("open") or close,
                    "high": r.get("high") or close,
                    "low": r.get("low") or close,
                    "close": close,
                    "volume": r.get("volume"),
                    "amount": r.get("amount"),
                    "turnover": None,
                    "pct_chg": r.get("pct_chg"),
                    "limit_up": None,
                    "limit_down": None,
                }
            )
        seen.append(symbol)
    if not rows:
        return 0
    import pandas as pd

    store.upsert_bars(pd.DataFrame(rows))
    for symbol in seen:
        store.set_state("bars:qfq", symbol, last_ok=on, status="ok")
        store.set_state("bars:none", symbol, last_ok=on, status="ok")
    return len(seen)


def _resolve_cn_symbols(store: MarketStore, universe: str, extra: list[str] | None) -> list[str]:
    from zhiku.data.universe_live import resolve_universe_symbols

    symbols = resolve_universe_symbols(store, universe)
    for s in extra or []:
        ns = normalize_cn(s) if not s.endswith((".HK", ".US")) and not s.isalpha() else s
        if ns not in symbols:
            symbols.append(ns)
    return symbols


def _quarter_ends(start: date, end: date) -> list[date]:
    out: list[date] = []
    y, q = start.year, (start.month - 1) // 3 + 1
    month = q * 3
    day = 31 if month in {3, 12} else 30
    cur = date(y, month, day)
    if cur < start:
        cur = _next_quarter(cur)
    while cur <= end:
        out.append(cur)
        cur = _next_quarter(cur)
    return out


def _next_quarter(d: date) -> date:
    if d.month == 3:
        return date(d.year, 6, 30)
    if d.month == 6:
        return date(d.year, 9, 30)
    if d.month == 9:
        return date(d.year, 12, 31)
    return date(d.year + 1, 3, 31)


def _last_quarter(end: date) -> date:
    ends = _quarter_ends(date(end.year - 1, 1, 1), end)
    return ends[-1] if ends else date(end.year, 3, 31)


def _needs_detail(store: MarketStore, symbol: str) -> bool:
    from zhiku.data.compute import pick_num

    stmts = store.statements_asof(symbol, now_shanghai().date())
    income = (stmts.get("income") or {}).get("payload") or {}
    balance = (stmts.get("balance") or {}).get("payload") or {}
    has_ebit = pick_num(income, "ebit") is not None
    has_ca = pick_num(balance, "current_assets") is not None
    has_cl = pick_num(balance, "current_liab") is not None
    has_fa = pick_num(balance, "fixed_assets") is not None
    return not (has_ebit and has_ca and has_cl and has_fa)


def _sync_detail_financials(
    store: MarketStore,
    fetcher: MarketFetcher,
    symbols: list[str],
    resume: bool,
    report: SyncReport,
    workers: int,
    progress: Progress | None,
) -> None:
    from zhiku.data.detail_fin import merge_payload

    job = "detail_fin"
    pending: list[str] = []
    for symbol in symbols:
        if resume and store.job_status(job, symbol) == "missing":
            continue
        if resume and store.last_ok(job, symbol) is not None and not _needs_detail(store, symbol):
            continue
        if not _needs_detail(store, symbol):
            store.set_state(job, symbol, last_ok=now_shanghai().date(), status="ok")
            continue
        pending.append(symbol)
    _log(progress, f"明细财务待拉 {len(pending)}/{len(symbols)}")
    if not pending:
        return

    def _fetch(symbol: str):
        raw = timed_call(fetcher.cn_statement_detail, 50, symbol, symbol=symbol, field=job)
        return symbol, raw

    def _commit(symbol: str, raw) -> None:
        if isinstance(raw, MissingData):
            store.log_missing(symbol, job, None, raw.reason)
            store.set_state(job, symbol, last_ok=None, status="missing", error=raw.reason)
            report.missing += 1
            return
        for row in raw:
            period = row.get("period")
            kind = row.get("kind")
            existing = store.statement_period(symbol, kind, period) if period and kind else None
            if existing:
                row["payload"] = merge_payload(existing.get("payload"), row.get("payload"))
                if row.get("announced_is_period") and existing.get("announced"):
                    try:
                        row["announced"] = date.fromisoformat(str(existing["announced"])[:10])
                        row["announced_is_period"] = False
                    except ValueError:
                        pass
            if row.get("announced_is_period"):
                store.log_missing(symbol, f"{kind}.announced", period, "明细财务无公告日，用报告期")
                report.missing += 1
            store.upsert_statements([row])
        store.set_state(job, symbol, last_ok=now_shanghai().date(), status="ok")
        report.statement_symbols += 1

    done = 0
    pool_n = min(max(1, workers), len(pending))
    if pool_n <= 1:
        for symbol in pending:
            _, raw = _fetch(symbol)
            _commit(symbol, raw)
            done += 1
            if done == 1 or done % 20 == 0 or done == len(pending):
                _log(progress, f"明细财务 {done}/{len(pending)} {symbol}")
        return
    with ThreadPoolExecutor(max_workers=pool_n) as pool:
        futs = {pool.submit(_fetch, s): s for s in pending}
        for fut in as_completed(futs):
            symbol = futs[fut]
            try:
                symbol, raw = fut.result()
            except Exception as exc:  # noqa: BLE001
                raw = MissingData(str(exc), symbol=symbol)
            _commit(symbol, raw)
            done += 1
            if done == 1 or done % 20 == 0 or done == len(pending):
                _log(progress, f"明细财务 {done}/{len(pending)} {symbol}")


def _sync_bulk_financials(
    store: MarketStore,
    fetcher: MarketFetcher,
    start: date,
    end: date,
    resume: bool,
    report: SyncReport,
    progress: Progress | None,
) -> bool:
    periods = _quarter_ends(start, end)
    if not periods:
        return False
    got = 0
    for period in periods:
        for kind in ("income", "balance", "cashflow", "yjbb"):
            job = f"bulk:{kind}:{period.isoformat()}"
            if resume and store.last_ok(job, "*") is not None:
                got += 1
                continue
            _log(progress, f"整表财报 {kind} {period}")
            raw = timed_call(
                fetcher.bulk_financials,
                90,
                period,
                kind,
                field=f"bulk:{kind}",
            )
            if isinstance(raw, MissingData):
                if "未提供整表财务" in (raw.reason or ""):
                    return False
                store.log_missing(None, job, period, raw.reason)
                store.set_state(job, "*", last_ok=None, status="missing", error=raw.reason)
                report.missing += 1
                continue
            store.upsert_statements(raw)
            names = []
            for row in raw:
                nm = (row.get("payload") or {}).get("name")
                if nm:
                    names.append(
                        {
                            "symbol": row["symbol"],
                            "market": "CN",
                            "name": nm,
                            "instrument": "etf" if is_cn_etf(row["symbol"]) else "stock",
                        }
                    )
            if names:
                store.upsert_listings(names)
            store.set_state(job, "*", last_ok=end, status="ok")
            got += 1
        div_job = f"bulk:div:{period.isoformat()}"
        if resume and store.last_ok(div_job, "*") is not None:
            got += 1
            continue
        _log(progress, f"整表分红 {period}")
        raw_div = timed_call(fetcher.bulk_dividends, 90, period, field="bulk:div")
        if isinstance(raw_div, MissingData):
            if "未提供整表分红" not in (raw_div.reason or ""):
                store.log_missing(None, div_job, period, raw_div.reason)
                store.set_state(div_job, "*", last_ok=None, status="missing", error=raw_div.reason)
                report.missing += 1
        else:
            store.upsert_dividends(raw_div)
            store.set_state(div_job, "*", last_ok=end, status="ok")
            got += 1
    report.notes.append(f"整表财报完成 {got}/{len(periods)*5} 次")
    return got > 0


def _sync_cn_bars(
    store: MarketStore,
    fetcher: MarketFetcher,
    symbols: list[str],
    start: date,
    end: date,
    resume: bool,
    report: SyncReport,
    adjust: str,
    workers: int,
    progress: Progress | None,
) -> None:
    job = f"bars:{adjust}"
    pending: list[tuple[str, date]] = []
    for symbol in symbols:
        if resume and store.job_status(job, symbol) == "missing":
            report.missing += 1
            continue
        last = store.last_ok(job, symbol) if resume else None
        if last and last >= end:
            continue
        fetch_start = start
        if last:
            fetch_start = max(start, _plus_one(last) or start)
        pending.append((symbol, fetch_start))
    _log(progress, f"{adjust} 日线待拉 {len(pending)}/{len(symbols)}")
    if not pending:
        return

    def _fetch(item: tuple[str, date]):
        symbol, fetch_start = item
        raw = timed_call(
            fetcher.cn_bars,
            40,
            symbol,
            fetch_start,
            end,
            adjust,
            symbol=symbol,
            field=job,
        )
        if isinstance(raw, MissingData) and isinstance(fetcher, AkshareFetcher):
            ts = tushare_bars(symbol, fetch_start, end, adjust)
            if not isinstance(ts, MissingData):
                report.used_fallback = "tushare"
                raw = ts
        return symbol, raw

    done = 0
    pool_n = min(workers, len(pending))
    if pool_n <= 1:
        iterator = (_fetch(item) for item in pending)
        for symbol, raw in iterator:
            done += 1
            _commit_bars(store, job, symbol, end, raw, report)
            if done == 1 or done % 5 == 0 or done == len(pending):
                _log(progress, f"{adjust} 日线 {done}/{len(pending)} {symbol}")
        return

    with ThreadPoolExecutor(max_workers=pool_n) as pool:
        futs = {pool.submit(_fetch, item): item[0] for item in pending}
        for fut in as_completed(futs):
            symbol = futs[fut]
            try:
                symbol, raw = fut.result()
            except Exception as exc:  # noqa: BLE001
                raw = MissingData(str(exc), symbol=symbol)
            done += 1
            _commit_bars(store, job, symbol, end, raw, report)
            if done == 1 or done % 5 == 0 or done == len(pending):
                _log(progress, f"{adjust} 日线 {done}/{len(pending)} {symbol}")


def _commit_bars(
    store: MarketStore,
    job: str,
    symbol: str,
    end: date,
    raw,
    report: SyncReport,
) -> None:
    if isinstance(raw, MissingData):
        store.log_missing(symbol, job, end, raw.reason)
        store.set_state(job, symbol, last_ok=None, status="missing", error=raw.reason)
        report.missing += 1
        return
    store.upsert_bars(raw)
    store.set_state(job, symbol, last_ok=end, status="ok")


def _sync_cn_symbol_bars(
    store: MarketStore,
    fetcher: MarketFetcher,
    symbol: str,
    start: date,
    end: date,
    resume: bool,
    report: SyncReport,
) -> None:
    """单标的入口（测试/兼容）。"""
    _sync_cn_bars(store, fetcher, [symbol], start, end, resume, report, "qfq", 1, None)
    _sync_cn_bars(store, fetcher, [symbol], start, end, resume, report, "none", 1, None)


def _sync_cn_symbol_fin(
    store: MarketStore,
    fetcher: MarketFetcher,
    symbol: str,
    resume: bool,
    report: SyncReport,
) -> None:
    job = "financials"
    if resume and store.last_ok(job, symbol) is not None:
        # 财报按整表覆盖；已成功则跳过，除非调用方关闭 resume
        report.statement_symbols += 1
        return
    ok = True
    for kind in ("income", "balance", "cashflow"):
        raw = fetcher.cn_statement(symbol, kind)
        if isinstance(raw, MissingData):
            store.log_missing(symbol, kind, None, raw.reason)
            report.missing += 1
            ok = False
            continue
        store.upsert_statements(raw)
    divs = fetcher.cn_dividends(symbol)
    if isinstance(divs, MissingData):
        store.log_missing(symbol, "dividend", None, divs.reason)
        report.missing += 1
    else:
        store.upsert_dividends(divs)
    if ok:
        store.set_state(job, symbol, last_ok=now_shanghai().date(), status="ok")
        report.statement_symbols += 1
    else:
        store.set_state(job, symbol, last_ok=None, status="partial")


def _sync_global_lists(
    store: MarketStore,
    fetcher: MarketFetcher,
    end: date,
    hk_lists: list[str],
    us_lists: list[str],
    report: SyncReport,
    progress: Progress | None,
) -> list[str]:
    symbols: list[str] = []
    for name in [*hk_lists, *us_lists]:
        members = fetcher.global_members(name)
        if isinstance(members, MissingData):
            store.log_missing(None, f"index:{name}", end, members.reason)
            report.notes.append(f"缺数据：{name} 成分 {members.reason}")
            report.missing += 1
            continue
        store.upsert_index_members(name, end, members)
        market = "HK" if name.startswith("hs") else "US"
        store.upsert_listings(
            [{"symbol": s, "market": market, "instrument": "stock"} for s in members]
        )
        symbols.extend(members)
        _log(progress, f"{name} 成分 {len(members)}")
    # 去重保持顺序
    seen: set[str] = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _sync_global_symbol(
    store: MarketStore,
    fetcher: MarketFetcher,
    symbol: str,
    start: date,
    end: date,
    resume: bool,
    report: SyncReport,
) -> None:
    job = "bars:global"
    last = store.last_ok(job, symbol) if resume else None
    fetch_start = start
    if last and last >= end:
        pass
    else:
        if last:
            fetch_start = max(start, _plus_one(last) or start)
        raw = fetcher.global_bars(symbol, fetch_start, end)
        if isinstance(raw, MissingData):
            store.log_missing(symbol, job, end, raw.reason)
            store.set_state(job, symbol, last_ok=last, status="missing", error=raw.reason)
            report.missing += 1
        else:
            store.upsert_bars(raw)
            store.set_state(job, symbol, last_ok=end, status="ok")
    fin_job = "financials:global"
    if resume and store.last_ok(fin_job, symbol) is not None:
        return
    raw_f = fetcher.global_financials(symbol)
    if isinstance(raw_f, MissingData):
        store.log_missing(symbol, fin_job, end, raw_f.reason)
        store.set_state(fin_job, symbol, last_ok=None, status="missing", error=raw_f.reason)
        report.missing += 1
        return
    for kind, rows in raw_f.items():
        announced_is_period = any(r.get("announced_is_period") for r in rows)
        store.upsert_statements(rows)
        if announced_is_period:
            store.log_missing(symbol, f"{kind}.announced", None, "yfinance 无公告日，用报告期并标注缺数据")
            report.missing += 1
    store.set_state(fin_job, symbol, last_ok=end, status="ok")
    report.statement_symbols += 1
