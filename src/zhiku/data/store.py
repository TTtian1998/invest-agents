"""本地行情库：SQLite 为主，可选把日线导出 parquet。不进 git。"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from zhiku.models import Market


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _iso(d: date | datetime | str | None) -> str | None:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


class MarketStore:
    """ZHIKU_KB/data 下的增量库。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.root / "market.sqlite"
        self.bars_dir = self.root / "bars"
        self._init_schema()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(dividends)").fetchall()}
        if "yield" not in cols:
            conn.execute("ALTER TABLE dividends ADD COLUMN yield REAL")

    def _init_schema(self) -> None:
        with _connect(self.sqlite_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS listings (
                    symbol TEXT PRIMARY KEY,
                    market TEXT NOT NULL,
                    name TEXT,
                    industry TEXT,
                    list_date TEXT,
                    delist_date TEXT,
                    st INTEGER NOT NULL DEFAULT 0,
                    suspended INTEGER NOT NULL DEFAULT 0,
                    instrument TEXT NOT NULL DEFAULT 'stock'
                );
                CREATE TABLE IF NOT EXISTS index_members (
                    index_code TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    asof TEXT NOT NULL,
                    PRIMARY KEY (index_code, symbol, asof)
                );
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    date TEXT NOT NULL,
                    adjust TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL, turnover REAL, pct_chg REAL,
                    limit_up INTEGER, limit_down INTEGER,
                    PRIMARY KEY (symbol, date, adjust)
                );
                CREATE INDEX IF NOT EXISTS idx_bars_date ON bars(date);
                CREATE TABLE IF NOT EXISTS statements (
                    symbol TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    period TEXT NOT NULL,
                    announced TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (symbol, kind, period, announced)
                );
                CREATE TABLE IF NOT EXISTS dividends (
                    symbol TEXT NOT NULL,
                    ex_date TEXT NOT NULL,
                    announced TEXT,
                    amount REAL,
                    yield REAL,
                    PRIMARY KEY (symbol, ex_date)
                );
                CREATE TABLE IF NOT EXISTS spot (
                    symbol TEXT NOT NULL,
                    asof TEXT NOT NULL,
                    pe REAL, pb REAL, mktcap REAL, amount REAL, name TEXT,
                    PRIMARY KEY (symbol, asof)
                );
                CREATE TABLE IF NOT EXISTS sync_state (
                    job TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    last_ok TEXT,
                    status TEXT,
                    error TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (job, symbol)
                );
                CREATE TABLE IF NOT EXISTS missing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    field TEXT,
                    on_date TEXT,
                    reason TEXT
                );
                """
            )
            self._migrate(conn)

    def is_empty(self) -> bool:
        with _connect(self.sqlite_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
            b = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        return int(n) + int(b) == 0

    def log_missing(self, symbol: str | None, field: str | None, on: date | None, reason: str) -> None:
        with _connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO missing_log(symbol, field, on_date, reason) VALUES (?,?,?,?)",
                (symbol, field, _iso(on), reason),
            )

    def set_state(self, job: str, symbol: str, *, last_ok: date | None, status: str, error: str | None = None) -> None:
        with _connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sync_state(job, symbol, last_ok, status, error, updated_at)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    job,
                    symbol,
                    _iso(last_ok),
                    status,
                    error,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def last_ok(self, job: str, symbol: str) -> date | None:
        with _connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT last_ok FROM sync_state WHERE job=? AND symbol=?",
                (job, symbol),
            ).fetchone()
        if not row or not row[0]:
            return None
        return date.fromisoformat(str(row[0])[:10])

    def job_status(self, job: str, symbol: str) -> str | None:
        with _connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT status FROM sync_state WHERE job=? AND symbol=?",
                (job, symbol),
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def upsert_listings(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = []
        for row in rows:
            payload.append(
                (
                    str(row["symbol"]),
                    str(row.get("market") or Market.CN.value),
                    row.get("name"),
                    row.get("industry"),
                    _iso(row.get("list_date")),
                    _iso(row.get("delist_date")),
                    1 if row.get("st") else 0,
                    1 if row.get("suspended") else 0,
                    str(row.get("instrument") or "stock"),
                )
            )
        if not payload:
            return 0
        with _connect(self.sqlite_path) as conn:
            conn.executemany(
                """
                INSERT INTO listings(symbol, market, name, industry, list_date, delist_date, st, suspended, instrument)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol) DO UPDATE SET
                    market=excluded.market,
                    name=COALESCE(excluded.name, listings.name),
                    industry=COALESCE(excluded.industry, listings.industry),
                    list_date=COALESCE(excluded.list_date, listings.list_date),
                    delist_date=COALESCE(excluded.delist_date, listings.delist_date),
                    st=excluded.st,
                    suspended=excluded.suspended,
                    instrument=excluded.instrument
                """,
                payload,
            )
        return len(payload)

    def mark_delisted(self, symbol: str, on: date, name: str | None = None) -> None:
        self.upsert_listings(
            [{"symbol": symbol, "market": "CN", "name": name, "delist_date": on, "instrument": "stock"}]
        )

    def upsert_index_members(self, index_code: str, asof: date, symbols: list[str]) -> int:
        if not symbols:
            return 0
        key = _iso(asof)
        rows = [(index_code, s, key) for s in symbols]
        with _connect(self.sqlite_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO index_members(index_code, symbol, asof) VALUES (?,?,?)",
                rows,
            )
        return len(rows)

    def upsert_bars(self, frames: pd.DataFrame) -> int:
        if frames is None or frames.empty:
            return 0
        df = frames.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        cols = [
            "symbol",
            "market",
            "date",
            "adjust",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "pct_chg",
            "limit_up",
            "limit_down",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        records = df[cols].where(pd.notna(df[cols]), None).values.tolist()
        with _connect(self.sqlite_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO bars(
                    symbol, market, date, adjust, open, high, low, close,
                    volume, amount, turnover, pct_chg, limit_up, limit_down
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                records,
            )
        return len(records)

    def upsert_statements(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = []
        for row in rows:
            payload.append(
                (
                    str(row["symbol"]),
                    str(row["kind"]),
                    _iso(row["period"]),
                    _iso(row["announced"]),
                    json.dumps(row.get("payload") or {}, ensure_ascii=False, default=str),
                )
            )
        if not payload:
            return 0
        with _connect(self.sqlite_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO statements(symbol, kind, period, announced, payload)
                VALUES (?,?,?,?,?)
                """,
                payload,
            )
        return len(payload)

    def upsert_dividends(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = []
        for row in rows:
            payload.append(
                (
                    str(row["symbol"]),
                    _iso(row["ex_date"]),
                    _iso(row.get("announced")),
                    row.get("amount"),
                    row.get("yield"),
                )
            )
        if not payload:
            return 0
        with _connect(self.sqlite_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO dividends(symbol, ex_date, announced, amount, yield) VALUES (?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def statement_period(self, symbol: str, kind: str, period: date) -> dict[str, Any] | None:
        with _connect(self.sqlite_path) as conn:
            row = conn.execute(
                """
                SELECT period, announced, payload FROM statements
                WHERE symbol=? AND kind=? AND period=?
                ORDER BY announced DESC LIMIT 1
                """,
                (symbol, kind, _iso(period)),
            ).fetchone()
        if not row:
            return None
        return {"period": row[0], "announced": row[1], "payload": json.loads(row[2])}

    def max_last_ok(self, job: str) -> date | None:
        with _connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT MAX(last_ok) FROM sync_state WHERE job=? AND status='ok'",
                (job,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return date.fromisoformat(str(row[0])[:10])

    def bar_symbols(self, adjust: str = "qfq") -> set[str]:
        with _connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM bars WHERE adjust=?",
                (adjust,),
            ).fetchall()
        return {str(r[0]) for r in rows}

    def coverage(self, symbols: list[str], *, adjust: str = "qfq") -> dict[str, Any]:
        have = self.bar_symbols(adjust)
        wanted = [s for s in symbols]
        ok = [s for s in wanted if s in have]
        return {
            "wanted": len(wanted),
            "have": len(ok),
            "missing": len(wanted) - len(ok),
            "ratio": (len(ok) / len(wanted)) if wanted else 0.0,
        }

    def upsert_spot(self, asof: date, rows: Iterable[dict[str, Any]]) -> int:
        key = _iso(asof)
        payload = [
            (
                str(r["symbol"]),
                key,
                r.get("pe"),
                r.get("pb"),
                r.get("mktcap"),
                r.get("amount"),
                r.get("name"),
            )
            for r in rows
        ]
        if not payload:
            return 0
        with _connect(self.sqlite_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO spot(symbol, asof, pe, pb, mktcap, amount, name) VALUES (?,?,?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def listings(self, market: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM listings"
        params: tuple = ()
        if market:
            q += " WHERE market=?"
            params = (market,)
        with _connect(self.sqlite_path) as conn:
            return pd.read_sql_query(q, conn, params=params)

    def index_symbols(self, index_code: str, asof: date | None = None) -> list[str]:
        with _connect(self.sqlite_path) as conn:
            if asof is None:
                row = conn.execute(
                    "SELECT MAX(asof) FROM index_members WHERE index_code=?",
                    (index_code,),
                ).fetchone()
                if not row or not row[0]:
                    return []
                asof_s = row[0]
            else:
                asof_s = _iso(asof)
            rows = conn.execute(
                "SELECT symbol FROM index_members WHERE index_code=? AND asof=?",
                (index_code, asof_s),
            ).fetchall()
        return [r[0] for r in rows]

    def delist_map(self) -> dict[str, date]:
        with _connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT symbol, delist_date FROM listings WHERE delist_date IS NOT NULL AND delist_date != ''"
            ).fetchall()
        out: dict[str, date] = {}
        for symbol, d in rows:
            try:
                out[str(symbol)] = date.fromisoformat(str(d)[:10])
            except ValueError:
                continue
        return out

    def bars(
        self,
        symbols: list[str] | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        q = "SELECT * FROM bars WHERE adjust=?"
        params: list[Any] = [adjust]
        if symbols:
            marks = ",".join("?" * len(symbols))
            q += f" AND symbol IN ({marks})"
            params.extend(symbols)
        if start is not None:
            q += " AND date>=?"
            params.append(_iso(start))
        if end is not None:
            q += " AND date<=?"
            params.append(_iso(end))
        q += " ORDER BY date"
        with _connect(self.sqlite_path) as conn:
            df = pd.read_sql_query(q, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def price_panel(
        self,
        symbols: list[str],
        *,
        start: date | None = None,
        end: date | None = None,
        adjust: str = "qfq",
        field: str = "close",
    ) -> pd.DataFrame:
        df = self.bars(symbols, start=start, end=end, adjust=adjust)
        if df.empty or field not in df.columns:
            return pd.DataFrame()
        panel = df.pivot(index="date", columns="symbol", values=field).sort_index()
        return panel

    def limit_panels(
        self,
        symbols: list[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.bars(symbols, start=start, end=end, adjust="none")
        if df.empty:
            # 不复权未同步完时用前复权涨跌幅近似涨跌停，不编造缺失日
            df = self.bars(symbols, start=start, end=end, adjust="qfq")
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        up = df.pivot(index="date", columns="symbol", values="limit_up").fillna(0).astype(bool)
        down = df.pivot(index="date", columns="symbol", values="limit_down").fillna(0).astype(bool)
        return up.sort_index(), down.sort_index()

    def statements_asof(self, symbol: str, asof: date) -> dict[str, dict[str, Any]]:
        """每个 kind 取 announced<=asof 的最新一条。period 日不得替代公告日。"""
        with _connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT kind, period, announced, payload FROM statements
                WHERE symbol=? AND announced<=?
                ORDER BY announced DESC
                """,
                (symbol, _iso(asof)),
            ).fetchall()
        chosen: dict[str, dict[str, Any]] = {}
        for kind, period, announced, payload in rows:
            if kind in chosen:
                continue
            chosen[kind] = {
                "kind": kind,
                "period": period,
                "announced": announced,
                "payload": json.loads(payload),
            }
        return chosen

    def statements_history(self, symbol: str, kind: str, asof: date) -> list[dict[str, Any]]:
        with _connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT period, announced, payload FROM statements
                WHERE symbol=? AND kind=? AND announced<=?
                ORDER BY announced DESC
                """,
                (symbol, kind, _iso(asof)),
            ).fetchall()
        out = []
        for period, announced, payload in rows:
            out.append(
                {
                    "period": period,
                    "announced": announced,
                    "payload": json.loads(payload),
                }
            )
        return out

    def latest_spot(self, asof: date) -> pd.DataFrame:
        with _connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT MAX(asof) FROM spot WHERE asof<=?",
                (_iso(asof),),
            ).fetchone()
            if not row or not row[0]:
                return pd.DataFrame()
            return pd.read_sql_query(
                "SELECT * FROM spot WHERE asof=?",
                conn,
                params=(row[0],),
            )

    def dividends_asof(self, symbol: str, asof: date) -> pd.DataFrame:
        with _connect(self.sqlite_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM dividends WHERE symbol=? AND (announced IS NULL OR announced<=?) AND ex_date<=?",
                conn,
                params=(symbol, _iso(asof), _iso(asof)),
            )
        return df

    def counts(self) -> dict[str, int]:
        with _connect(self.sqlite_path) as conn:
            listings = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
            bars = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            stmts = conn.execute("SELECT COUNT(*) FROM statements").fetchone()[0]
            missing = conn.execute("SELECT COUNT(*) FROM missing_log").fetchone()[0]
            symbols_bars = conn.execute("SELECT COUNT(DISTINCT symbol) FROM bars").fetchone()[0]
            stmt_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM statements").fetchone()[0]
        return {
            "listings": int(listings),
            "bars": int(bars),
            "bar_symbols": int(symbols_bars),
            "statements": int(stmts),
            "statement_symbols": int(stmt_symbols),
            "missing": int(missing),
        }

    def last_bar(
        self, symbol: str, adjust: str, *, before: date | None = None
    ) -> dict[str, Any] | None:
        q = "SELECT * FROM bars WHERE symbol=? AND adjust=?"
        params: list[Any] = [symbol, adjust]
        if before is not None:
            q += " AND date<?"
            params.append(_iso(before))
        q += " ORDER BY date DESC LIMIT 1"
        with _connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(q, params).fetchone()
        return dict(row) if row else None

    def delete_bars(
        self,
        *,
        symbol: str | None = None,
        on: date | None = None,
        adjust: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol)
        if on is not None:
            clauses.append("date=?")
            params.append(_iso(on))
        if adjust:
            clauses.append("adjust=?")
            params.append(adjust)
        if not clauses:
            return 0
        with _connect(self.sqlite_path) as conn:
            cur = conn.execute(f"DELETE FROM bars WHERE {' AND '.join(clauses)}", params)
            return int(cur.rowcount)

    def listing_name_map(self, market: str | None = "CN") -> dict[str, str]:
        df = self.listings(market)
        out: dict[str, str] = {}
        if df.empty:
            return out
        for _, row in df.iterrows():
            name = str(row.get("name") or "").strip()
            symbol = str(row.get("symbol") or "").strip()
            if name and symbol and name not in {"nan", "None"}:
                out[name] = symbol
        return out

    def export_parquet(self, adjust: str = "qfq") -> list[Path]:
        """有 pyarrow 时写成 parquet，否则写成 csv。失败不抛。"""
        df = self.bars(adjust=adjust)
        if df.empty:
            return []
        written: list[Path] = []
        for symbol, part in df.groupby("symbol"):
            folder = self.bars_dir / adjust
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{symbol}.parquet"
            try:
                part.to_parquet(path, index=False)
            except Exception:  # noqa: BLE001
                path = folder / f"{symbol}.csv"
                part.to_csv(path, index=False)
            written.append(path)
        return written
