"""预期差为核心：事前一致预期、隐含涨跌幅、事后标准化 surprise。缺预期不编造。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from zhiku.events.models import SurpriseCard
from zhiku.macro.surprise import surprise_z
from zhiku.models import MissingData
from zhiku.news.models import ClusteredNews


def vs_implied(actual_move: float | None, implied_move: float | None) -> str:
    if actual_move is None or implied_move is None:
        return "缺数据：无实际涨跌或隐含涨跌幅"
    if abs(implied_move) < 1e-12:
        return "缺数据：隐含涨跌幅为 0"
    if abs(actual_move) > abs(implied_move) * 1.1:
        return "实际波动超过隐含涨跌幅"
    if abs(actual_move) < abs(implied_move) * 0.5:
        return "实际波动低于隐含，预期差可能仍未定价"
    return "实际波动接近隐含涨跌幅"


def implied_move_from_straddle(
    spot: float | None,
    call_mid: float | None,
    put_mid: float | None,
) -> float | MissingData:
    """平值跨式近似： (call+put)/spot。缺一条腿就缺数据。"""
    if spot is None or spot <= 0:
        return MissingData("缺数据：无现价，不算隐含涨跌幅")
    if call_mid is None or put_mid is None:
        return MissingData("缺数据：期权链缺少平值 call/put")
    if call_mid < 0 or put_mid < 0:
        return MissingData("缺数据：期权中间价非法")
    return (call_mid + put_mid) / spot


def load_consensus(path: Path) -> list[SurpriseCard]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("items") if isinstance(raw, dict) else raw
    out: list[SurpriseCard] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        card = SurpriseCard.model_validate(row)
        z, missing = surprise_z(card.actual, card.consensus, card.sigma)
        card.surprise_z = z
        card.surprise_missing = missing
        if not card.vs_implied:
            card.vs_implied = vs_implied(card.actual_move, card.implied_move)
        out.append(card)
    return out


def load_implied_moves(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("moves") if isinstance(raw, dict) else raw
    out: dict[str, float] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "")
        move = row.get("implied_move")
        if sym and move is not None:
            try:
                out[sym] = float(move)
            except (TypeError, ValueError):
                continue
    return out


def _macro_match(cluster: ClusteredNews, card: SurpriseCard) -> bool:
    token = (card.event_id or card.source or "").lower()
    if not token:
        return False
    key = token.split("_")[0]
    if len(key) < 3:
        return False
    blob = " ".join(
        [cluster.headline, cluster.catalyst, *[i.headline for i in cluster.items]]
    ).lower()
    return key in blob.replace(" ", "")


def attach_surprise(
    clusters: list[ClusteredNews],
    cards: list[SurpriseCard],
    implied: dict[str, float] | None = None,
) -> None:
    implied = implied or {}
    by_sym: dict[str, SurpriseCard] = {}
    macros: list[SurpriseCard] = []
    for c in cards:
        if c.kind == "macro" or c.event_id:
            macros.append(c)
        if not c.symbol:
            continue
        prev = by_sym.get(c.symbol)
        if prev is None or (c.kind == "eps" and prev.kind != "eps"):
            by_sym[c.symbol] = c
    for cluster in clusters:
        sym = cluster.symbol or ""
        card = by_sym.get(sym)
        if card is None:
            hit = next((m for m in macros if _macro_match(cluster, m)), None)
            if hit is not None:
                cluster.surprise_z = hit.surprise_z
                cluster.surprise_missing = hit.surprise_missing or (
                    None if hit.surprise_z is not None else "缺数据：宏观尚无实际值"
                )
                cluster.surprise = hit
                continue
            cluster.surprise_missing = "缺数据：无事前一致预期，不编造预期差"
            if sym in implied:
                cluster.implied_move = implied[sym]
                cluster.vs_implied = vs_implied(cluster.excess_return, cluster.implied_move)
            continue
        cluster.surprise_z = card.surprise_z
        cluster.surprise_missing = card.surprise_missing
        cluster.implied_move = (
            card.implied_move if card.implied_move is not None else implied.get(sym)
        )
        cluster.actual_move = (
            card.actual_move if card.actual_move is not None else cluster.excess_return
        )
        cluster.vs_implied = vs_implied(cluster.actual_move, cluster.implied_move)
        cluster.surprise = card


def surprise_component(cluster: ClusteredNews) -> float:
    """打分用预期差幅度，不看消息好坏。缺数据给中性 0.5。"""
    z = cluster.surprise_z
    if z is None:
        return 0.5
    mag = min(1.0, abs(z) / 2.0)
    implied = cluster.implied_move
    actual = cluster.actual_move if cluster.actual_move is not None else cluster.excess_return
    if implied is not None and actual is not None and abs(actual) >= abs(implied):
        mag *= 0.55
    return mag


def live_ashare_forecast(symbol: str) -> SurpriseCard:
    """akshare 盈利预测。失败标缺数据，不编造。"""
    try:
        import akshare as ak

        df = ak.stock_profit_forecast_em()
    except Exception as exc:  # noqa: BLE001
        return SurpriseCard(
            symbol=symbol, kind="eps", surprise_missing=f"缺数据：盈利预测不可用（{exc}）"
        )
    if df is None or getattr(df, "empty", True):
        return SurpriseCard(symbol=symbol, kind="eps", surprise_missing="缺数据：盈利预测为空")
    code_col = next((c for c in df.columns if str(c) in {"代码", "symbol", "证券代码"}), None)
    if code_col is None:
        return SurpriseCard(symbol=symbol, kind="eps", surprise_missing="缺数据：盈利预测无代码列")
    hit = df[df[code_col].astype(str).str.zfill(6) == symbol.zfill(6)]
    if hit.empty:
        return SurpriseCard(symbol=symbol, kind="eps", surprise_missing="缺数据：无该标的盈利预测")
    row = hit.iloc[0]
    cons = row.get("预测EPS均值") or row.get("EPS") or row.get("eps")
    try:
        cons_f = float(cons)
    except (TypeError, ValueError):
        return SurpriseCard(symbol=symbol, kind="eps", surprise_missing="缺数据：预测 EPS 无法解析")
    return SurpriseCard(symbol=symbol, kind="eps", consensus=cons_f, source="akshare")


def live_implied_move(symbol: str) -> float | MissingData:
    """yfinance 期权链平值跨式近似。失败标缺数据。"""
    try:
        import yfinance as yf
    except ImportError as exc:
        return MissingData(f"缺数据：未安装 yfinance（{exc}）")
    try:
        ticker = yf.Ticker(symbol)
        spot = float(
            (ticker.fast_info or {}).get("lastPrice") or ticker.info.get("regularMarketPrice") or 0
        )
        expiries = list(ticker.options or [])
        if not expiries or spot <= 0:
            return MissingData("缺数据：无期权到期日或现价")
        chain = ticker.option_chain(expiries[0])
        calls, puts = chain.calls, chain.puts
        if calls is None or puts is None or calls.empty or puts.empty:
            return MissingData("缺数据：期权链为空")
        calls = calls.copy()
        puts = puts.copy()
        calls["dist"] = (calls["strike"] - spot).abs()
        puts["dist"] = (puts["strike"] - spot).abs()
        c = calls.sort_values("dist").iloc[0]
        p = puts.sort_values("dist").iloc[0]
        call_mid = _mid(c)
        put_mid = _mid(p)
        return implied_move_from_straddle(spot, call_mid, put_mid)
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"缺数据：期权链失败（{exc}）")


def _mid(row) -> float | None:
    bid, ask, last = row.get("bid"), row.get("ask"), row.get("lastPrice")
    try:
        b, a = float(bid), float(ask)
        if b > 0 and a > 0:
            return (b + a) / 2.0
    except (TypeError, ValueError):
        pass
    try:
        return float(last) if last is not None else None
    except (TypeError, ValueError):
        return None


def asof_filter(cards: list[SurpriseCard], on: date) -> list[SurpriseCard]:
    """只保留 asof<=on 的事前预期，防止前视。"""
    out: list[SurpriseCard] = []
    for c in cards:
        if c.asof is None or c.asof <= on:
            out.append(c)
    return out
