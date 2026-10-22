"""经典思想里能写成截面规则的公式。缺字段不估算；过闸才进每日候选。"""

from __future__ import annotations

from zhiku.pipeline.settings import ClassicScreenSettings
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import ScreenRow
from zhiku.screens.ranking import drop_missing, minmax_from_ranks, rank_higher_better, rank_lower_better


def _finite(mapping: dict) -> dict[str, float]:
    out = {}
    for k, v in drop_missing(mapping).items():
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def graham_number_ratio(row: Fundamentals) -> float | None:
    """价格相对格雷厄姆数字：sqrt(PE*PB/22.5)。<1 更便宜。PE、PB 须为正。"""
    if row.pe is None or row.pb is None or row.pe <= 0 or row.pb <= 0:
        return None
    return (row.pe * row.pb / 22.5) ** 0.5


def ncav_to_mkt(row: Fundamentals) -> float | None:
    ncav = row.ncav if row.ncav is not None else row.nwc
    if ncav is None or row.mktcap is None or row.mktcap <= 0:
        return None
    return ncav / row.mktcap


def peg(row: Fundamentals) -> float | None:
    if row.pe is None or row.pe <= 0 or row.eps_growth is None or row.eps_growth <= 0:
        return None
    g = row.eps_growth * 100.0 if row.eps_growth < 1 else row.eps_growth
    if g <= 0:
        return None
    return row.pe / g


def screen_graham(rows: list[Fundamentals], settings: ClassicScreenSettings) -> list[ScreenRow]:
    cheap = {r.symbol: graham_number_ratio(r) for r in rows}
    ncav = {r.symbol: ncav_to_mkt(r) for r in rows}
    cheap_ok = _finite(cheap)
    ncav_ok = _finite(ncav)
    common = [s for s in set(cheap_ok) | set(ncav_ok)]
    if not common:
        return []
    parts: dict[str, float] = {}
    cheap_ranks = rank_lower_better(cheap_ok) if cheap_ok else {}
    ncav_ranks = rank_higher_better(ncav_ok) if ncav_ok else {}
    for s in common:
        vals = [v for v in (cheap_ranks.get(s), ncav_ranks.get(s)) if v is not None]
        if not vals:
            continue
        parts[s] = sum(vals) / len(vals)
    if not parts:
        return []
    scores = minmax_from_ranks(rank_lower_better(parts))
    return _pack(rows, scores, settings, lambda r: {
        "graham_number_ratio": graham_number_ratio(r),
        "ncav_to_mkt": ncav_to_mkt(r),
    })


def screen_lynch_peg(rows: list[Fundamentals], settings: ClassicScreenSettings) -> list[ScreenRow]:
    raw = {r.symbol: peg(r) for r in rows}
    ok = _finite(raw)
    if not ok:
        return []
    scores = minmax_from_ranks(rank_lower_better(ok))
    return _pack(rows, scores, settings, lambda r: {"peg": peg(r), "pe": r.pe, "eps_growth": r.eps_growth})


def _canslim_flags(row: Fundamentals) -> dict[str, int | None]:
    c = None if row.roa is None or row.roa_prev is None else int(row.roa > row.roa_prev)
    a = None
    if row.eps_growth is not None:
        a = int(row.eps_growth > 0)
    elif row.roe_3y:
        a = int(all(x > 0 for x in row.roe_3y))
    n = None if row.listed_years is None else int(row.listed_years >= 1)
    s = None if row.shares_issued is None else int(not row.shares_issued)
    leader = int(bool(row.industry_leader) or (row.momentum_12_1 is not None and row.momentum_12_1 > 0))
    leader_flag = leader if row.momentum_12_1 is not None or row.industry_leader else None
    return {"C": c, "A": a, "N": n, "S": s, "L": leader_flag, "I": None, "M": None}


def screen_canslim(rows: list[Fundamentals], settings: ClassicScreenSettings) -> list[ScreenRow]:
    out: list[ScreenRow] = []
    for row in rows:
        flags = _canslim_flags(row)
        known = [v for v in flags.values() if v is not None]
        if len(known) < 3:
            continue
        raw = sum(known)
        score = 100.0 * raw / max(len(known), 1)
        if score < settings.min_score:
            continue
        out.append(
            ScreenRow(
                symbol=row.symbol,
                name=row.name,
                score=score,
                breakdown={k: (None if v is None else float(v)) for k, v in flags.items()} | {"hits": float(raw)},
                missing=[k for k, v in flags.items() if v is None],
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: settings.top_n]


def screen_livermore(rows: list[Fundamentals], settings: ClassicScreenSettings) -> list[ScreenRow]:
    mom = {r.symbol: r.momentum_12_1 for r in rows}
    near = {r.symbol: r.dist_to_high for r in rows}
    mom_ok = _finite(mom)
    # 接近高点：dist_to_high 越大（越接近 0）越好，所以用负距离
    near_ok = {s: -v for s, v in _finite(near).items()}
    common = [s for s in mom_ok if s in near_ok] or list(mom_ok)
    if not common:
        return []
    ranks = {}
    mr = rank_higher_better({s: mom_ok[s] for s in common if s in mom_ok})
    nr = rank_higher_better({s: near_ok[s] for s in common if s in near_ok})
    for s in common:
        vals = [v for v in (mr.get(s), nr.get(s)) if v is not None]
        if vals:
            ranks[s] = sum(vals) / len(vals)
    scores = minmax_from_ranks(rank_lower_better(ranks))
    return _pack(rows, scores, settings, lambda r: {
        "momentum_12_1": r.momentum_12_1,
        "dist_to_high": r.dist_to_high,
    })


def screen_fisher(rows: list[Fundamentals], settings: ClassicScreenSettings) -> list[ScreenRow]:
    roe = {r.symbol: r.roe_3y_mean for r in rows}
    stab: dict[str, float] = {}
    for r in rows:
        series = r.gross_margin_3y or ([r.gross_margin] if r.gross_margin is not None else [])
        if len(series) >= 2:
            mu = sum(series) / len(series)
            var = sum((x - mu) ** 2 for x in series) / len(series)
            stab[r.symbol] = -var ** 0.5
        elif r.gross_margin is not None:
            stab[r.symbol] = 0.0
    rd = {r.symbol: r.rd_to_sales for r in rows if r.rd_to_sales is not None}
    roe_ok = _finite(roe)
    common = [s for s in roe_ok if s in stab]
    if not common:
        return []
    rr = rank_higher_better({s: roe_ok[s] for s in common})
    sr = rank_higher_better({s: stab[s] for s in common})
    rdr = rank_higher_better(rd) if rd else {}
    combined = {}
    for s in common:
        vals = [rr[s], sr[s]]
        if s in rdr:
            vals.append(rdr[s])
        combined[s] = sum(vals) / len(vals)
    scores = minmax_from_ranks(rank_lower_better(combined))
    return _pack(rows, scores, settings, lambda r: {
        "roe_3y_mean": r.roe_3y_mean,
        "rd_to_sales": r.rd_to_sales,
        "gm_vol": None if r.symbol not in stab else -stab[r.symbol],
    })


def _pack(rows, scores: dict[str, float], settings: ClassicScreenSettings, breakdown) -> list[ScreenRow]:
    by_id = {r.symbol: r for r in rows}
    out = []
    for symbol, score in scores.items():
        if score < settings.min_score:
            continue
        row = by_id[symbol]
        out.append(ScreenRow(symbol=symbol, name=row.name, score=score, breakdown=breakdown(row)))
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: settings.top_n]
