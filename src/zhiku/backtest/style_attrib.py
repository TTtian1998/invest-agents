"""价值类公式超额：分年、按行业、价值/成长风格，以及单变量市场状态开关。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from zhiku.backtest.metrics import rebalance_equal_weight, year_excesses
from zhiku.backtest.pit_gate import (
    backtest_timed_picks,
    month_end_trading_days,
    monthly_screen_picks,
    panel_momentum_12_1,
    universe_ew_picks,
)
from zhiku.factors.dataset import TEST_START, VALID_END, VALID_START
from zhiku.pipeline.settings import BacktestSettings, PipelineSettings

VALUE_SCREENS = ("dividend", "graham")


def ew_index_12_1(close: pd.DataFrame, membership: dict, asof) -> float | None:
    """时点股票池等权 12−1 动量。只用 asof 及之前的收盘。"""
    from zhiku.backtest.universe import members_on

    hist = close.loc[: pd.Timestamp(asof)]
    if len(hist) < 252:
        return None
    members = [s for s in members_on(membership, asof) if s in hist.columns]
    if not members:
        members = [c for c in hist.columns if c in close.columns]
    if not members:
        return None
    m12 = hist.iloc[-252][members].astype(float)
    m1 = hist.iloc[-22][members].astype(float)
    mom = (m1 / m12 - 1.0).replace([np.inf, -np.inf], np.nan)
    val = float(mom.mean(skipna=True))
    return None if not np.isfinite(val) else val


def month_hold_returns(
    close: pd.DataFrame,
    opens: pd.DataFrame | None,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """决策日 T → 下一决策日，T+1 开盘到下一 T+1 开盘的持有收益。"""
    px = close.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    op = None
    if opens is not None and not opens.empty:
        op = opens.copy()
        op.index = pd.to_datetime(op.index)
        op = op.sort_index().reindex(px.index)
    exec_px = op if op is not None else px
    rows = []
    ordered = sorted(pd.Timestamp(d) for d in dates)
    for i, dt in enumerate(ordered[:-1]):
        nxt = ordered[i + 1]
        loc = exec_px.index.searchsorted(dt, side="right")
        loc2 = exec_px.index.searchsorted(nxt, side="right")
        if loc >= len(exec_px.index) or loc2 >= len(exec_px.index):
            continue
        a, b = exec_px.index[int(loc)], exec_px.index[int(loc2)]
        if a == exec_px.index[-1]:
            continue
        pa, pb = exec_px.loc[a], exec_px.loc[b]
        ret = (pb / pa.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        rows.append(ret.rename(dt))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def industry_excess_table(
    picks_by_date: dict,
    month_ret: pd.DataFrame,
    membership: dict,
    industry: dict[str, str],
) -> list[dict[str, Any]]:
    """行业：配置差 × 行业收益 + 行业内选股。"""
    from zhiku.backtest.universe import members_on

    acc: dict[str, list[float]] = {}
    alloc_acc: dict[str, list[float]] = {}
    sel_acc: dict[str, list[float]] = {}
    for dt, ret in month_ret.iterrows():
        members = [s for s in members_on(membership, dt) if s in ret.index and pd.notna(ret[s])]
        held = [s for s in (picks_by_date.get(pd.Timestamp(dt)) or []) if s in members]
        if not members or not held:
            continue
        def _w(names: list[str]) -> dict[str, float]:
            w: dict[str, float] = {}
            n = len(names) or 1
            for s in names:
                ind = industry.get(s) or "未知"
                w[ind] = w.get(ind, 0.0) + 1.0 / n
            return w

        ws, wb = _w(held), _w(members)
        inds = set(ws) | set(wb)
        for ind in inds:
            in_m = [s for s in members if (industry.get(s) or "未知") == ind]
            in_h = [s for s in held if (industry.get(s) or "未知") == ind]
            r_b = float(ret[in_m].mean()) if in_m else 0.0
            r_s = float(ret[in_h].mean()) if in_h else r_b
            alloc = (ws.get(ind, 0.0) - wb.get(ind, 0.0)) * r_b
            sel = ws.get(ind, 0.0) * (r_s - r_b)
            acc.setdefault(ind, []).append(r_s - r_b if in_h else np.nan)
            alloc_acc.setdefault(ind, []).append(alloc)
            sel_acc.setdefault(ind, []).append(sel)
    rows = []
    for ind in sorted(acc):
        xs = [x for x in acc[ind] if pd.notna(x)]
        rows.append(
            {
                "industry": ind,
                "n_months": len(xs),
                "mean_ind_excess": float(np.mean(xs)) if xs else None,
                "allocation": float(np.nansum(alloc_acc.get(ind, []))),
                "selection": float(np.nansum(sel_acc.get(ind, []))),
            }
        )
    rows.sort(key=lambda r: abs(r.get("allocation") or 0) + abs(r.get("selection") or 0), reverse=True)
    return rows


def value_minus_growth(
    month_ret: pd.DataFrame,
    membership: dict,
    *,
    pb: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.Series:
    """每月价值减成长。有 PB 用低/高 PB 三分位，否则用 12−1 低/高动量。"""
    from zhiku.backtest.universe import members_on

    out = {}
    for dt, ret in month_ret.iterrows():
        members = [s for s in members_on(membership, dt) if s in ret.index and pd.notna(ret[s])]
        if len(members) < 12:
            continue
        score = None
        if pb:
            raw = pb.get(pd.Timestamp(dt))
            if raw is None:
                for k, v in pb.items():
                    if pd.Timestamp(k) == pd.Timestamp(dt):
                        raw = v
                        break
            if raw is not None:
                score = raw.reindex(members).astype(float)
                score = score.where(score > 0)
        if score is None or score.dropna().shape[0] < 12:
            continue
        q = score.rank(pct=True)
        value = [s for s in members if pd.notna(q.get(s)) and q[s] <= 1.0 / 3.0]
        growth = [s for s in members if pd.notna(q.get(s)) and q[s] >= 2.0 / 3.0]
        if len(value) < 3 or len(growth) < 3:
            continue
        out[pd.Timestamp(dt)] = float(ret[value].mean() - ret[growth].mean())
    return pd.Series(out, dtype=float)


def value_minus_growth_from_mom(
    month_ret: pd.DataFrame,
    close: pd.DataFrame,
    membership: dict,
) -> pd.Series:
    """无 PB 时：低 12−1 动量当价值、高动量当成长。"""
    from zhiku.backtest.universe import members_on

    out = {}
    for dt, ret in month_ret.iterrows():
        members = [s for s in members_on(membership, dt) if s in ret.index and pd.notna(ret[s])]
        if len(members) < 12:
            continue
        mom = panel_momentum_12_1(close, dt)
        score = mom.reindex(members).astype(float)
        if score.dropna().shape[0] < 12:
            continue
        q = score.rank(pct=True)
        value = [s for s in members if pd.notna(q.get(s)) and q[s] <= 1.0 / 3.0]
        growth = [s for s in members if pd.notna(q.get(s)) and q[s] >= 2.0 / 3.0]
        if len(value) < 3 or len(growth) < 3:
            continue
        out[pd.Timestamp(dt)] = float(ret[value].mean() - ret[growth].mean())
    return pd.Series(out, dtype=float)


def screen_month_excess(
    picks_by_date: dict,
    month_ret: pd.DataFrame,
    membership: dict,
) -> pd.Series:
    from zhiku.backtest.universe import members_on

    out = {}
    for dt, ret in month_ret.iterrows():
        members = [s for s in members_on(membership, dt) if s in ret.index and pd.notna(ret[s])]
        held = [s for s in (picks_by_date.get(pd.Timestamp(dt)) or []) if s in members]
        if not members or not held:
            continue
        out[pd.Timestamp(dt)] = float(ret[held].mean() - ret[members].mean())
    return pd.Series(out, dtype=float)


def style_rotation_stats(excess: pd.Series, vg: pd.Series) -> dict[str, Any]:
    common = excess.dropna().index.intersection(vg.dropna().index)
    if len(common) < 6:
        return {"n": int(len(common)), "rotation": False, "reason": "共同月份不足 6"}
    xs = excess.loc[common].astype(float)
    style = vg.loc[common].astype(float)
    corr = float(xs.corr(style))
    val_on = xs[style > 0]
    grw_on = xs[style <= 0]
    h2 = common[len(common) // 2 :]
    h2_vg = float(style.loc[h2].mean()) if len(h2) else None
    h2_xs = float(xs.loc[h2].mean()) if len(h2) else None
    rotation = bool(corr is not None and corr > 0.25 and h2_vg is not None and h2_vg < 0 and (h2_xs or 0) < 0)
    return {
        "n": int(len(common)),
        "corr_excess_vg": corr,
        "mean_excess_when_value_wins": float(val_on.mean()) if not val_on.empty else None,
        "mean_excess_when_growth_wins": float(grw_on.mean()) if not grw_on.empty else None,
        "h2_mean_vg": h2_vg,
        "h2_mean_excess": h2_xs,
        "rotation": rotation,
        "reason": (
            "后半段成长占优且价值类超额为负，相关为正，判为风格轮动"
            if rotation
            else "相关弱或后半段不是成长占优，不能单靠风格轮动解释"
        ),
    }


def apply_ew_12_1_switch(
    picks_by_date: dict,
    close: pd.DataFrame,
    membership: dict,
    *,
    enable_le: float = 0.0,
) -> dict[pd.Timestamp, list[str]]:
    """单变量：等权指数 12−1 ≤ 阈值才启用价值类篮子，否则空仓。"""
    out: dict[pd.Timestamp, list[str]] = {}
    for key, names in picks_by_date.items():
        dt = pd.Timestamp(key)
        mom = ew_index_12_1(close, membership, dt)
        if mom is not None and mom <= enable_le:
            out[dt] = list(names)
        else:
            out[dt] = []
    return out


def _window_excess(nav: pd.Series, bench: pd.Series, start: date, end: date) -> float | None:
    n = nav.dropna().astype(float)
    b = bench.reindex(n.index).dropna().astype(float)
    common = n.index.intersection(b.index)
    common = common[(common >= pd.Timestamp(start)) & (common <= pd.Timestamp(end))]
    if len(common) < 2:
        return None
    a = float(n.loc[common].iloc[-1] / n.loc[common].iloc[0] - 1.0)
    bb = float(b.loc[common].iloc[-1] / b.loc[common].iloc[0] - 1.0)
    return a - bb


def run_style_attribution(
    store,
    settings: PipelineSettings,
    *,
    asof: date,
    names: Iterable[str] = VALUE_SCREENS,
    out_dir: Path | None = None,
    peek_test: bool = True,
) -> dict[str, Any]:
    from zhiku.backtest.engine import load_price_panel_from_store, sanitize_mixed_adjust
    from zhiku.backtest.pit_gate import load_statement_cache
    from zhiku.backtest.universe import build_pit_universe

    wanted = list(names)
    pit = build_pit_universe(store, asof=asof, years=settings.data.years, universe=settings.data.universe, allow_fetch=False)
    membership = pit.membership
    symbols = list(pit.all_symbols)
    close = load_price_panel_from_store(store, symbols)
    opens = store.price_panel(list(close.columns), field="open")
    raw_px = store.price_panel(list(close.columns), adjust="none", field="close")
    amt_px = store.price_panel(list(close.columns), field="amount")
    if opens is not None and not opens.empty:
        opens = sanitize_mixed_adjust(opens, raw_px)
        opens = opens.reindex(close.index)
    cache = load_statement_cache(store, symbols)
    dates = month_end_trading_days(close.index)
    picks = monthly_screen_picks(
        settings,
        membership,
        close=close,
        cache=cache,
        names=wanted,
        decision_dates=dates,
        raw_close=raw_px,
        amounts=amt_px,
    )
    ew = universe_ew_picks(membership, dates, list(close.columns))
    bench_nav, _, _ = rebalance_equal_weight(
        close, ew, opens=opens, cost_bps=0, slippage_bps=0, membership=membership, fill=settings.backtest.fill
    )
    month_ret = month_hold_returns(close, opens, dates)
    industry = store.listing_industry(symbols)
    pb_map: dict[pd.Timestamp, pd.Series] = {}
    from zhiku.backtest.universe import members_on

    for dt in dates:
        members = [s for s in members_on(membership, dt) if s in close.columns]
        if not members:
            continue
        asof_d = dt.date() if hasattr(dt, "date") else date.fromisoformat(str(dt)[:10])
        pb_map[pd.Timestamp(dt)] = _cheap_pb(cache, asof_d, members, raw_px)

    vg = value_minus_growth(month_ret, membership, pb=pb_map)
    if vg.dropna().shape[0] < 6:
        vg = value_minus_growth_from_mom(month_ret, close, membership)
        vg_kind = "mom_12_1"
    else:
        vg_kind = "pb"

    screens: dict[str, Any] = {}
    rotation_any = False
    for name in wanted:
        hold = picks.get(name, {})
        bt = backtest_timed_picks(
            close,
            hold,
            settings.backtest,
            screen=name,
            opens=opens,
            membership=membership,
            bench_nav=bench_nav,
        )
        excess = screen_month_excess(hold, month_ret, membership)
        rot = style_rotation_stats(excess, vg)
        rotation_any = rotation_any or bool(rot.get("rotation"))
        year_ex = dict(bt.year_excess or {}) or year_excesses(_nav_from_result(bt), bench_nav)
        screens[name] = {
            "annualized": bt.annualized,
            "excess_annualized": bt.excess_annualized,
            "excess_first_half": bt.excess_first_half,
            "excess_second_half": bt.excess_second_half,
            "year_excess": year_ex or bt.year_excess,
            "industry": industry_excess_table(hold, month_ret, membership, industry)[:12],
            "style": rot,
            "n_pick_months": int(sum(1 for v in hold.values() if v)),
        }

    regime = None
    if rotation_any:
        regime = _eval_regime(
            picks,
            close,
            opens,
            membership,
            bench_nav,
            settings.backtest,
            peek_test=peek_test,
        )

    payload = {
        "asof": asof.isoformat(),
        "universe": pit.label,
        "vg_kind": vg_kind,
        "vg_mean": float(vg.mean()) if not vg.empty else None,
        "screens": screens,
        "rotation": rotation_any,
        "regime": regime,
        "split": {
            "valid": f"{VALID_START}–{VALID_END}",
            "test_start": TEST_START.isoformat(),
            "knob": "universe EW 12-1 ≤ 0 才启用价值类",
        },
        "notes": [
            "分年超额与行业 Brinson（配置+选股）。价值减成长用低 PB 对高 PB，PB 不足则用低/高 12−1。",
            "若判为风格轮动，才试等权指数 12−1 开关；阈值 0 预先锁定，验证集决定去留，测试集只看一次。",
        ],
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "style_attrib.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (dest / "style_attrib.md").write_text(_attrib_md(payload), encoding="utf-8")
    return payload


def _cheap_pb(cache, asof: date, symbols: list[str], raw_close: pd.DataFrame | None) -> pd.Series:
    from zhiku.data.compute import pick_num

    ts = pd.Timestamp(asof)
    raw = None
    if raw_close is not None and not raw_close.empty:
        raw = raw_close.loc[:ts]
    out: dict[str, float] = {}
    for symbol in symbols:
        stmts = cache.asof(symbol, asof)
        income = (stmts.get("income") or {}).get("payload") or {}
        balance = (stmts.get("balance") or {}).get("payload") or {}
        bps = pick_num(income, "bps") or pick_num(balance, "bps")
        if bps in (None, 0):
            continue
        px = None
        if raw is not None and symbol in raw.columns:
            hist = raw[symbol].dropna()
            if not hist.empty:
                px = float(hist.iloc[-1])
        if px in (None, 0):
            continue
        out[symbol] = px / float(bps)
    return pd.Series(out, dtype=float)


def _nav_from_result(bt) -> pd.Series:
    eq = getattr(bt, "equity", None)
    if eq is None:
        raw = (bt.to_dict() if hasattr(bt, "to_dict") else {}) or {}
        eq = raw.get("equity")
    if not eq:
        return pd.Series(dtype=float)
    if isinstance(eq, pd.Series):
        return eq
    return pd.Series({pd.Timestamp(a): b for a, b in eq})


def _eval_regime(
    picks: dict[str, dict],
    close: pd.DataFrame,
    opens: pd.DataFrame | None,
    membership: dict,
    bench_nav: pd.Series,
    settings: BacktestSettings,
    *,
    peek_test: bool,
) -> dict[str, Any]:
    """单变量 EW 12-1 开关。验证/测试前窗口决定，测试集只看一次。"""
    out: dict[str, Any] = {"knob": "ew_12_1<=0", "enable_le": 0.0, "screens": {}}
    keep = False
    for name, hold in picks.items():
        gated = apply_ew_12_1_switch(hold, close, membership, enable_le=0.0)
        always = backtest_timed_picks(
            close, hold, settings, screen=name, opens=opens, membership=membership, bench_nav=bench_nav
        )
        switched = backtest_timed_picks(
            close, gated, settings, screen=f"{name}_regime", opens=opens, membership=membership, bench_nav=bench_nav
        )
        # 决策用测试集之前（含验证段），不读测试段
        pre_always = _window_excess(
            _nav_from_result(always), bench_nav, date(2023, 11, 1), date(2025, 10, 31)
        )
        pre_sw = _window_excess(
            _nav_from_result(switched), bench_nav, date(2023, 11, 1), date(2025, 10, 31)
        )
        valid_always = _window_excess(_nav_from_result(always), bench_nav, VALID_START, VALID_END)
        valid_sw = _window_excess(_nav_from_result(switched), bench_nav, VALID_START, VALID_END)
        better = (valid_sw or -999) > (valid_always or -999) + 1e-6
        keep = keep or better
        rec: dict[str, Any] = {
            "pre_test_excess_always": pre_always,
            "pre_test_excess_switch": pre_sw,
            "valid_excess_always": valid_always,
            "valid_excess_switch": valid_sw,
            "keep": better,
            "n_on": int(sum(1 for v in gated.values() if v)),
            "n_off": int(sum(1 for v in gated.values() if not v)),
        }
        out["screens"][name] = rec
    out["kept"] = keep
    if peek_test and keep:
        for name, hold in picks.items():
            gated = apply_ew_12_1_switch(hold, close, membership, enable_le=0.0)
            switched = backtest_timed_picks(
                close, gated, settings, screen=f"{name}_regime", opens=opens, membership=membership, bench_nav=bench_nav
            )
            always = backtest_timed_picks(
                close, hold, settings, screen=name, opens=opens, membership=membership, bench_nav=bench_nav
            )
            out["screens"][name]["test_excess_always"] = _window_excess(
                _nav_from_result(always), bench_nav, TEST_START, date(2099, 12, 31)
            )
            out["screens"][name]["test_excess_switch"] = _window_excess(
                _nav_from_result(switched), bench_nav, TEST_START, date(2099, 12, 31)
            )
        out["test_looks"] = 1
        out["test_note"] = "验证集开关优于常开后才看测试集一次"
    else:
        out["test_looks"] = 0
        out["test_note"] = "验证集开关未优于常开，测试集未看" if not keep else "未请求测试集"
    return out


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _attrib_md(payload: dict[str, Any]) -> str:
    lines = [
        "# dividend / graham 超额拆分",
        "",
        f"股票池：{payload.get('universe')}",
        f"价值−成长：{payload.get('vg_kind')} 月均 {_fmt(payload.get('vg_mean'))}",
        f"风格轮动：{'是' if payload.get('rotation') else '否'}",
        "",
        "## 分年超额（对同池 T+1 等权）",
        "",
        "| 公式 | 年化超额 | 前半 | 后半 | 2023 | 2024 | 2025 | 2026 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rec in (payload.get("screens") or {}).items():
        ye = rec.get("year_excess") or {}
        lines.append(
            f"| {name} | {_fmt(rec.get('excess_annualized'))} | {_fmt(rec.get('excess_first_half'))} | "
            f"{_fmt(rec.get('excess_second_half'))} | {_fmt(ye.get('2023'))} | {_fmt(ye.get('2024'))} | "
            f"{_fmt(ye.get('2025'))} | {_fmt(ye.get('2026'))} |"
        )
    lines += ["", "## 行业贡献（配置+选股，前 8）", ""]
    for name, rec in (payload.get("screens") or {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| 行业 | 月数 | 行业内超额 | 配置 | 选股 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in (rec.get("industry") or [])[:8]:
            lines.append(
                f"| {row.get('industry')} | {row.get('n_months')} | {_fmt(row.get('mean_ind_excess'))} | "
                f"{_fmt(row.get('allocation'))} | {_fmt(row.get('selection'))} |"
            )
        st = rec.get("style") or {}
        lines += [
            "",
            f"与价值−成长相关 {_fmt(st.get('corr_excess_vg'))}；"
            f"价值月超额 {_fmt(st.get('mean_excess_when_value_wins'))}；"
            f"成长月超额 {_fmt(st.get('mean_excess_when_growth_wins'))}；"
            f"{st.get('reason')}",
            "",
        ]
    regime = payload.get("regime")
    if regime:
        lines += [
            "## 市场状态开关（等权 12−1 ≤ 0 启用价值类）",
            "",
            f"验证集决定去留：{'保留' if regime.get('kept') else '不保留'}；测试集查看 {regime.get('test_looks')} 次。",
            f"{regime.get('test_note')}",
            "",
            "| 公式 | 验证常开 | 验证开关 | 测试常开 | 测试开关 | 开/关月 | 保留 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, rec in (regime.get("screens") or {}).items():
            lines.append(
                f"| {name} | {_fmt(rec.get('valid_excess_always'))} | {_fmt(rec.get('valid_excess_switch'))} | "
                f"{_fmt(rec.get('test_excess_always'))} | {_fmt(rec.get('test_excess_switch'))} | "
                f"{rec.get('n_on')}/{rec.get('n_off')} | {'是' if rec.get('keep') else '否'} |"
            )
    lines += ["", "不含持仓或个人信息。", ""]
    return "\n".join(lines)
