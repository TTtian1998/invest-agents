"""Walk-forward / 单变量消融。验证集决定去留，测试集只在全部实验结束后看一次。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from zhiku.backtest.engine import attach_benchmarks, backtest_screen, compute_benchmarks, finalize_gate
from zhiku.backtest.universe import build_pit_universe, members_on
from zhiku.data.store import MarketStore
from zhiku.factors.dataset import (
    TEST_START,
    TRAIN_END,
    VALID_END,
    VALID_START,
    add_ch_ep,
    assign_split,
    attach_industry_size,
    excess_vs_pool,
    label_open_horizon,
    load_flow_features,
    load_fund_features,
    load_price_features,
    make_ch_features,
    make_labels,
    neutralize,
    split_of,
)
from zhiku.factors.qlib_adapter import _quintile_returns, _rank_ic_stats
from zhiku.pipeline.settings import BacktestSettings

COMPARE_FIELDS = ("label", "features", "neutralize", "universe", "model", "freq")


@dataclass
class ExperimentConfig:
    name: str
    label: str = "t2_close"
    features: tuple[str, ...] = ("px",)
    neutralize: bool = False
    universe: str = "csi800"  # csi800 | alla
    score_in: str = "csi800"
    model: str = "mse"  # mse | rank | wf_mse | zscore
    freq: str = "daily"  # daily | month
    versus: str | None = None  # 对照实验名；空则对照 baseline
    notes: str = ""


@dataclass
class ExperimentResult:
    name: str
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    valid: dict[str, Any] = field(default_factory=dict)
    test: dict[str, Any] = field(default_factory=dict)
    top30: dict[str, Any] = field(default_factory=dict)
    kept: bool = False
    notes: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pit_symbols(store: MarketStore, asof: date) -> tuple[list[str], dict]:
    pit = build_pit_universe(store, asof=asof, years=3, universe="csi800", allow_fetch=False)
    return pit.all_symbols, pit.membership


def _month_mask(index: pd.MultiIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(index.get_level_values("date"))
    month_ends = dates.to_series().groupby(dates.to_period("M")).max()
    return dates.isin(set(month_ends))


def changed_fields(parent: ExperimentConfig, cfg: ExperimentConfig) -> list[str]:
    out = []
    for key in COMPARE_FIELDS:
        if getattr(parent, key) != getattr(cfg, key):
            out.append(key)
    return out


class _PoolCache:
    """同一股票池的量价/标签只建一次。"""

    def __init__(self) -> None:
        self.px: dict[str, pd.DataFrame] = {}
        self.ch: dict[str, pd.DataFrame] = {}
        self.flow: dict[str, pd.DataFrame] = {}
        self.fund: dict[str, pd.DataFrame] = {}
        self.labels: dict[tuple[str, str], pd.Series] = {}
        self.econ: dict[str, pd.Series] = {}
        self.bars: dict[str, pd.DataFrame] = {}
        self.score_syms: list[str] | None = None
        self.membership: dict | None = None
        self.alla_syms: list[str] | None = None

    def universe(self, store: MarketStore, asof: date) -> tuple[list[str], dict]:
        if self.score_syms is None:
            self.score_syms, self.membership = pit_symbols(store, asof)
        return self.score_syms, self.membership

    def train_syms(self, store: MarketStore, cfg: ExperimentConfig, asof: date) -> list[str]:
        score_syms, _ = self.universe(store, asof)
        if cfg.universe != "alla":
            return list(score_syms)
        if self.alla_syms is None:
            listings = store.listings(market="CN")
            raw = [s for s in listings["symbol"].tolist() if not str(s).startswith("5")] if not listings.empty else score_syms
            qfq = store.bars(None, adjust="qfq")
            have = set(qfq["symbol"].unique()) if not qfq.empty else set()
            self.alla_syms = [s for s in raw if s in have]
            self.bars["alla"] = qfq
        return list(self.alla_syms)

    def _bars(self, store: MarketStore, key: str, syms: list[str]) -> pd.DataFrame:
        if key not in self.bars:
            self.bars[key] = store.bars(syms, adjust="qfq")
        return self.bars[key]

    def features(self, store: MarketStore, cfg: ExperimentConfig, asof: date) -> pd.DataFrame:
        key = "alla" if cfg.universe == "alla" else "csi800"
        syms = self.train_syms(store, cfg, asof)
        if key not in self.px:
            if key == "alla" and "alla" in self.bars:
                from zhiku.factors.features import make_feature_panel

                self.px[key] = make_feature_panel(self.bars["alla"])
            else:
                self.px[key] = load_price_features(store, syms)
            self.ch[key] = make_ch_features(self.px[key])
        parts = []
        if "px" in cfg.features:
            parts.append(self.px[key])
        if "ch" in cfg.features:
            parts.append(self.ch[key])
        if "flow" in cfg.features:
            if key not in self.flow:
                self.flow[key] = load_flow_features(store, syms)
            parts.append(self.flow[key])
        if "fund" in cfg.features:
            if key not in self.fund:
                cal = self.px[key].index.get_level_values("date").unique()
                if cfg.freq == "month":
                    cal = pd.DatetimeIndex(cal)
                    cal = cal[cal.isin(set(cal.to_series().groupby(cal.to_period("M")).max()))]
                self.fund[key] = add_ch_ep(load_fund_features(store, syms, cal))
            parts.append(self.fund[key])
        if not parts:
            return self.px[key]
        return pd.concat(parts, axis=1)

    def label(self, store: MarketStore, cfg: ExperimentConfig, asof: date) -> pd.Series:
        key = "alla" if cfg.universe == "alla" else "csi800"
        _, membership = self.universe(store, asof)
        lk = (key, cfg.label)
        if lk not in self.labels:
            bars = self._bars(store, key, self.train_syms(store, cfg, asof))
            self.labels[lk] = make_labels(bars, kind=cfg.label, membership=membership)
        return self.labels[lk]

    def economic(self, store: MarketStore, cfg: ExperimentConfig, asof: date) -> pd.Series:
        key = "alla" if cfg.universe == "alla" else "csi800"
        if key not in self.econ:
            _, membership = self.universe(store, asof)
            bars = self._bars(store, key, self.train_syms(store, cfg, asof))
            self.econ[key] = excess_vs_pool(label_open_horizon(bars, 20), membership)
        return self.econ[key]


def build_frame(
    store: MarketStore,
    cfg: ExperimentConfig,
    *,
    asof: date,
    cache: _PoolCache | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    cache = cache or _PoolCache()
    score_syms, membership = cache.universe(store, asof)
    feats = cache.features(store, cfg, asof)
    y = cache.label(store, cfg, asof)
    if feats.empty or y.empty:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=object), pd.Series(dtype=float), membership
    data = feats.join(y, how="inner")
    data = data[data["label"].notna()]
    data = data.loc[:, ~data.columns.duplicated()]
    if cfg.freq == "month" and not data.empty:
        data = data.loc[_month_mask(data.index)]
    industry, size = attach_industry_size(data.index, store, data if "f_log_size" in data.columns else None)
    if cfg.neutralize:
        data["label"] = neutralize(data["label"], industry=industry, size=size)
        for col in [c for c in data.columns if c != "label"]:
            data[col] = neutralize(data[col], industry=industry, size=size)
    data["split"] = assign_split(pd.Series(data.index.get_level_values("date"), index=data.index))
    in_pool = pd.Series(False, index=data.index)
    for dt, part in data.groupby(level="date"):
        allowed = members_on(membership, dt)
        if cfg.score_in == "csi800" and allowed:
            in_pool.loc[part.index] = part.index.get_level_values("symbol").isin(list(allowed))
        else:
            in_pool.loc[part.index] = True
    data["in_score_pool"] = in_pool
    del score_syms
    return data, industry.reindex(data.index), size.reindex(data.index), membership


def _feat_cols(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in {"label", "split", "in_score_pool"}]


def _xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[_feat_cols(data)], data["label"]


def _fit_mse(X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None, y_val: pd.Series | None):
    import lightgbm as lgb

    ds = lgb.Dataset(X, label=y)
    valid = lgb.Dataset(X_val, label=y_val) if X_val is not None and not X_val.empty else None
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(20, verbose=False)] if valid is not None else []
    return lgb.train(
        params,
        ds,
        num_boost_round=120,
        valid_sets=[valid] if valid is not None else None,
        callbacks=callbacks,
    )


def _fit_rank(X: pd.DataFrame, y: pd.Series, dates: pd.Series):
    import lightgbm as lgb

    frame = pd.DataFrame({"y": y.to_numpy(), "date": pd.to_datetime(dates)})
    rel = []
    groups = []
    for _, g in frame.groupby("date"):
        groups.append(len(g))
        try:
            rel.extend(pd.qcut(g["y"].rank(method="first"), 5, labels=False).astype(int).tolist())
        except ValueError:
            rel.extend([2] * len(g))
    ds = lgb.Dataset(X, label=np.array(rel), group=groups)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "verbosity": -1,
        "seed": 42,
        "label_gain": [0, 1, 3, 7, 15],
    }
    return lgb.train(params, ds, num_boost_round=80)


def _predict_zscore(data: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = []
    for _, g in data.groupby(level="date"):
        zparts = []
        for col in cols:
            s = g[col].astype(float)
            sd = float(s.std(ddof=0) or 0.0)
            if sd < 1e-12:
                zparts.append(pd.Series(0.0, index=g.index))
            else:
                zparts.append((s - s.mean()) / sd)
        parts.append(pd.concat(zparts, axis=1).mean(axis=1))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts)


def _metrics(pred: pd.Series, label: pd.Series, dates: pd.Series) -> dict[str, Any]:
    stats = _rank_ic_stats(pred.reset_index(drop=True), label.reset_index(drop=True), dates.reset_index(drop=True))
    q = _quintile_returns(pred.reset_index(drop=True), label.reset_index(drop=True), dates.reset_index(drop=True))
    out = dict(stats)
    out["quintiles"] = q
    out["q5_q1"] = q.get("Q5_minus_Q1")
    return out


def _eval_split(pred: pd.Series, label: pd.Series, econ: pd.Series | None) -> dict[str, Any]:
    dates = pd.Series(pred.index.get_level_values("date"))
    own = _metrics(pred, label.reindex(pred.index), dates)
    payload = {"own_label": own}
    if econ is not None:
        payload["t20_xs"] = _metrics(pred, econ.reindex(pred.index), dates)
    return payload


def train_predict(
    data: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    report_splits: Iterable[str] = ("valid",),
) -> tuple[pd.Series, Any]:
    train = data[data["split"] == "train"]
    valid = data[data["split"] == "valid"]
    feat_cols = _feat_cols(data)
    report = set(report_splits)
    if cfg.model == "zscore":
        pred = _predict_zscore(data, feat_cols)
        return pred, {"kind": "zscore", "features": feat_cols}
    if cfg.model == "rank":
        model = _fit_rank(train[feat_cols], train["label"], pd.Series(train.index.get_level_values("date")))
        pred = pd.Series(model.predict(data[feat_cols]), index=data.index)
        return pred, model
    if cfg.model == "wf_mse":
        pred = pd.Series(index=data.index, dtype=float)
        dates = pd.DatetimeIndex(data.index.get_level_values("date")).unique().sort_values()
        month_ends = dates.to_series().groupby(dates.to_period("M")).max()
        last_model = None
        for end in month_ends:
            role = split_of(end)
            if role == "train":
                continue
            if role == "test" and "test" not in report:
                continue
            hist = data.loc[data.index.get_level_values("date") < end]
            if len(hist) < 80:
                continue
            if role == "test":
                hist_tr = hist[hist.index.get_level_values("date") < pd.Timestamp(TEST_START)]
            else:
                hist_tr = pd.concat(
                    [
                        hist[hist["split"] == "train"],
                        hist[(hist["split"] == "valid") & (hist.index.get_level_values("date") < end)],
                    ]
                )
            if hist_tr.empty:
                continue
            cut = hist_tr.index.get_level_values("date")
            if len(cut.unique()) < 10:
                continue
            val_cut = cut.max() - pd.Timedelta(days=90)
            tr = hist_tr[cut <= val_cut]
            va = hist_tr[cut > val_cut]
            if tr.empty:
                tr = hist_tr
                va = hist_tr.iloc[0:0]
            last_model = _fit_mse(*_xy(tr), *(_xy(va) if not va.empty else (None, None)))
            month_ix = pd.DatetimeIndex(data.index.get_level_values("date")).to_period("M") == pd.Timestamp(end).to_period("M")
            month = data.loc[month_ix]
            if not month.empty:
                pred.loc[month.index] = last_model.predict(month[feat_cols])
        return pred, last_model
    model = _fit_mse(*_xy(train), *(_xy(valid) if not valid.empty else (None, None)))
    pred = pd.Series(model.predict(data[feat_cols]), index=data.index)
    return pred, model


def top30_backtest(
    store: MarketStore,
    pred: pd.Series,
    membership: dict,
    *,
    asof: date,
    top_n: int = 30,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    if pred.empty:
        return {}
    if start is not None:
        pred = pred[pd.DatetimeIndex(pred.index.get_level_values("date")) >= pd.Timestamp(start)]
    if end is not None:
        pred = pred[pd.DatetimeIndex(pred.index.get_level_values("date")) <= pd.Timestamp(end)]
    if pred.empty:
        return {}
    by_month: dict[Any, pd.Timestamp] = {}
    for dt in pred.index.get_level_values("date"):
        p = pd.Timestamp(dt).to_period("M")
        cur = by_month.get(p)
        if cur is None or pd.Timestamp(dt) > cur:
            by_month[p] = pd.Timestamp(dt)
    picks: list[str] = []
    seen: set[str] = set()
    monthly_sets = {}
    for _period, dt in sorted(by_month.items()):
        g = pred[pred.index.get_level_values("date") == dt]
        if isinstance(g.index, pd.MultiIndex):
            g = g.droplevel("date")
        chosen = [str(s) for s in g.sort_values(ascending=False).index.tolist()[:top_n]]
        monthly_sets[pd.Timestamp(dt)] = chosen
        for s in chosen:
            if s not in seen:
                seen.add(s)
                picks.append(s)
    if not picks:
        return {}
    from zhiku.backtest.engine import load_price_panel_from_store

    prices = load_price_panel_from_store(store, picks + ["510300", "000300", "000906"])
    if prices is None or prices.empty:
        return {"notes": ["缺价格，无法回测 Top30"]}
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end)]
    hold_map = {dt: monthly_sets[dt] for dt in monthly_sets}
    settings = BacktestSettings(
        cost_bps=10,
        slippage_bps=5,
        require_beat_benchmark=True,
        require_sharpe_ge_benchmark=True,
        benchmark="universe_ew",
        top_n=top_n,
    )
    benches = compute_benchmarks(prices, settings, membership=membership, universe_note="时点中证800等权")
    bt = backtest_screen(
        prices,
        picks,
        settings,
        screen="qlib_top30",
        membership=hold_map,
        universe_note="量化分月度 Top30（时点池）",
    )
    attach_benchmarks(bt, benches)
    finalize_gate(bt, settings)
    payload = bt.to_dict()
    payload["window"] = {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else asof.isoformat(),
    }
    return payload


def run_one(
    store: MarketStore,
    cfg: ExperimentConfig,
    *,
    asof: date,
    report_splits: tuple[str, ...] = ("valid",),
    cache: _PoolCache | None = None,
    parent: ExperimentConfig | None = None,
) -> tuple[ExperimentResult, pd.Series, Any]:
    data, _industry, _size, membership = build_frame(store, cfg, asof=asof, cache=cache)
    result = ExperimentResult(name=cfg.name, notes=[cfg.notes] if cfg.notes else [])
    result.changed = changed_fields(parent or BASELINE, cfg)
    if data.empty:
        result.notes.append("无样本")
        return result, pd.Series(dtype=float), None
    cache = cache or _PoolCache()
    econ = cache.economic(store, cfg, asof)
    pred, model = train_predict(data, cfg, report_splits=report_splits)
    pred = pred.dropna()
    for split in report_splits:
        part = data[(data["split"] == split) & (data["in_score_pool"] == True)]  # noqa: E712
        if part.empty:
            continue
        p = pred.reindex(part.index).dropna()
        if p.empty:
            continue
        metrics = _eval_split(p, part["label"], econ)
        setattr(result, split, metrics)
        setattr(result, f"n_{split}", int(len(p)))
    result.n_train = int((data["split"] == "train").sum())
    window_start, window_end = None, None
    if "test" in report_splits:
        window_start, window_end = TEST_START, asof
        split_for_top = "test"
    else:
        window_start, window_end = VALID_START, VALID_END
        split_for_top = "valid"
    split_pred = pred.reindex(
        data[(data["split"] == split_for_top) & (data["in_score_pool"] == True)].index  # noqa: E712
    ).dropna()
    if not split_pred.empty:
        result.top30 = top30_backtest(
            store, split_pred, membership, asof=asof, start=window_start, end=window_end
        )
    result.notes.append(
        f"分割 train≤{TRAIN_END} valid {VALID_START}–{VALID_END} test≥{TEST_START}；"
        f"本轮报告 {','.join(report_splits)}"
    )
    if "test" not in report_splits:
        result.notes.append("测试集未计算、未写入")
    return result, pred, model


BASELINE = ExperimentConfig(
    name="baseline_t2_px_csi800_mse",
    label="t2_close",
    features=("px",),
    neutralize=False,
    universe="csi800",
    model="mse",
    freq="daily",
    notes="对照：T+1收→T+2收，仅量价，中证800，单次切分 MSE",
)

# 按调研预期收益/成本排序。除全 A 因内存必须月末外，每个实验只改一个字段。
CANDIDATES = [
    ExperimentConfig("label_t20_xs", label="t20_open_xs", notes="只改标签：T+1开→T+20收 同池超额"),
    ExperimentConfig("feat_ch", features=("px", "ch"), notes="只改特征：加入反转/低换手/低波"),
    ExperimentConfig("feat_fund_ep", features=("px", "fund"), notes="只改特征：时点财报 + EP"),
    ExperimentConfig("neut_ind_size", neutralize=True, notes="只改中性化：行业均值 + 对数规模残差"),
    ExperimentConfig("freq_month", freq="month", notes="只改频率：月末抽样"),
    ExperimentConfig("model_zscore", model="zscore", notes="只改模型：截面 z-score 等权（线性）"),
    ExperimentConfig("model_wf", model="wf_mse", notes="只改训练：按月 walk-forward"),
    ExperimentConfig("model_rank", model="rank", freq="month", versus="freq_month", notes="相对月末对照改目标：LambdaRank"),
    ExperimentConfig(
        "univ_alla",
        universe="alla",
        freq="month",
        versus="freq_month",
        notes="相对月末对照改样本：全 A 训练，中证800 内评估（全 A 日度做不到，必须月末）",
    ),
]


def decide_keep(baseline: ExperimentResult, cand: ExperimentResult) -> bool:
    """只用验证集 T+20 超额 RankICIR / RankIC 判断，不看测试集。"""
    b = ((baseline.valid or {}).get("t20_xs") or {})
    c = ((cand.valid or {}).get("t20_xs") or {})
    b_ir = b.get("rank_icir")
    c_ir = c.get("rank_icir")
    b_ic = b.get("rank_ic") or 0.0
    c_ic = c.get("rank_ic") or 0.0
    if c_ir is None:
        return False
    if b_ir is None:
        return (c_ic or 0) > 0
    return float(c_ir) > float(b_ir) + 0.02 and float(c_ic) >= float(b_ic) - 0.002


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _log_row(res: ExperimentResult) -> dict[str, Any]:
    xs = (res.valid or {}).get("t20_xs") or {}
    top = res.top30 or {}
    ew = ((top.get("benchmarks") or {}).get("universe_ew") or {})
    return {
        "name": res.name,
        "changed": res.changed,
        "kept": res.kept,
        "n_valid": res.n_valid,
        "rank_ic": xs.get("rank_ic"),
        "rank_icir": xs.get("rank_icir"),
        "q5_q1": xs.get("q5_q1"),
        "top30_ann": top.get("annualized"),
        "top30_excess": top.get("excess_annualized"),
        "ew_ann": ew.get("annualized"),
        "top30_sharpe": top.get("sharpe"),
        "ew_sharpe": ew.get("sharpe"),
    }


def _write_log(out_dir: Path, rows: list[ExperimentResult], extra: dict[str, Any] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = [_log_row(r) for r in rows]
    payload = {
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "valid": f"{VALID_START}–{VALID_END}",
            "test_start": TEST_START.isoformat(),
            "rule": "验证集决定去留；测试集只在全部实验结束后看一次",
        },
        "n_experiments": len(rows),
        "test_looks": 0,
        "rows": table,
        "results": [r.to_dict() for r in rows],
    }
    if extra:
        payload.update(extra)
    (out_dir / "experiment_log.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 量化模型实验日志",
        "",
        f"分割：train≤{TRAIN_END} / valid {VALID_START}–{VALID_END} / test≥{TEST_START}",
        "规则：每个实验只改一个变量；去留看验证集 T+20 同池超额；测试集未看（除非文末写了「测试集一次」）。",
        f"已完成实验数：{len(rows)}",
        "",
        "| # | 名称 | 改了什么 | RankIC | RankICIR | Q5-Q1 | Top30年化 | 同池等权 | Top30超额 | 保留 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, r in enumerate(table):
        lines.append(
            f"| {i} | {r['name']} | {','.join(r['changed']) or '对照'} | {_fmt(r['rank_ic'])} | "
            f"{_fmt(r['rank_icir'])} | {_fmt(r['q5_q1'])} | {_fmt(r['top30_ann'], 4)} | "
            f"{_fmt(r['ew_ann'], 4)} | {_fmt(r['top30_excess'], 4)} | {'是' if r['kept'] else '否'} |"
        )
    (out_dir / "experiment_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combo_from(kept: list[ExperimentConfig]) -> ExperimentConfig:
    return ExperimentConfig(
        name="combo_winners",
        label="t20_open_xs" if any(c.label.startswith("t20") for c in kept) else "t2_close",
        features=tuple(dict.fromkeys(f for c in kept for f in c.features)),
        neutralize=any(c.neutralize for c in kept),
        universe="alla" if any(c.universe == "alla" for c in kept) else "csi800",
        model=(
            "wf_mse"
            if any(c.model == "wf_mse" for c in kept)
            else (
                "rank"
                if any(c.model == "rank" for c in kept)
                else ("zscore" if any(c.model == "zscore" for c in kept) else "mse")
            )
        ),
        freq="month" if any(c.freq == "month" for c in kept) else "daily",
        notes="验证集上单独有效的改动组合；测试集仍只报一次",
    )


def lock_recipe(cfg: ExperimentConfig, models_dir: Path) -> Path:
    path = Path(models_dir) / "locked_recipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    raw["features"] = list(cfg.features)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_ablation(
    store: MarketStore,
    *,
    asof: date,
    out_dir: Path | None = None,
    models_dir: Path | None = None,
    peek_test: bool = True,
) -> dict[str, Any]:
    import lightgbm  # noqa: F401

    cache = _PoolCache()
    by_name: dict[str, ExperimentResult] = {}
    rows: list[ExperimentResult] = []
    out = Path(out_dir) if out_dir else None
    done: set[str] = set()
    if out and (out / "ablation.json").is_file():
        try:
            prev = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
            for raw in prev.get("results") or []:
                if raw.get("name") and (raw.get("valid") or raw.get("notes")):
                    res = ExperimentResult(
                        name=raw["name"],
                        n_train=int(raw.get("n_train") or 0),
                        n_valid=int(raw.get("n_valid") or 0),
                        n_test=int(raw.get("n_test") or 0),
                        valid=raw.get("valid") or {},
                        test={},  # 续跑仍不得带出上次误写的测试集
                        top30=raw.get("top30") or {},
                        kept=bool(raw.get("kept")),
                        notes=list(raw.get("notes") or []),
                        changed=list(raw.get("changed") or []),
                    )
                    by_name[res.name] = res
                    rows.append(res)
                    done.add(res.name)
        except (OSError, TypeError, json.JSONDecodeError, KeyError):
            done = set()
            by_name = {}
            rows = []

    if BASELINE.name not in done:
        print(f"[experiment] start {BASELINE.name}", flush=True)
        base_res, _, _ = run_one(store, BASELINE, asof=asof, report_splits=("valid",), cache=cache)
        by_name[BASELINE.name] = base_res
        rows.append(base_res)
        if out:
            _write_log(out, rows)
            (out / "ablation.json").write_text(
                json.dumps({"results": [r.to_dict() for r in rows], "n_experiments": len(rows), "test_looks": 0}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    base_res = by_name[BASELINE.name]

    kept_cfgs: list[ExperimentConfig] = []
    for cfg in CANDIDATES:
        if cfg.name in done:
            if by_name[cfg.name].kept:
                kept_cfgs.append(cfg)
            continue
        parent_cfg = BASELINE
        parent_res = base_res
        if cfg.versus and cfg.versus in by_name:
            parent_res = by_name[cfg.versus]
            parent_cfg = next((c for c in CANDIDATES if c.name == cfg.versus), BASELINE)
            if cfg.versus == BASELINE.name:
                parent_cfg = BASELINE
            elif cfg.versus == "freq_month":
                parent_cfg = next(c for c in CANDIDATES if c.name == "freq_month")
        print(f"[experiment] start {cfg.name} vs {parent_cfg.name}", flush=True)
        res, _, _ = run_one(
            store, cfg, asof=asof, report_splits=("valid",), cache=cache, parent=parent_cfg
        )
        res.kept = decide_keep(parent_res, res)
        xs = ((res.valid or {}).get("t20_xs") or {})
        print(
            f"[experiment] done {cfg.name} keep={res.kept} "
            f"RankIC={xs.get('rank_ic')} RankICIR={xs.get('rank_icir')} Q5-Q1={xs.get('q5_q1')}",
            flush=True,
        )
        if res.kept:
            kept_cfgs.append(cfg)
        by_name[cfg.name] = res
        rows.append(res)
        if out:
            _write_log(out, rows)
            (out / "ablation.json").write_text(
                json.dumps(
                    {"results": [r.to_dict() for r in rows], "n_experiments": len(rows), "test_looks": 0},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    combo = None
    combo_res = None
    pred = model = None
    if "combo_winners" in done:
        combo_res = by_name.get("combo_winners")
        if kept_cfgs:
            combo = _combo_from(kept_cfgs)
    elif kept_cfgs:
        combo = _combo_from(kept_cfgs)
        print(f"[experiment] start {combo.name}", flush=True)
        combo_res, pred, model = run_one(
            store, combo, asof=asof, report_splits=("valid",), cache=cache
        )
        combo_res.kept = decide_keep(base_res, combo_res)
        rows.append(combo_res)
        if out:
            _write_log(out, rows)

    locked = BASELINE
    kept_rows = [r for r in rows if r.kept and r.name != "combo_winners"]
    if kept_rows:
        best = max(
            kept_rows,
            key=lambda r: float(((r.valid or {}).get("t20_xs") or {}).get("rank_icir") or -999),
        )
        locked = next((c for c in (*CANDIDATES, combo) if c and c.name == best.name), BASELINE)
    if combo_res and combo_res.kept and combo is not None:
        locked = combo

    test_payload = None
    if peek_test and out and (out / "ablation.json").is_file():
        try:
            prev_full = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
            if prev_full.get("test_looks") == 1 and prev_full.get("test"):
                test_payload = prev_full["test"]
                peek_test = False
        except (OSError, json.JSONDecodeError):
            pass
    if peek_test:
        test_rows = []
        for cfg in (BASELINE, locked):
            if cfg is None:
                continue
            print(f"[experiment] test-once {cfg.name}", flush=True)
            tres, tpred, tmodel = run_one(
                store, cfg, asof=asof, report_splits=("test",), cache=cache, parent=BASELINE
            )
            test_rows.append(tres.to_dict())
            if cfg.name == locked.name:
                pred, model = tpred, tmodel
        test_payload = {
            "looks": 1,
            "note": "全部验证实验结束后才计算；不回写去留",
            "results": test_rows,
        }
        if out:
            _write_log(out, rows, extra={"test_looks": 1, "test": test_payload})

    if models_dir is not None:
        if locked.name != BASELINE.name or (combo_res and combo_res.kept):
            lock_recipe(locked, Path(models_dir))
        elif (Path(models_dir) / "locked_recipe.json").is_file():
            # 验证集没有赢家时不保留旧锁，避免把未验证配方当生产
            (Path(models_dir) / "locked_recipe.json").unlink(missing_ok=True)

    payload = {
        "asof": asof.isoformat(),
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "valid": f"{VALID_START}–{VALID_END}",
            "test_start": TEST_START.isoformat(),
            "rule": "验证集决定去留，测试集不调参、只看一次",
        },
        "n_experiments": len(rows),
        "test_looks": 1 if test_payload else 0,
        "results": [r.to_dict() for r in rows],
        "kept": [r.name for r in rows if r.kept],
        "locked": locked.name if locked else None,
        "test": test_payload,
    }
    if out is not None:
        (out / "ablation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "payload": payload,
        "combo": combo,
        "combo_res": combo_res,
        "locked": locked,
        "model": model,
        "pred": pred,
        "rows": rows,
    }


def main() -> None:
    import argparse
    import os

    from zhiku.calendar import now_shanghai
    from zhiku.config import Paths

    parser = argparse.ArgumentParser(description="单变量量化消融（测试集只看一次）")
    parser.add_argument("--out", default="/opt/cursor/artifacts/zhiku-quant-research")
    parser.add_argument("--asof", default=None)
    args = parser.parse_args()
    paths = Paths.load()
    store = MarketStore(paths.data_dir)
    asof = date.fromisoformat(args.asof) if args.asof else now_shanghai().date()
    run_ablation(store, asof=asof, out_dir=Path(args.out), models_dir=paths.models_dir)
    del os


if __name__ == "__main__":
    main()
