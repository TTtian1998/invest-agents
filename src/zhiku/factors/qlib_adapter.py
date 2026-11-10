"""qlib 数据准备 / Alpha158+LightGBM 训练 / 预测。

`zk factor prepare` 把本地库日线写成 qlib bin。
`zk factor train` 优先走 pyqlib Alpha158；失败或未安装则用 pandas 特征 + LightGBM。
没模型时预测返回空 dict，日报写「未训练」。
"""

from __future__ import annotations

import json
import pickle
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhiku.calendar import now_shanghai
from zhiku.data.codes import qlib_instrument
from zhiku.data.store import MarketStore
from zhiku.factors.features import FEATURE_NAMES, asof_feature_row, label_forward, make_feature_panel

QLIB_HINT = """未安装量化因子 extra。

  uv pip install -e ".[qlib]"

随后运行 `zk factor prepare` 把本地库日线转成 qlib 格式，再 `zk factor train`。
未安装且未训练时每日流水线只用公式分数，并在报告写「未训练」。
"""


class QlibNotInstalled(RuntimeError):
    pass


def qlib_available() -> bool:
    try:
        import qlib  # noqa: F401

        return True
    except ImportError:
        return False


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def prepare_from_csv(csv_dir: Path, qlib_dir: Path) -> str:
    """兼容旧入口：目录里的 CSV（需 date/symbol/open/high/low/close/volume）转 qlib bin。"""
    frames = []
    for path in sorted(Path(csv_dir).glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "date" in df.columns and "symbol" in df.columns:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{csv_dir} 里没有可用日线 CSV")
    bars = pd.concat(frames, ignore_index=True)
    n = dump_qlib(bars, Path(qlib_dir))
    return f"已将 {csv_dir} 的 CSV 写入 qlib 目录 {qlib_dir}（{n} 个标的）"


def prepare_from_store(
    store: MarketStore,
    qlib_dir: Path,
    *,
    adjust: str = "qfq",
    symbols: list[str] | None = None,
    min_bars: int = 80,
) -> str:
    bars = store.bars(symbols, adjust=adjust)
    if bars.empty:
        return "本地库无日线，请先 zk data sync"
    if min_bars > 0 and "symbol" in bars.columns:
        counts = bars.groupby("symbol").size()
        keep = counts[counts >= min_bars].index
        dropped = int((counts < min_bars).sum())
        bars = bars[bars["symbol"].isin(keep)]
    else:
        dropped = 0
    if bars.empty:
        return "有效日线过短，无法导出 qlib"
    n = dump_qlib(bars, Path(qlib_dir))
    extra = f"（跳过日线不足 {min_bars} 根的 {dropped} 只）" if dropped else ""
    return f"已从本地库导出 {n} 个标的到 {qlib_dir}{extra}"


def dump_qlib(bars: pd.DataFrame, qlib_dir: Path) -> int:
    """写成 qlib 日历 / 合约 / float32 bin（首值是日历起始下标）。"""
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])
    calendar = pd.Index(sorted(df["date"].unique()))
    qlib_dir = Path(qlib_dir)
    (qlib_dir / "calendars").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "instruments").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "features").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "calendars" / "day.txt").write_text(
        "\n".join(d.date().isoformat() for d in calendar) + "\n",
        encoding="utf-8",
    )
    inst_lines = []
    count = 0
    for symbol, part in df.groupby("symbol"):
        inst = qlib_instrument(str(symbol))
        part = part.sort_values("date")
        start = pd.Timestamp(part["date"].iloc[0]).date().isoformat()
        end = pd.Timestamp(part["date"].iloc[-1]).date().isoformat()
        inst_lines.append(f"{inst}\t{start}\t{end}")
        feat_dir = qlib_dir / "features" / inst.lower()
        feat_dir.mkdir(parents=True, exist_ok=True)
        indexed = part.set_index("date")
        for field in ("open", "high", "low", "close", "volume"):
            if field not in indexed.columns:
                continue
            series = indexed[field].astype(float).reindex(calendar)
            _write_bin(series, calendar, feat_dir / f"{field}.day.bin")
        count += 1
    (qlib_dir / "instruments" / "all.txt").write_text("\n".join(inst_lines) + "\n", encoding="utf-8")
    return count


def _write_bin(series: pd.Series, calendar: pd.Index, path: Path) -> None:
    valid = series.first_valid_index()
    if valid is None:
        return
    start_i = int(calendar.get_loc(valid))
    values = series.iloc[start_i:].to_numpy(dtype="float32")
    payload = np.hstack([np.array([start_i], dtype="float32"), values]).astype("<f")
    payload.tofile(path)


def latest_model(models_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return None, None
    panel_pkl = models_dir / "lgb_panel.pkl"
    panel_meta = models_dir / "lgb_panel.meta.json"
    recipe = models_dir / "locked_recipe.json"
    if recipe.is_file() and panel_pkl.is_file():
        meta = {}
        if panel_meta.is_file():
            meta = json.loads(panel_meta.read_text(encoding="utf-8"))
        meta.setdefault("path", str(panel_pkl))
        meta.setdefault("status", "trained")
        return panel_pkl, meta
    selected_pkl = models_dir / "selected.pkl"
    selected_meta = models_dir / "selected.meta.json"
    if selected_pkl.is_file():
        meta = {}
        if selected_meta.is_file():
            meta = json.loads(selected_meta.read_text(encoding="utf-8"))
        meta.setdefault("path", str(selected_pkl))
        meta.setdefault("status", "trained")
        return selected_pkl, meta
    metas = sorted(
        [p for p in models_dir.glob("*.meta.json") if p.name != "selected.meta.json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not metas:
        pkls = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pkls:
            return None, None
        return pkls[0], {"status": "trained", "path": str(pkls[0]), "ic": None, "rank_ic": None}
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    pkl = Path(meta.get("path") or metas[0].with_suffix(".pkl"))
    if not pkl.is_file():
        pkl = metas[0].with_name(metas[0].name.replace(".meta.json", ".pkl"))
    return (pkl if pkl.is_file() else None), meta


def needs_retrain(meta: dict[str, Any] | None, asof: date) -> bool:
    if not meta or meta.get("status") == "未训练":
        return True
    trained = meta.get("trained_on")
    if not trained:
        return True
    try:
        d = date.fromisoformat(str(trained)[:10])
    except ValueError:
        return True
    return (d.year, d.month) < (asof.year, asof.month)


def _score_tuple(meta: dict[str, Any]) -> tuple[float, float, float]:
    def _f(*keys: str) -> float:
        for key in keys:
            v = meta.get(key)
            try:
                if v is not None:
                    return float(v)
            except (TypeError, ValueError):
                continue
        return float("-inf")

    # 滚动样本外：RankICIR（稳定）优先，再 RankIC 均值，IC 仅作备注
    return (_f("rank_icir", "icir"), _f("rank_ic"), _f("ic"))


def pick_better(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """RankICIR → RankIC 均值 → IC。ICIR 0.00x 视为噪声，让位给更稳的模型。"""
    return a if _score_tuple(a) >= _score_tuple(b) else b


def _daily_rank_ic(pred: pd.Series, label: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    rows = []
    for dt, g in frame.groupby("date"):
        if len(g) < 5:
            continue
        val = g["pred"].corr(g["label"], method="spearman")
        if val is not None and pd.notna(val):
            rows.append((dt, float(val)))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({dt: v for dt, v in rows})


def _rank_ic_stats(pred: pd.Series, label: pd.Series, dates: pd.Series) -> dict[str, Any]:
    daily = _daily_rank_ic(pred, label, dates)
    if daily.empty:
        return {"rank_ic": None, "rank_icir": None, "icir": None, "n_oos_days": 0}
    mean = float(daily.mean())
    std = float(daily.std(ddof=0)) if len(daily) >= 2 else None
    ir = (mean / std) if std and std != 0 else None
    return {
        "rank_ic": mean,
        "rank_icir": ir,
        "icir": ir,
        "n_oos_days": int(len(daily)),
    }


def _icir(pred: pd.Series, label: pd.Series, dates: pd.Series) -> float | None:
    return _rank_ic_stats(pred, label, dates).get("rank_icir")


def _quintile_returns(pred: pd.Series, label: pd.Series, dates: pd.Series) -> dict[str, float]:
    """样本外按日五分位，再对日均收益取平均。Q1=最低分，Q5=最高分。"""
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {}
    parts: list[pd.Series] = []
    for _, g in frame.groupby("date"):
        if len(g) < 10:
            continue
        ranks = g["pred"].rank(method="first")
        try:
            bucket = pd.qcut(ranks, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        except ValueError:
            continue
        parts.append(g.groupby(bucket, observed=False)["label"].mean())
    if not parts:
        return {}
    avg = pd.concat(parts, axis=1).mean(axis=1)
    out = {str(k): float(v) for k, v in avg.items() if pd.notna(v)}
    if "Q1" in out and "Q5" in out:
        out["Q5_minus_Q1"] = out["Q5"] - out["Q1"]
    return out


def train_alpha158(qlib_dir: Path, model_path: Path) -> str:
    """旧 CLI 入口。"""
    return train_model(qlib_dir=qlib_dir, model_path=model_path, store=None)


def train_model(
    *,
    qlib_dir: Path,
    model_path: Path,
    store: MarketStore | None,
    start: date | None = None,
    end: date | None = None,
) -> str:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if qlib_available():
        try:
            msg = _train_qlib(Path(qlib_dir), model_path)
            notes.append(msg)
            return msg
        except Exception as exc:  # noqa: BLE001
            notes.append(f"pyqlib Alpha158 训练失败，改用 pandas+LightGBM：{exc}")
    if not lightgbm_available():
        raise QlibNotInstalled(QLIB_HINT.strip() + "\n也需要 lightgbm。")
    if store is None:
        raise QlibNotInstalled("未提供本地库，无法走 pandas 训练。请先 zk data sync。")
    meta = _train_lgb_lite(store, model_path, start=start, end=end)
    meta["notes"] = notes
    _write_meta(model_path, meta)
    return (
        f"已训练 LightGBM（{meta.get('n_samples')} 样本）保存到 {model_path}；"
        f"IC={meta.get('ic')} RankIC={meta.get('rank_ic')} 日期={meta.get('trained_on')}"
    )


def _train_qlib(qlib_dir: Path, model_path: Path) -> str:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.utils import init_instance_by_config

    del Alpha158, DatasetH, init_instance_by_config  # 保留导入探测
    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)
    cal = Path(qlib_dir) / "calendars" / "day.txt"
    if not cal.is_file():
        raise FileNotFoundError("qlib 目录缺日历，请先 zk factor prepare")
    days = [ln.strip() for ln in cal.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(days) < 80:
        raise RuntimeError("日历过短，无法切分训练/验证")
    train_end = days[int(len(days) * 0.7)]
    valid_end = days[int(len(days) * 0.85)]
    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": days[0],
            "end_time": days[-1],
            "fit_start_time": days[0],
            "fit_end_time": train_end,
            "instruments": "all",
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (days[0], train_end),
                "valid": (days[int(len(days) * 0.7) + 1], valid_end),
                "test": (days[int(len(days) * 0.85) + 1], days[-1]),
            },
        },
    }
    from qlib.utils import init_instance_by_config as init_cfg

    dataset = init_cfg(dataset_config)
    model = LGBModel(loss="mse", num_leaves=64, num_boost_round=80, learning_rate=0.05)
    model.fit(dataset)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump({"kind": "qlib_lgb", "model": model}, fh)
    pred = model.predict(dataset, segment="test")
    ic = rank_ic = icir = None
    quintiles: dict[str, float] = {}
    try:
        label = dataset.prepare("test", col_set="label")
        aligned = pd.concat([pred, label], axis=1)
        aligned.columns = ["pred", "label"]
        ic = float(aligned["pred"].corr(aligned["label"]))
        dates = aligned.index.get_level_values(0) if hasattr(aligned.index, "get_level_values") else None
        if dates is not None:
            stats = _rank_ic_stats(
                aligned["pred"].reset_index(drop=True),
                aligned["label"].reset_index(drop=True),
                pd.Series(dates),
            )
            rank_ic = stats.get("rank_ic")
            icir = stats.get("rank_icir")
            quintiles = _quintile_returns(
                aligned["pred"].reset_index(drop=True),
                aligned["label"].reset_index(drop=True),
                pd.Series(dates),
            )
        else:
            rank_ic = float(aligned["pred"].corr(aligned["label"], method="spearman"))
    except Exception:  # noqa: BLE001
        pass
    meta = {
        "kind": "qlib_alpha158",
        "trained_on": now_shanghai().date().isoformat(),
        "ic": ic,
        "rank_ic": rank_ic,
        "rank_icir": icir,
        "icir": icir,
        "quintiles": quintiles,
        "label": "qlib Alpha158 默认 Ref($close,-2)/Ref($close,-1)-1（T+1 收到 T+2 收，无当日收盘前视）",
        "path": str(model_path),
        "qlib_dir": str(qlib_dir),
        "status": "trained",
    }
    _write_meta(model_path, meta)
    return f"qlib Alpha158+LightGBM 已保存 {model_path}；IC={ic} RankIC={rank_ic}"


def _train_lgb_lite(
    store: MarketStore,
    model_path: Path,
    *,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    import lightgbm as lgb

    bars = store.bars(start=start, end=end, adjust="qfq")
    if bars.empty:
        raise RuntimeError("本地库无日线")
    feats = make_feature_panel(bars)
    label = label_forward(bars, 5)
    data = feats.join(label, how="inner")
    data = data[data["label"].notna()]
    # 特征 NaN 交给 LightGBM，不当作插值
    if len(data) < 50:
        raise RuntimeError(f"有效样本过少（{len(data)}），无法训练")
    dates = data.index.get_level_values("date")
    cut = dates[int(len(dates) * 0.8)]
    train = data[dates <= cut]
    test = data[dates > cut]
    X_train = train[FEATURE_NAMES]
    y_train = train["label"]
    ds = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": 42,
    }
    booster = lgb.train(params, ds, num_boost_round=80)
    ic = rank_ic = None
    icir = None
    quintiles: dict[str, float] = {}
    if not test.empty:
        pred = booster.predict(test[FEATURE_NAMES])
        y = test["label"].to_numpy()
        ic = float(pd.Series(pred).corr(pd.Series(y)))
        oos_dates = pd.Series(test.index.get_level_values("date"))
        stats = _rank_ic_stats(pd.Series(pred), test["label"].reset_index(drop=True), oos_dates)
        rank_ic = stats.get("rank_ic")
        icir = stats.get("rank_icir")
        quintiles = _quintile_returns(pd.Series(pred), test["label"].reset_index(drop=True), oos_dates)
    payload = {"kind": "lgb_lite", "model": booster, "features": FEATURE_NAMES}
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return {
        "kind": "lgb_lite",
        "trained_on": now_shanghai().date().isoformat(),
        "ic": ic,
        "rank_ic": rank_ic,
        "rank_icir": icir,
        "icir": icir,
        "quintiles": quintiles,
        "label": "T+1 开盘到 T+5 收盘（无当日收盘前视）",
        "n_samples": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_oos_days": None,
        "path": str(model_path),
        "status": "trained",
        "features": FEATURE_NAMES,
    }


def train_panel_recipe(
    store: MarketStore,
    models_dir: Path,
    *,
    asof: date | None = None,
) -> dict[str, Any]:
    """用验证集锁过的配方训练面板模型；测试集指标只记录不回写调参。"""
    from zhiku.factors.experiment import ExperimentConfig, run_one

    asof = asof or now_shanghai().date()
    recipe_path = Path(models_dir) / "locked_recipe.json"
    if not recipe_path.is_file():
        return {
            "status": "skipped",
            "notes": ["无 locked_recipe.json：验证集消融未锁配方，不把未验证组合写入生产"],
        }
    try:
        raw = json.loads(recipe_path.read_text(encoding="utf-8"))
        params: dict[str, Any] = {
            k: tuple(raw[k]) if k == "features" and isinstance(raw[k], list) else raw[k]
            for k in ExperimentConfig.__dataclass_fields__
            if k in raw
        }
        cfg = ExperimentConfig(**params)
    except (OSError, TypeError, json.JSONDecodeError, KeyError) as exc:
        return {"status": "skipped", "notes": [f"locked_recipe.json 无法读取：{exc}"]}
    res, pred, model = run_one(store, cfg, asof=asof, report_splits=("valid",))
    path = Path(models_dir) / "lgb_panel.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    feat_names = []
    kind = "lgb_panel"
    if cfg.model == "zscore" or (isinstance(model, dict) and model.get("kind") == "zscore"):
        kind = "zscore_panel"
        feat_names = list((model or {}).get("features") or [])
        model = None
    elif model is not None and hasattr(model, "feature_name"):
        feat_names = list(model.feature_name())
    with path.open("wb") as fh:
        pickle.dump({"kind": kind, "model": model, "features": feat_names, "config": cfg.__dict__}, fh)
    meta = {
        "kind": kind,
        "trained_on": now_shanghai().date().isoformat(),
        "ic": ((res.valid.get("t20_xs") or {}) or {}).get("rank_ic"),
        "rank_ic": (res.valid.get("t20_xs") or res.valid.get("own_label") or {}).get("rank_ic"),
        "rank_icir": (res.valid.get("t20_xs") or res.valid.get("own_label") or {}).get("rank_icir"),
        "icir": (res.valid.get("t20_xs") or res.valid.get("own_label") or {}).get("rank_icir"),
        "quintiles": (res.valid.get("t20_xs") or res.valid.get("own_label") or {}).get("quintiles") or {},
        "label": cfg.label,
        "valid": res.valid,
        "test": res.test,
        "top30": res.top30,
        "n_train": res.n_train,
        "n_valid": res.n_valid,
        "n_test": res.n_test,
        "notes": res.notes,
        "path": str(path),
        "status": "trained",
        "features": feat_names,
        "config": cfg.__dict__,
    }
    _write_meta(path, meta)
    return meta


def train_and_select(
    *,
    qlib_dir: Path,
    models_dir: Path,
    store: MarketStore,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """训练 lgb_lite，尝试 Alpha158；按 RankICIR → RankIC 均值选默认。"""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    lite_path = models_dir / "lgb_lite.pkl"
    qlib_path = models_dir / "lgb_alpha158.pkl"
    comparison: dict[str, Any] = {"notes": []}
    lite_meta = _train_lgb_lite(store, lite_path, start=start, end=end)
    _write_meta(lite_path, lite_meta)
    comparison["lgb_lite"] = {
        k: lite_meta.get(k)
        for k in ("kind", "ic", "rank_ic", "rank_icir", "icir", "quintiles", "label", "n_samples", "trained_on")
    }
    qlib_meta: dict[str, Any] | None = None
    if qlib_available():
        try:
            msg = _train_qlib(Path(qlib_dir), qlib_path)
            qlib_meta = json.loads((qlib_path.with_name("lgb_alpha158.meta.json")).read_text(encoding="utf-8"))
            qlib_meta["notes_msg"] = msg
        except Exception as exc:  # noqa: BLE001
            comparison["notes"].append(f"pyqlib Alpha158 训练失败：{exc}")
    else:
        comparison["notes"].append("未安装 pyqlib，默认 lgb_lite")
    if qlib_meta:
        comparison["qlib_alpha158"] = {
            k: qlib_meta.get(k)
            for k in ("kind", "ic", "rank_ic", "rank_icir", "icir", "quintiles", "label", "trained_on")
        }
        winner = pick_better(qlib_meta, lite_meta)
    else:
        winner = lite_meta
    winner = dict(winner)
    winner["selected"] = True
    winner["comparison"] = comparison
    selected_path = models_dir / "selected.pkl"
    src = Path(winner.get("path") or lite_path)
    if src.is_file():
        selected_path.write_bytes(src.read_bytes())
        winner["path"] = str(selected_path)
    _write_meta(selected_path, winner)
    (models_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return winner


def _write_meta(model_path: Path, meta: dict[str, Any]) -> None:
    meta = dict(meta)
    meta.setdefault("path", str(model_path))
    meta.setdefault(
        "saved_at",
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    path = model_path.with_suffix(".meta.json") if model_path.suffix == ".pkl" else Path(str(model_path) + ".meta.json")
    if model_path.name.endswith(".pkl"):
        path = model_path.with_name(model_path.stem + ".meta.json")
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def predict_scores(
    qlib_dir: Path | None = None,
    model_path: Path | None = None,
    *,
    store: MarketStore | None = None,
    asof: date | None = None,
    symbols: list[str] | None = None,
    models_dir: Path | None = None,
) -> dict[str, float]:
    """返回 symbol -> 0-100 分。没模型或不可用则空 dict。"""
    del qlib_dir
    path = Path(model_path) if model_path else None
    meta = None
    if path is None or not path.is_file():
        if models_dir is not None:
            path, meta = latest_model(Path(models_dir))
        if path is None or not path.is_file():
            return {}
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return {}
    if store is None:
        return {}
    asof = asof or now_shanghai().date()
    bars = store.bars(symbols, end=asof, adjust="qfq")
    if bars.empty:
        return {}
    feats = asof_feature_row(bars, asof)
    if feats.empty:
        return {}
    if isinstance(payload, dict) and payload.get("kind") in {"lgb_lite", "lgb_panel", "zscore_panel"}:
        cfg = payload.get("config") or {}
        feat_keys = tuple(cfg.get("features") or ("px",))
        if payload.get("kind") in {"lgb_panel", "zscore_panel"}:
            from zhiku.factors.dataset import asof_score_frame

            feats = asof_score_frame(store, list(feats.index if symbols is None else symbols), asof, feat_keys)
            if feats.empty:
                return {}
            if cfg.get("neutralize"):
                from zhiku.factors.dataset import attach_industry_size, neutralize

                idx = pd.MultiIndex.from_arrays(
                    [[pd.Timestamp(asof)] * len(feats), feats.index.astype(str)],
                    names=["date", "symbol"],
                )
                tmp = feats.copy()
                tmp.index = idx
                industry, size = attach_industry_size(tmp.index, store, tmp if "f_log_size" in tmp.columns else None)
                for col in tmp.columns:
                    tmp[col] = neutralize(tmp[col], industry=industry, size=size)
                feats = tmp.droplevel("date")
        model = payload.get("model")
        cols = payload.get("features") or list(feats.columns)
        X = feats.reindex(columns=cols)
        if payload.get("kind") == "zscore_panel" or cfg.get("model") == "zscore":
            z = X.astype(float)
            pred = ((z - z.mean()) / z.std(ddof=0).replace(0, np.nan)).mean(axis=1)
            series = pred
        else:
            if model is None:
                return {}
            pred = model.predict(X)
            series = pd.Series(pred, index=feats.index)
    else:
        lite = None
        if models_dir is not None:
            lite_path = Path(models_dir) / "lgb_lite.pkl"
            if lite_path.is_file():
                try:
                    with lite_path.open("rb") as fh:
                        lite = pickle.load(fh)
                except Exception:
                    lite = None
        if not (isinstance(lite, dict) and lite.get("kind") == "lgb_lite"):
            return {}
        model = lite["model"]
        cols = lite.get("features") or FEATURE_NAMES
        X = feats.reindex(columns=cols)
        pred = model.predict(X)
        series = pd.Series(pred, index=feats.index)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {}
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-12:
        scaled = pd.Series(50.0, index=series.index)
    else:
        scaled = (series - lo) / (hi - lo) * 100.0
    return {str(k): float(v) for k, v in scaled.items()}


def model_status(models_dir: Path | None, model_path: Path | None = None) -> dict[str, Any]:
    if model_path and Path(model_path).is_file():
        meta_path = Path(model_path).with_name(Path(model_path).stem + ".meta.json")
        if meta_path.is_file():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {"status": "trained", "path": str(model_path), "ic": None, "rank_ic": None}
    if models_dir:
        _, meta = latest_model(Path(models_dir))
        if meta:
            return meta
    return {"status": "未训练", "ic": None, "rank_ic": None, "trained_on": None}
