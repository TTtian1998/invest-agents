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


def prepare_from_store(store: MarketStore, qlib_dir: Path, *, adjust: str = "qfq") -> str:
    bars = store.bars(adjust=adjust)
    if bars.empty:
        return "本地库无日线，请先 zk data sync"
    n = dump_qlib(bars, Path(qlib_dir))
    return f"已从本地库导出 {n} 个标的到 {qlib_dir}"


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
    def _f(key: str) -> float:
        v = meta.get(key)
        try:
            return float(v) if v is not None else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    return (_f("rank_ic"), _f("ic"), _f("icir"))


def pick_better(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """RankIC → IC → ICIR，取更好的。"""
    return a if _score_tuple(a) >= _score_tuple(b) else b


def _icir(pred: pd.Series, label: pd.Series, dates: pd.Series) -> float | None:
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return None
    daily = frame.groupby("date").apply(
        lambda g: g["pred"].corr(g["label"], method="spearman") if len(g) >= 5 else np.nan
    )
    daily = daily.dropna()
    if len(daily) < 3 or float(daily.std()) == 0:
        return None
    return float(daily.mean() / daily.std())


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
    try:
        label = dataset.prepare("test", col_set="label")
        aligned = pd.concat([pred, label], axis=1)
        aligned.columns = ["pred", "label"]
        ic = float(aligned["pred"].corr(aligned["label"]))
        rank_ic = float(aligned["pred"].corr(aligned["label"], method="spearman"))
        dates = aligned.index.get_level_values(0) if hasattr(aligned.index, "get_level_values") else None
        if dates is not None:
            icir = _icir(aligned["pred"].reset_index(drop=True), aligned["label"].reset_index(drop=True), pd.Series(dates))
    except Exception:  # noqa: BLE001
        pass
    meta = {
        "kind": "qlib_alpha158",
        "trained_on": now_shanghai().date().isoformat(),
        "ic": ic,
        "rank_ic": rank_ic,
        "icir": icir,
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
    if not test.empty:
        pred = booster.predict(test[FEATURE_NAMES])
        y = test["label"].to_numpy()
        ic = float(pd.Series(pred).corr(pd.Series(y)))
        rank_ic = float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
        icir = _icir(pd.Series(pred), test["label"].reset_index(drop=True), pd.Series(test.index.get_level_values("date")))
    payload = {"kind": "lgb_lite", "model": booster, "features": FEATURE_NAMES}
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return {
        "kind": "lgb_lite",
        "trained_on": now_shanghai().date().isoformat(),
        "ic": ic,
        "rank_ic": rank_ic,
        "icir": icir,
        "n_samples": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "path": str(model_path),
        "status": "trained",
        "features": FEATURE_NAMES,
    }


def train_and_select(
    *,
    qlib_dir: Path,
    models_dir: Path,
    store: MarketStore,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """训练 lgb_lite，尝试 Alpha158；按 RankIC/IC/ICIR 选默认。"""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    lite_path = models_dir / "lgb_lite.pkl"
    qlib_path = models_dir / "lgb_alpha158.pkl"
    comparison: dict[str, Any] = {"notes": []}
    lite_meta = _train_lgb_lite(store, lite_path, start=start, end=end)
    _write_meta(lite_path, lite_meta)
    comparison["lgb_lite"] = {k: lite_meta.get(k) for k in ("kind", "ic", "rank_ic", "icir", "n_samples", "trained_on")}
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
            k: qlib_meta.get(k) for k in ("kind", "ic", "rank_ic", "icir", "trained_on")
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
    if isinstance(payload, dict) and payload.get("kind") == "lgb_lite":
        model = payload["model"]
        cols = payload.get("features") or FEATURE_NAMES
        X = feats.reindex(columns=cols)
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
