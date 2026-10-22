"""Trusted local model publication is explicit, versioned, and fail-closed."""

from __future__ import annotations

import json
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhiku.data.store import MarketStore
from zhiku.factors import model_artifacts
from zhiku.factors.experiment import ExperimentResult
from zhiku.factors.model_artifacts import ModelArtifactError, artifact_digest, publish_model
from zhiku.factors.qlib_adapter import (
    _save_training_artifact,
    latest_model,
    model_status,
    predict_published_scores,
    predict_scores,
    train_panel_recipe,
)

ASOF = date(2026, 10, 20)
FINGERPRINT = "test-config-v1"


@pytest.fixture(autouse=True)
def freeze_publication_clock(monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 20, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(model_artifacts, "datetime", FrozenDatetime)


class ToyModel:
    def __init__(self, output: str = "finite") -> None:
        self.output = output

    def feature_name(self) -> list[str]:
        return ["ret_1"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.output == "nan":
            return np.full(len(frame), np.nan)
        if self.output == "empty":
            return np.array([])
        if self.output == "error":
            raise ValueError("unfitted")
        if self.output == "reverse":
            return np.arange(len(frame) - 1, -1, -1, dtype=float)
        return np.arange(len(frame), dtype=float)


class InvalidTreeModel(ToyModel):
    def dump_model(self) -> dict:
        return {"tree_info": [{"tree_structure": {"leaf_value": float("inf")}}]}


def candidate(
    root: Path,
    name: str = "candidate",
    *,
    kind: str = "lgb_lite",
    model: ToyModel | None = None,
    changes: dict | None = None,
    payload_changes: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.pkl"
    payload = {"kind": kind, "model": model or ToyModel(), "features": ["ret_1"]}
    payload.update(payload_changes or {})
    data = pickle.dumps(payload)
    path.write_bytes(data)
    meta = {
        "kind": kind,
        "status": "trained",
        "trained_on": ASOF.isoformat(),
        "provider_kind": "live",
        "config_fingerprint": FINGERPRINT,
        "data_asof": ASOF.isoformat(),
        "features": ["ret_1"],
        "n_train": 100,
        "artifact_sha256": artifact_digest(data),
        "path": str(path),
    }
    meta.update(changes or {})
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def approve(root: Path, path: Path, **kwargs) -> dict:
    return publish_model(root, path, asof=ASOF, config_fingerprint=FINGERPRINT, **kwargs)


def status(root: Path, **kwargs) -> dict:
    args = {"asof": ASOF, "config_fingerprint": FINGERPRINT, **kwargs}
    return model_status(root, **args)


def test_candidates_never_become_live_implicitly(tmp_path: Path) -> None:
    candidate(tmp_path, "selected")
    candidate(tmp_path, "lgb_panel")
    (tmp_path / "locked_recipe.json").write_text("{}")
    assert status(tmp_path)["reason"] == "not_published"
    assert latest_model(tmp_path, asof=ASOF, config_fingerprint=FINGERPRINT) == (None, None)
    assert not (tmp_path / "approved.json").exists()


def test_publication_pins_a_version_and_ignores_newer_research(tmp_path: Path) -> None:
    source = candidate(tmp_path)
    published = approve(tmp_path, source)
    approved_bytes = (tmp_path / "approved.json").read_bytes()
    model_bytes = Path(published["path"]).read_bytes()
    source.write_bytes(b"new broken research run")
    candidate(tmp_path, "selected")
    candidate(tmp_path, "lgb_panel")
    (tmp_path / "locked_recipe.json").write_text("{}")
    current = status(tmp_path)
    assert current["available"] is True
    assert current["version"] == published["version"]
    assert current["publication_status"] == "approved"
    assert Path(current["path"]).read_bytes() == model_bytes
    assert (tmp_path / "approved.json").read_bytes() == approved_bytes


def test_republication_keeps_previous_version_bytes(tmp_path: Path) -> None:
    source = candidate(tmp_path)
    first = approve(tmp_path, source)
    first_model = Path(first["path"]).read_bytes()
    second = approve(tmp_path, source)
    assert first["version"] != second["version"]
    assert Path(first["path"]).read_bytes() == first_model
    assert status(tmp_path)["version"] == second["version"]


@pytest.mark.parametrize(
    ("changes", "payload_changes", "reason"),
    [
        ({"n_train": 0}, {}, "empty_training"),
        ({"n_train": None}, {}, "empty_training"),
        ({"status": "failed"}, {}, "invalid_artifact"),
        ({"kind": "qlib_lgb"}, {}, "identity_mismatch"),
        ({"features": []}, {"features": []}, "invalid_artifact"),
        ({"features": ["other"]}, {}, "identity_mismatch"),
        ({}, {"model": None}, "invalid_artifact"),
        ({"artifact_sha256": None}, {}, "identity_mismatch"),
        ({"data_asof": None}, {}, "invalid_artifact"),
        ({"trained_on": "2026-10-21"}, {}, "future_model"),
        ({"data_asof": "2026-10-21"}, {}, "future_model"),
        ({"data_asof": "2026-01-01"}, {}, "stale_model"),
        ({"provider_kind": "demo"}, {}, "identity_mismatch"),
        ({"provider_kind": None}, {}, "identity_mismatch"),
        ({"config_fingerprint": None}, {}, "identity_mismatch"),
        ({"config_fingerprint": "different"}, {}, "identity_mismatch"),
    ],
)
def test_bad_publication_preserves_last_approval(
    tmp_path: Path,
    changes: dict,
    payload_changes: dict,
    reason: str,
) -> None:
    original = approve(tmp_path, candidate(tmp_path))
    pointer = (tmp_path / "approved.json").read_bytes()
    bad = candidate(tmp_path, "bad", changes=changes, payload_changes=payload_changes)
    with pytest.raises(ModelArtifactError) as error:
        approve(tmp_path, bad)
    assert error.value.reason == reason
    assert (tmp_path / "approved.json").read_bytes() == pointer
    assert status(tmp_path)["version"] == original["version"]


@pytest.mark.parametrize("output", ["nan", "empty", "error"])
def test_publication_probes_prediction_capability(tmp_path: Path, output: str) -> None:
    source = candidate(tmp_path, model=ToyModel(output))
    with pytest.raises(ModelArtifactError, match="预测验证失败"):
        approve(tmp_path, source)
    assert status(tmp_path)["reason"] == "not_published"


def test_missing_publication_fingerprint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError) as error:
        publish_model(tmp_path, candidate(tmp_path), asof=ASOF, config_fingerprint="")
    assert error.value.reason in {"config_mismatch", "identity_mismatch"}
    assert not (tmp_path / "approved.json").exists()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"asof": ASOF - timedelta(days=1)}, "future_model"),
        ({"asof": ASOF + timedelta(days=46)}, "stale_model"),
        ({"provider_kind": "demo"}, "provider_mismatch"),
        ({"provider_kind": "fixture"}, "provider_mismatch"),
        ({"config_fingerprint": None}, "config_mismatch"),
        ({"config_fingerprint": "other"}, "config_mismatch"),
    ],
)
def test_formal_resolution_enforces_context(tmp_path: Path, kwargs: dict, reason: str) -> None:
    approve(tmp_path, candidate(tmp_path))
    result = status(tmp_path, **kwargs)
    assert result["available"] is False
    assert result["reason"] == reason


def test_fixture_alias_is_demo_not_live(tmp_path: Path) -> None:
    approve(
        tmp_path, candidate(tmp_path, changes={"provider_kind": "demo"}), provider_kind="fixture"
    )
    assert status(tmp_path, provider_kind="demo")["available"] is True
    assert status(tmp_path, provider_kind="fixture")["available"] is True
    assert status(tmp_path, provider_kind="live")["reason"] == "provider_mismatch"


@pytest.mark.parametrize("target", ["model.pkl", "manifest.json"])
def test_published_tampering_fails_closed(tmp_path: Path, target: str) -> None:
    published = approve(tmp_path, candidate(tmp_path))
    (Path(published["path"]).parent / target).write_bytes(b"tampered")
    assert status(tmp_path)["reason"] == "identity_mismatch"


def test_bad_json_and_pointer_path_traversal_fail_closed(tmp_path: Path) -> None:
    pointer = tmp_path / "approved.json"
    pointer.write_text("{")
    assert status(tmp_path)["reason"] == "invalid_artifact"
    pointer.write_text(json.dumps({"schema_version": 1, "version": "../../candidate"}))
    assert status(tmp_path)["reason"] == "invalid_artifact"


def test_failed_pointer_write_preserves_last_approval(tmp_path: Path, monkeypatch) -> None:
    original = approve(tmp_path, candidate(tmp_path))
    pointer = (tmp_path / "approved.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(model_artifacts, "atomic_write", fail)
    with pytest.raises(OSError, match="disk full"):
        approve(tmp_path, candidate(tmp_path, "next"))
    assert (tmp_path / "approved.json").read_bytes() == pointer
    assert status(tmp_path)["version"] == original["version"]


def test_unsupported_qlib_never_scores_lite_fallback(tmp_path: Path) -> None:
    selected = candidate(tmp_path, "selected", kind="qlib_lgb")
    candidate(tmp_path, "lgb_lite")
    with pytest.raises(ModelArtifactError) as error:
        approve(tmp_path, selected)
    assert error.value.reason == "unsupported_kind"
    store = price_store(tmp_path / "data")
    assert predict_scores(model_path=selected, models_dir=tmp_path, store=store, asof=ASOF) == {}


def price_store(root: Path) -> MarketStore:
    store = MarketStore(root)
    rows = []
    for symbol in ("000001", "600519", "000333"):
        for i, day in enumerate(pd.bdate_range(end=ASOF, periods=80)):
            rows.append(
                {
                    "symbol": symbol,
                    "market": "CN",
                    "date": day.date(),
                    "adjust": "qfq",
                    "open": 10 + i * 0.1,
                    "high": 11 + i * 0.1,
                    "low": 9 + i * 0.1,
                    "close": 10 + i * 0.1,
                    "volume": 1000,
                    "turnover": 1.0,
                }
            )
    store.upsert_bars(pd.DataFrame(rows))
    return store


def test_formal_scores_only_use_matching_approved_identity(tmp_path: Path) -> None:
    source = candidate(tmp_path)
    store = price_store(tmp_path / "data")
    assert (
        predict_scores(models_dir=tmp_path, store=store, asof=ASOF, config_fingerprint=FINGERPRINT)
        == {}
    )
    # Explicit research inference remains possible before promotion.
    assert predict_scores(model_path=source, store=store, asof=ASOF)
    published = approve(tmp_path, source)
    scores = predict_scores(
        models_dir=tmp_path, store=store, asof=ASOF, config_fingerprint=FINGERPRINT
    )
    assert set(scores.values()) == {0.0, 50.0, 100.0}
    assert status(tmp_path)["version"] == published["version"]
    assert (
        predict_scores(models_dir=tmp_path, store=store, asof=ASOF, config_fingerprint="changed")
        == {}
    )
    assert (
        predict_scores(
            model_path=tmp_path / "missing.pkl",
            models_dir=tmp_path,
            store=store,
            asof=ASOF,
            config_fingerprint=FINGERPRINT,
        )
        == {}
    )


def test_empty_panel_training_does_not_replace_research_or_publication(
    tmp_path: Path, monkeypatch
) -> None:
    source = candidate(tmp_path, "lgb_panel")
    original = approve(tmp_path, source)
    model_bytes = source.read_bytes()
    meta_bytes = source.with_suffix(".meta.json").read_bytes()
    (tmp_path / "locked_recipe.json").write_text(json.dumps({"name": "empty"}))
    result = ExperimentResult(name="empty", notes=["无样本"])
    monkeypatch.setattr(
        "zhiku.factors.experiment.run_one", lambda *a, **k: (result, pd.Series(dtype=float), None)
    )
    failed = train_panel_recipe(MarketStore(tmp_path / "data"), tmp_path, asof=ASOF)
    assert failed["status"] == "failed"
    assert failed["reason"] == "empty_training"
    assert source.read_bytes() == model_bytes
    assert source.with_suffix(".meta.json").read_bytes() == meta_bytes
    assert status(tmp_path)["version"] == original["version"]


def test_invalid_training_model_does_not_replace_research(tmp_path: Path) -> None:
    source = candidate(tmp_path)
    before = source.read_bytes()
    meta = json.loads(source.with_suffix(".meta.json").read_text())
    with pytest.raises(ModelArtifactError):
        _save_training_artifact(
            source, {"kind": "lgb_lite", "model": ToyModel("nan"), "features": ["ret_1"]}, meta
        )
    assert source.read_bytes() == before


def test_successful_panel_training_stays_research_and_binds_identity(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "locked_recipe.json").write_text(json.dumps({"name": "panel"}))
    result = ExperimentResult(name="panel", n_train=100, n_valid=3)
    monkeypatch.setattr(
        "zhiku.factors.experiment.run_one",
        lambda *a, **k: (result, pd.Series([1.0, 2.0, 3.0]), ToyModel()),
    )
    trained = train_panel_recipe(price_store(tmp_path / "data"), tmp_path, asof=ASOF)
    assert trained["status"] == "trained"
    assert not (tmp_path / "approved.json").exists()
    assert status(tmp_path)["reason"] == "not_published"
    research = model_status(None, Path(trained["path"]))
    assert research["publication_status"] == "research"
    assert research["artifact_sha256"] == artifact_digest(Path(trained["path"]).read_bytes())
    assert research["data_asof"] == ASOF.isoformat()


def test_metadata_failure_restores_last_research_model(tmp_path: Path, monkeypatch) -> None:
    from zhiku.factors import qlib_adapter

    source = candidate(tmp_path)
    model_bytes = source.read_bytes()
    meta_bytes = source.with_suffix(".meta.json").read_bytes()
    meta = json.loads(meta_bytes)
    real_write = qlib_adapter.atomic_write

    def fail_metadata(path: Path, content: bytes) -> None:
        if path.suffix == ".json":
            raise OSError("metadata write failed")
        real_write(path, content)

    monkeypatch.setattr(qlib_adapter, "atomic_write", fail_metadata)
    with pytest.raises(OSError, match="metadata write failed"):
        _save_training_artifact(
            source, {"kind": "lgb_lite", "model": ToyModel("changed"), "features": ["ret_1"]}, meta
        )
    assert source.read_bytes() == model_bytes
    assert source.with_suffix(".meta.json").read_bytes() == meta_bytes


def test_offline_lightgbm_training_never_approves_its_output(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from zhiku.factors import qlib_adapter

    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(
            Dataset=lambda *a, **k: object(),
            train=lambda *a, **k: ToyModel(),
        ),
    )
    monkeypatch.setattr(qlib_adapter, "qlib_available", lambda: False)
    monkeypatch.setattr(qlib_adapter, "lightgbm_available", lambda: True)
    source = tmp_path / "trained.pkl"
    store = price_store(tmp_path / "data")
    message = qlib_adapter.train_model(
        qlib_dir=tmp_path / "qlib",
        model_path=source,
        store=store,
        provider_kind="fixture",
        config_fingerprint=FINGERPRINT,
    )
    assert "已训练" in message
    assert model_status(None, source)["n_train"] > 0
    assert model_status(None, source)["provider_kind"] == "demo"
    assert model_status(None, source)["config_fingerprint"] == FINGERPRINT
    assert model_status(None, source)["artifact_sha256"] == artifact_digest(source.read_bytes())
    assert not (tmp_path / "approved.json").exists()
    assert predict_scores(model_path=source, store=store, asof=ASOF)


def test_future_publication_is_not_backfilled_into_historical_reports(tmp_path: Path) -> None:
    source = candidate(tmp_path)
    publish_model(tmp_path, source, asof=ASOF + timedelta(days=1), config_fingerprint=FINGERPRINT)
    assert status(tmp_path)["reason"] == "future_model"


def test_zscore_publication_preserves_its_declared_model_kind(tmp_path: Path, monkeypatch) -> None:
    source = candidate(
        tmp_path,
        kind="zscore_panel",
        payload_changes={"model": None, "config": {"model": "zscore", "features": ("px",)}},
        changes={"config": {"model": "zscore", "features": ["px"]}},
    )
    approved = approve(tmp_path, source)
    assert approved["kind"] == "zscore_panel"
    monkeypatch.setattr(
        "zhiku.factors.dataset.asof_score_frame",
        lambda *a, **k: pd.DataFrame(
            {"ret_1": [1.0, 1.0, 1.0]},
            index=["000001", "600519", "000333"],
        ),
    )
    scores = predict_scores(
        models_dir=tmp_path,
        store=price_store(tmp_path / "data"),
        asof=ASOF,
        config_fingerprint=FINGERPRINT,
    )
    assert scores == {"000001": 50.0, "600519": 50.0, "000333": 50.0}


def test_nonfinite_parameters_are_rejected_even_if_probe_is_finite(tmp_path: Path) -> None:
    source = candidate(tmp_path, model=InvalidTreeModel())
    with pytest.raises(ModelArtifactError, match="树参数为空或包含非有限值"):
        approve(tmp_path, source)


def test_unbound_research_artifact_cannot_be_promoted(tmp_path: Path) -> None:
    source = candidate(tmp_path, changes={"config_fingerprint": None})
    assert model_status(None, source)["publication_status"] == "research"
    with pytest.raises(ModelArtifactError) as error:
        approve(tmp_path, source)
    assert error.value.reason == "identity_mismatch"
    assert status(tmp_path)["reason"] == "not_published"


def test_score_metadata_pins_one_version_during_concurrent_approval(
    tmp_path: Path, monkeypatch
) -> None:
    from zhiku.factors import qlib_adapter

    first_source = candidate(tmp_path, "first", model=ToyModel())
    second_source = candidate(tmp_path, "second", model=ToyModel("reverse"))
    first = approve(tmp_path, first_source)
    store = price_store(tmp_path / "data")
    expected = predict_scores(model_path=first_source, store=store, asof=ASOF)
    other = predict_scores(model_path=second_source, store=store, asof=ASOF)
    assert expected != other
    original_resolver = qlib_adapter.resolve_published_model
    calls = []
    newer = {}

    def resolve_then_publish(*args, **kwargs):
        artifact = original_resolver(*args, **kwargs)
        calls.append(artifact.metadata["version"])
        newer.update(approve(tmp_path, second_source))
        return artifact

    monkeypatch.setattr(qlib_adapter, "resolve_published_model", resolve_then_publish)
    scores, metadata = predict_published_scores(
        tmp_path,
        store=store,
        asof=ASOF,
        config_fingerprint=FINGERPRINT,
    )
    assert calls == [first["version"]]
    assert newer["version"] != first["version"]
    assert scores == expected
    assert metadata["version"] == first["version"]
    assert metadata["artifact_sha256"] == first["artifact_sha256"]
    assert metadata["path"] == first["path"]
    assert json.loads((tmp_path / "approved.json").read_text())["version"] == newer["version"]


def test_pinned_prediction_returns_unavailable_status_without_publication(tmp_path: Path) -> None:
    scores, metadata = predict_published_scores(
        tmp_path,
        store=price_store(tmp_path / "data"),
        asof=ASOF,
        config_fingerprint=FINGERPRINT,
    )
    assert scores == {}
    assert metadata["available"] is False
    assert metadata["reason"] == "not_published"


def test_live_approval_cannot_be_backdated_into_reports(tmp_path: Path, monkeypatch) -> None:
    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 22, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(model_artifacts, "datetime", LaterDatetime)
    published = approve(tmp_path, candidate(tmp_path))
    assert published["published_asof"] == ASOF.isoformat()
    assert published["published_at"] == "2026-10-22T12:00:00+00:00"
    assert status(tmp_path)["reason"] == "future_model"
    assert status(tmp_path, asof=date(2026, 10, 22))["available"] is True


def test_demo_publication_allows_explicit_dated_replay(tmp_path: Path, monkeypatch) -> None:
    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 22, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(model_artifacts, "datetime", LaterDatetime)
    published = approve(
        tmp_path, candidate(tmp_path, changes={"provider_kind": "demo"}), provider_kind="demo"
    )
    assert published["published_at"] == "2026-10-22T12:00:00+00:00"
    assert status(tmp_path, provider_kind="demo")["available"] is True


@pytest.mark.parametrize("recorded", [None, "invalid", "2026-10-20T12:00:00"])
def test_missing_or_invalid_actual_publication_time_is_rejected(tmp_path: Path, recorded) -> None:
    published = approve(tmp_path, candidate(tmp_path))
    manifest_path = Path(published["path"]).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["published_at"] = recorded
    content = json.dumps(manifest).encode("utf-8")
    manifest_path.write_bytes(content)
    pointer_path = tmp_path / "approved.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = artifact_digest(content)
    pointer_path.write_text(json.dumps(pointer))
    assert status(tmp_path)["reason"] == "invalid_artifact"



def test_live_approval_uses_shanghai_date_after_utc_midday(tmp_path, monkeypatch):
    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 22, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(model_artifacts, "datetime", LaterDatetime)
    approve(tmp_path, candidate(tmp_path))
    assert status(tmp_path, asof=date(2026, 10, 22))["reason"] == "future_model"
    assert status(tmp_path, asof=date(2026, 10, 23))["available"] is True
