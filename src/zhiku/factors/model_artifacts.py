"""Explicit promotion of trusted local research models into immutable publications.

Pickle is not a safe interchange format: only publish artifacts produced by a
trusted local training job. Hashes detect accidental replacement, not a malicious
actor who can rewrite both the model and its publication pointer.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
APPROVED_MANIFEST = "approved.json"
SUPPORTED_KINDS = frozenset({"lgb_lite", "lgb_panel", "zscore_panel"})
DEFAULT_MAX_AGE_DAYS = 45


class ModelArtifactError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ModelArtifact:
    path: Path
    metadata: dict[str, Any]
    payload: dict[str, Any]


def provider_identity(provider_kind: str) -> str:
    return "demo" if provider_kind in {"demo", "fixture"} else provider_kind


def artifact_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    """Replace a single file only after its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def validate_payload(payload: Any, meta: dict[str, Any], *, probe: bool = False) -> None:
    """Validate identity, training evidence, and the supported inference contract."""
    if not isinstance(payload, dict):
        raise ModelArtifactError("invalid_artifact", "模型必须包含显式 kind/model/features")
    kind = payload.get("kind")
    if kind not in SUPPORTED_KINDS:
        raise ModelArtifactError(
            "unsupported_kind", f"模型类型 {kind!r} 无对应推理实现，禁止替用其他模型"
        )
    if meta.get("kind") != kind:
        raise ModelArtifactError("identity_mismatch", "元数据 kind 与模型 payload 不一致")
    if meta.get("status") != "trained":
        raise ModelArtifactError("invalid_artifact", "训练结果未成功，不能发布或预测")
    n_train = meta.get("n_train")
    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train <= 0:
        raise ModelArtifactError("empty_training", "缺少非空训练样本证据 n_train")
    features = payload.get("features")
    if (
        not isinstance(features, (list, tuple))
        or not features
        or any(not isinstance(f, str) or not f for f in features)
        or len(set(features)) != len(features)
    ):
        raise ModelArtifactError("invalid_artifact", "模型特征列表为空或无效")
    if meta.get("features") != list(features):
        raise ModelArtifactError("identity_mismatch", "元数据特征与模型特征不一致")
    cfg = payload.get("config") or {}
    if not isinstance(cfg, dict) or _json_bytes(meta.get("config") or {}) != _json_bytes(cfg):
        raise ModelArtifactError("identity_mismatch", "元数据训练配置与模型配置不一致")
    if kind == "zscore_panel":
        if cfg.get("model") != "zscore" or payload.get("model") is not None:
            raise ModelArtifactError("identity_mismatch", "zscore 模型身份与配置不一致")
    else:
        model: Any = payload.get("model")
        if not callable(getattr(model, "predict", None)) or cfg.get("model") == "zscore":
            raise ModelArtifactError("invalid_artifact", "模型缺少对应的可用 predict 实现")
        if probe:
            try:
                # A bounded deterministic probe catches empty/unfitted estimators,
                # feature-count mismatches and nonfinite output before promotion.
                sample = pd.DataFrame(
                    np.arange(3 * len(features), dtype=float).reshape(3, len(features)) / 10,
                    columns=features,
                )
                pred = np.asarray(model.predict(sample), dtype=float)
                if pred.shape != (3,) or not np.isfinite(pred).all():
                    raise ValueError("预测为空、非有限值或形状错误")
                dump_model = getattr(model, "dump_model", None)
                if callable(dump_model):
                    trees = dump_model().get("tree_info")
                    if not trees or not _finite_parameters(trees):
                        raise ValueError("模型树参数为空或包含非有限值")
                for method in ("num_trees", "current_iteration"):
                    count = getattr(model, method, None)
                    if callable(count) and count() <= 0:
                        raise ValueError("模型没有有效训练参数")
            except Exception as exc:
                raise ModelArtifactError("invalid_artifact", f"模型预测验证失败：{exc}") from exc


def _finite_parameters(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_parameters(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_parameters(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _decode_artifact(
    path: Path, content: bytes, meta: dict[str, Any], *, probe: bool
) -> ModelArtifact:
    digest = meta.get("artifact_sha256")
    if not digest or digest != artifact_digest(content):
        raise ModelArtifactError("identity_mismatch", "模型内容与元数据 SHA-256 不一致或缺失")
    try:
        payload = pickle.loads(content)
    except Exception as exc:
        raise ModelArtifactError("invalid_artifact", f"模型无法加载：{exc}") from exc
    validate_payload(payload, meta, probe=probe)
    return ModelArtifact(path=path, metadata=meta, payload=payload)


def load_research_model(model_path: Path, *, probe: bool = False) -> ModelArtifact:
    """An explicit path is a research-only opt-in, never implicit live selection."""
    path = Path(model_path)
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("元数据必须为对象")
        return _decode_artifact(path, path.read_bytes(), meta, probe=probe)
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError("invalid_artifact", f"无法读取研究模型：{exc}") from exc


def _check_context(
    meta: dict[str, Any],
    *,
    asof: date,
    provider_kind: str,
    config_fingerprint: str | None,
    max_age_days: int,
) -> None:
    if not config_fingerprint or not config_fingerprint.strip():
        raise ModelArtifactError("config_mismatch", "正式推理缺少配置指纹")
    if meta.get("config_fingerprint") != config_fingerprint:
        raise ModelArtifactError("config_mismatch", "模型发布配置与当前配置不一致")
    if meta.get("provider_kind") != provider_identity(provider_kind):
        raise ModelArtifactError("provider_mismatch", "模型发布数据源与当前数据源不一致")
    if max_age_days < 0:
        raise ModelArtifactError("invalid_context", "模型最长有效期不能为负数")
    for field in ("trained_on", "data_asof", "published_asof"):
        try:
            day = date.fromisoformat(meta[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelArtifactError("invalid_artifact", f"模型缺少有效日期 {field}") from exc
        if day > asof:
            raise ModelArtifactError("future_model", f"模型 {field}={day} 晚于报告日期 {asof}")
        if field != "published_asof" and (asof - day).days > max_age_days:
            raise ModelArtifactError(
                "stale_model", f"模型 {field}={day} 已超过 {max_age_days} 天有效期"
            )


def _check_publication_time(meta: dict[str, Any], *, asof: date) -> None:
    try:
        recorded = datetime.fromisoformat(meta["published_at"])
        if recorded.tzinfo is None or recorded.utcoffset() is None:
            raise ValueError("发布时间必须包含时区")
        from zhiku.calendar import SHANGHAI
        actual_day = recorded.astimezone(SHANGHAI).date()
        declared_day = date.fromisoformat(meta["published_asof"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError(
            "invalid_artifact", "模型缺少有效的实际发布时间 published_at"
        ) from exc
    # Live approval cannot be made available before it really happened. Demo
    # publications retain declared dates for explicitly isolated replay fixtures.
    effective_day = (
        max(declared_day, actual_day) if meta["provider_kind"] == "live" else declared_day
    )
    if effective_day > asof:
        raise ModelArtifactError(
            "future_model", f"模型实际生效日期 {effective_day} 晚于报告日期 {asof}"
        )


def publish_model(
    models_dir: Path,
    model_path: Path,
    *,
    asof: date,
    provider_kind: str = "live",
    config_fingerprint: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> dict[str, Any]:
    """Validate then explicitly approve a new immutable version.

    This checks artifact integrity and supported finite inference, not predictive
    efficacy or an out-of-sample performance threshold. Training and candidate
    selection never call this function. Failure at any
    validation/write step before pointer replacement leaves the old approval in
    place. The source candidate can subsequently be overwritten without changing
    the published model.
    """
    artifact = load_research_model(model_path, probe=True)
    provider_kind = provider_identity(provider_kind)
    if provider_kind not in {"live", "demo"}:
        raise ModelArtifactError("provider_mismatch", "发布数据源必须为 live 或 demo")
    # Context already recorded by a producer must never be silently relabelled.
    for field, expected in (
        ("provider_kind", provider_kind),
        ("config_fingerprint", config_fingerprint),
    ):
        recorded = artifact.metadata.get(field)
        if not recorded or recorded != expected:
            raise ModelArtifactError("identity_mismatch", f"研究模型 {field} 与请求发布身份不一致")
    meta = dict(artifact.metadata)
    version = uuid4().hex
    directory = Path(models_dir) / "published" / version
    meta.update(
        schema_version=SCHEMA_VERSION,
        version=version,
        provider_kind=provider_kind,
        config_fingerprint=config_fingerprint,
        published_asof=asof.isoformat(),
        published_at=datetime.now(timezone.utc).isoformat(),
        publication_status="approved",
        validation_scope="artifact_integrity_and_inference_only",
    )
    _check_context(
        meta,
        asof=asof,
        provider_kind=provider_kind,
        config_fingerprint=config_fingerprint,
        max_age_days=max_age_days,
    )
    # Read once and verify again so a concurrent research write cannot substitute
    # different bytes after validation. No pickle reload is needed here.
    content = artifact.path.read_bytes()
    if artifact_digest(content) != meta["artifact_sha256"]:
        raise ModelArtifactError("identity_mismatch", "研究模型在发布验证后发生变化")
    meta["path"] = str(directory / "model.pkl")
    try:
        manifest = _json_bytes(meta)
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("invalid_artifact", f"模型元数据无法持久化：{exc}") from exc
    directory.mkdir(parents=True, exist_ok=False)
    # Unique version paths are write-once. Only approved.json is replaced.
    (directory / "model.pkl").write_bytes(content)
    (directory / "manifest.json").write_bytes(manifest)
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "manifest_sha256": artifact_digest(manifest),
    }
    atomic_write(Path(models_dir) / APPROVED_MANIFEST, _json_bytes(pointer))
    return {**meta, "available": True}


def resolve_published_model(
    models_dir: Path,
    *,
    asof: date,
    provider_kind: str = "live",
    config_fingerprint: str | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> ModelArtifact:
    """Resolve exactly the approved identity; never inspect candidate filenames."""
    pointer_path = Path(models_dir) / APPROVED_MANIFEST
    if not pointer_path.is_file():
        raise ModelArtifactError("not_published", "尚未显式发布可用模型")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        version = pointer["version"]
        if pointer.get("schema_version") != SCHEMA_VERSION or not re.fullmatch(
            r"[0-9a-f]{32}", version
        ):
            raise ValueError("发布指针版本无效")
        directory = Path(models_dir) / "published" / version
        manifest = (directory / "manifest.json").read_bytes()
        if artifact_digest(manifest) != pointer["manifest_sha256"]:
            raise ModelArtifactError("identity_mismatch", "发布清单内容已改变")
        meta = json.loads(manifest)
        if meta.get("version") != version or meta.get("schema_version") != SCHEMA_VERSION:
            raise ModelArtifactError("identity_mismatch", "发布清单版本不一致")
        if meta.get("publication_status") != "approved":
            raise ModelArtifactError("invalid_artifact", "模型没有有效发布状态")
        _check_context(
            meta,
            asof=asof,
            provider_kind=provider_kind,
            config_fingerprint=config_fingerprint,
            max_age_days=max_age_days,
        )
        _check_publication_time(meta, asof=asof)
        path = directory / "model.pkl"
        meta["path"] = str(path)
        artifact = _decode_artifact(path, path.read_bytes(), meta, probe=False)
        artifact.metadata["available"] = True
        return artifact
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError("invalid_artifact", f"无法读取发布模型：{exc}") from exc
