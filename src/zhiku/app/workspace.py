"""Read-only report adapter. No pipeline, network, account or broker operations.

Only named daily reports are readable. The public projection deliberately omits
review-pack paths, private holdings, arbitrary report fields and raw exceptions.
This local application has no authentication; never expose it directly online.
"""
from __future__ import annotations

import json
import os
import re
import stat
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from zhiku.calendar import now_shanghai
from zhiku.config import Paths

MAX_REPORT_BYTES = 8 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"
SCREENS = frozenset({"greenblatt", "piotroski", "dividend", "multifactor", "graham", "lynch_peg", "canslim", "livermore", "fisher"})
Mode = Literal["live", "demo", "disabled"]
SourceMode = Literal["live", "demo", "disabled", "unknown"]
DISCLAIMER = "仅供研究，不构成投资建议。不接券商、不下单；页面只读取已生成报告。"


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


class Candidate(PublicModel):
    symbol: str = Field(min_length=1, max_length=40)
    name: str = Field(default="", max_length=200)
    formula_score: float | None = None
    qlib_score: float | None = None
    total: float | None = None
    screens: dict[str, float] = Field(default_factory=dict)
    source: str = "unknown"

    @field_validator("screens")
    @classmethod
    def known_screens(cls, value: dict[str, float]) -> dict[str, float]:
        return {key: score for key, score in value.items() if key in SCREENS}


class Observation(Candidate):
    eligible: bool = False
    status: Literal["research_only"]


class Gate(PublicModel):
    status: str = "unavailable"
    reason: str = "not_published"
    approved_screens: list[str] = Field(default_factory=list)
    statuses: dict[str, Literal["passed", "failed", "insufficient"]] = Field(default_factory=dict)
    version: str | None = None
    evaluated_asof: str | None = None


class ModelStatus(PublicModel):
    status: str = "unknown"
    reason: str = "unavailable"
    available: bool = False
    inference_available: bool = False
    inference_reason: str | None = None
    publication_status: str | None = None
    validation_scope: str | None = None
    implementation: str | None = None
    kind: str | None = None
    rank_ic: float | None = None
    trained_on: str | None = None


class NewsSource(PublicModel):
    source: str = "unknown"
    headline: str = ""
    url: str = ""
    grade: int | None = None
    grade_label: str = ""


class NewsCard(PublicModel):
    fingerprint: str = ""
    symbol: str | None = None
    name: str = ""
    headline: str
    catalyst: str = ""
    tags: list[str] = Field(default_factory=list)
    freshness: str = "unknown"
    reaction: str | None = None
    source_grade: int | None = None
    source_grade_label: str = ""
    first_seen: str | None = None
    stale: bool | None = None
    sources: list[NewsSource] = Field(default_factory=list)


class Emergency(PublicModel):
    active: bool | None = None
    freeze_adds: bool | None = None
    reasons: list[str] = Field(default_factory=list)


class SourceError(PublicModel):
    source: str
    reason: str


class News(PublicModel):
    asof: str | None = None
    source_mode: SourceMode | None = None
    analysis_engine: str = "unknown"
    clusters: list[NewsCard] = Field(default_factory=list)
    source_errors: list[str] = Field(default_factory=list)
    emergency: Emergency = Field(default_factory=Emergency)

    @field_validator("source_errors", mode="before")
    @classmethod
    def normalize_source_errors(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        # NewsBundle persists typed {source, reason} errors. Keep the public
        # string-list contract while excluding arbitrary connector details.
        errors: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                errors.append(entry)
            else:
                error = SourceError.model_validate(entry)
                errors.append(f"{error.source}: {error.reason}")
        return errors


class MacroEvent(PublicModel):
    id: str = ""
    event_type: str = ""
    name: str
    region: str = ""
    importance: int | None = None
    scheduled_at: str | None = None
    scheduled_shanghai: str | None = None
    previous: float | None = None
    consensus: float | None = None
    actual: float | None = None
    unit: str = ""
    surprise_z: float | None = None
    surprise_missing: str | None = None


class Asset(PublicModel):
    asset: str
    label: str = ""
    price: float | None = None
    chg: float | None = None
    missing: str | None = None


class Macro(PublicModel):
    asof: str | None = None
    source_mode: SourceMode | None = None
    calendar_7d: list[MacroEvent] = Field(default_factory=list)
    anchors: list[Asset] = Field(default_factory=list)


class ScreenEvidence(PublicModel):
    symbol: str
    name: str = ""
    score: float
    breakdown: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("breakdown")
    @classmethod
    def public_metrics(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        private = re.compile(r"(?i)(secret|token|password|account|holding|position|cost|balance|quantity)")
        return {key: metric for key, metric in value.items()
                if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", key) and not private.search(key)}


class Opinion(PublicModel):
    master: str
    stance: str
    rationale: str
    skipped: bool = False


class Audit(PublicModel):
    symbol: str
    name: str = ""
    conclusion: str
    skipped: bool = False
    skip_reason: str | None = None
    opinions: list[Opinion] = Field(default_factory=list)


class DailyReport(PublicModel):
    asof: str
    source_mode: Mode
    fictional: bool
    snapshot_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    # Required boundary fields: legacy mixed pools must not acquire new meaning.
    candidates: list[Candidate]
    observations: list[Observation]
    screens: dict[str, list[ScreenEvidence]] = Field(default_factory=dict)
    audits: list[Audit] = Field(default_factory=list)
    gate_artifact: Gate
    qlib: ModelStatus = Field(default_factory=ModelStatus)
    universe_size: int = Field(default=0, ge=0)
    quality_passed: list[str] = Field(default_factory=list)
    missing_ratio: float | None = Field(default=None, ge=0, le=1)
    news: News = Field(default_factory=News)
    macro: Macro = Field(default_factory=Macro)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def no_mixed_candidates(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("candidates"), list):
            for row in value["candidates"]:
                if isinstance(row, dict) and (row.get("status") == "research_only" or row.get("eligible") is False):
                    raise ValueError("Research observation in formal pool")
        return value

    @field_validator("asof")
    @classmethod
    def valid_date(cls, value: str) -> str:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("Invalid date")
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> DailyReport:
        if self.fictional != (self.source_mode != "live"):
            raise ValueError("Source provenance mismatch")
        for source in (self.news, self.macro):
            if source.asof is not None:
                date.fromisoformat(source.asof)
                if source.asof != self.asof:
                    raise ValueError("Source date mismatch")
            if source.source_mode not in {None, "unknown", "disabled", self.source_mode}:
                raise ValueError("Source mode mismatch")
        gate = self.gate_artifact
        if gate.status == "approved" or gate.approved_screens:
            if (gate.status != "approved" or gate.reason != "validated" or not gate.version
                    or not re.fullmatch(r"[a-f0-9]{64}", gate.version)
                    or not gate.evaluated_asof or gate.evaluated_asof > self.asof
                    or not gate.statuses):
                raise ValueError("Incomplete approval provenance")
            date.fromisoformat(gate.evaluated_asof)
        if any(s not in SCREENS for s in gate.approved_screens):
            raise ValueError("Unknown approval")
        if any(gate.statuses.get(screen) != "passed" for screen in gate.approved_screens):
            raise ValueError("Approved screen has no passed evidence")
        if self.candidates:
            if (self.source_mode == "disabled" or gate.status != "approved" or gate.reason != "validated"
                    or not gate.approved_screens or not gate.version
                    or not re.fullmatch(r"[a-f0-9]{64}", gate.version)
                    or not gate.evaluated_asof or gate.evaluated_asof > self.asof):
                raise ValueError("Formal pool has no acceptance provenance")
            date.fromisoformat(gate.evaluated_asof)
            for screen in gate.approved_screens:
                if gate.statuses.get(screen) != "passed":
                    raise ValueError("Formal pool has no passed screen")
            for candidate in self.candidates:
                if not candidate.screens or not set(candidate.screens) <= set(gate.approved_screens):
                    raise ValueError("Formal pool contains unapproved screen")
                if candidate.formula_score is None or candidate.total is None:
                    raise ValueError("Formal pool is not formula led")
        return self


class ReportUnavailable(Exception):
    def __init__(self, state: str, issue: str):
        self.state = state
        self.issue = issue


def _read_beneath(root: Path, parts: tuple[str, ...], max_bytes: int = MAX_REPORT_BYTES) -> tuple[bytes, float]:
    """Open fixed path components with dir_fd and O_NOFOLLOW (including root).

    No symlink follows or check/open race, no pipes/devices, and bounded reads.
    All paths come from our registry, never directly from the request.
    """
    descriptors: list[int] = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(fd)
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        descriptors.append(fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ReportUnavailable("blocked", "报告不是普通文件或超过大小限制。")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > max_bytes:
            raise ReportUnavailable("blocked", "报告超过大小限制。")
        return content, info.st_mtime
    except FileNotFoundError as exc:
        raise ReportUnavailable("missing", "尚未找到已生成的日报；页面不会自动运行流水线或加载样例。") from exc
    except OSError as exc:
        raise ReportUnavailable("blocked", "报告不可安全读取；请检查本地文件类型与访问权限。") from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _registry(paths: Paths) -> dict[str, tuple[str, ...]]:
    registry: dict[str, tuple[str, ...]] = {"latest": ("reports", "latest-daily.json")}
    # Names are only a catalog; secure open independently checks every component.
    root = paths.kb / "reports"
    if root.is_symlink() or not root.is_dir():
        return registry
    try:
        for folder in sorted(root.iterdir(), reverse=True):
            if not re.fullmatch(r"\d{4}-\d{2}", folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            for entry in sorted(folder.iterdir(), reverse=True):
                match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-zk-daily\.json", entry.name)
                if match and not entry.is_symlink() and entry.is_file():
                    day = match.group(1)
                    if day[:7] == folder.name:
                        try:
                            date.fromisoformat(day)
                        except ValueError:
                            continue
                        registry[day] = ("reports", folder.name, entry.name)
                if len(registry) >= 367:
                    return registry
    except OSError:
        pass
    return registry


def _reject_constants(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _load_report(paths: Paths, parts: tuple[str, ...]) -> tuple[DailyReport, float]:
    data, modified = _read_beneath(paths.kb, parts)
    try:
        raw = json.loads(data, parse_constant=_reject_constants, object_pairs_hook=_object_without_duplicates)
        report = DailyReport.model_validate(raw)
    except (ValueError, TypeError, RecursionError, ValidationError) as exc:
        raise ReportUnavailable("invalid", "日报格式或来源校验未通过；旧版混合候选不会被自动视为正式候选。") from exc
    try:
        marker, _ = _read_beneath(paths.kb, (".zhiku-source-mode",), max_bytes=32)
    except ReportUnavailable as exc:
        if exc.state != "missing":
            raise
    else:
        try:
            mode = marker.decode("utf-8").strip()
        except UnicodeError as exc:
            raise ReportUnavailable("blocked", "知识库来源标记无效。") from exc
        if mode != report.source_mode:
            raise ReportUnavailable("blocked", "日报来源与知识库隔离模式不一致，已停止展示。")
    if len(parts) == 3 and report.asof != parts[-1][:10]:
        raise ReportUnavailable("invalid", "日报日期与所选历史快照不一致。")
    return report, modified


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return ""
        # Query/fragment can contain signed tokens or account identifiers.
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return ""


def _public_text(value: str) -> str:
    # HTTP errors often embed the failing URL in free text, not a `url` field.
    # Sanitize before token/path redaction so userinfo and signed queries cannot
    # leak through notes or the normalized SourceError reason.
    value = re.sub(r"(?i)https?://[^\s<>\"'，；。]+",
                   lambda match: _safe_url(match.group()) or "[已隐藏来源地址]", value)
    value = re.sub(r"(?i)\b(?:bearer\s+\S+|sk-[a-zA-Z0-9_-]{10,}|(?:[a-z0-9]+_)*(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+)", "[已隐藏敏感信息]", value)
    value = re.sub(r"(?<![:/\w])(?:/[a-zA-Z._-][^\s；，。]*|[A-Za-z]:\\[^\s；，。]+)", "[本地路径]", value)
    return value


def _public(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _public(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_public(item, key) for item in value]
    if isinstance(value, str):
        return _safe_url(value) if key == "url" else _public_text(value)
    return value


def _empty(snapshot: str, registry: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "service": {"read_only": True, "network_actions": False, "mode": "unknown", "demo": False, "disclaimer": DISCLAIMER},
        "report": {"state": "missing", "asof": None, "snapshot_id": None, "freshness": "unknown", "age_days": None, "updated_at": None, "filename": registry[snapshot][-1], "historical": snapshot != "latest", "issues": []},
        "snapshots": [{"id": key, "asof": None if key == "latest" else key, "label": "最新报告" if key == "latest" else key} for key in registry],
        "selected_snapshot": snapshot,
        "summary": {"universe_size": 0, "quality_passed": 0, "formal_candidates": 0, "research_observations": 0, "missing_ratio": None},
        "candidates": [], "research_observations": [], "screen_evidence": {}, "audits": [],
        "gates": Gate().model_dump(), "model": ModelStatus().model_dump(),
        "sources": [{"id": source, "mode": "unknown", "status": "unavailable", "asof": None, "detail": "未读取到有效报告；没有检测实时连接。"} for source in ("daily", "news", "macro")],
        "news": News().model_dump(),
        "macro": {"asof": None, "source_mode": None, "calendar_7d": [], "assets": []},
        "notes": [],
    }


def workspace_payload(paths: Paths, snapshot: str = "latest") -> dict[str, Any]:
    registry = _registry(paths)
    if snapshot not in registry:
        raise HTTPException(status_code=404, detail="未找到所选报告快照。")
    payload = _empty(snapshot, registry)
    try:
        daily, modified = _load_report(paths, registry[snapshot])
    except ReportUnavailable as exc:
        payload["report"].update(state=exc.state, issues=[exc.issue])
        return payload
    age = (now_shanghai().date() - date.fromisoformat(daily.asof)).days
    freshness = "current" if age == 0 else "stale" if age > 0 else "future"
    payload["service"].update(mode=daily.source_mode, demo=daily.source_mode == "demo")
    payload["report"].update(state="ready", asof=daily.asof, snapshot_id=daily.snapshot_id,
        freshness=freshness, age_days=age, updated_at=datetime.fromtimestamp(modified, timezone.utc).isoformat())
    if age != 0:
        payload["report"]["issues"].append("当前展示历史或未来截面，不代表今日可用信号。")
    if daily.source_mode == "demo":
        payload["report"]["issues"].append("演示模式：包含虚构样例，非真实市场状态。")
    payload["snapshots"][0]["asof"] = daily.asof if snapshot == "latest" else None
    payload["summary"].update(universe_size=daily.universe_size, quality_passed=len(daily.quality_passed),
        formal_candidates=len(daily.candidates), research_observations=len(daily.observations), missing_ratio=daily.missing_ratio)
    payload["candidates"] = [c.model_dump() for c in daily.candidates]
    payload["research_observations"] = [o.model_dump(exclude={"total"}) for o in daily.observations]
    payload["screen_evidence"] = {name: [row.model_dump() for row in rows] for name, rows in daily.screens.items() if name in SCREENS}
    payload["audits"] = [audit.model_dump() for audit in daily.audits]
    payload["gates"] = daily.gate_artifact.model_dump()
    payload["model"] = daily.qlib.model_dump()
    payload["model"]["implementation"] = daily.qlib.implementation or daily.qlib.kind
    if daily.qlib.inference_reason:
        payload["model"]["reason"] = daily.qlib.inference_reason
    elif daily.qlib.available and "reason" not in daily.qlib.model_fields_set:
        payload["model"]["reason"] = "published_artifact"
    payload["news"] = daily.news.model_dump()
    payload["macro"] = daily.macro.model_dump(exclude={"anchors"})
    payload["macro"]["assets"] = [asset.model_dump() for asset in daily.macro.anchors]
    mode: SourceMode | None
    asof: str | None
    for source in payload["sources"]:
        source_id = source["id"]
        if source_id == "daily":
            mode, asof, present, errors = daily.source_mode, daily.asof, True, False
        elif source_id == "news":
            mode, asof = daily.news.source_mode, daily.news.asof
            present, errors = bool(daily.news.clusters), bool(daily.news.source_errors)
        else:
            mode, asof = daily.macro.source_mode, daily.macro.asof
            present, errors = bool(daily.macro.anchors or daily.macro.calendar_7d), False
        status = "disabled" if mode == "disabled" else "unknown" if mode in {None, "unknown"} else "partial" if errors else "missing" if not present else "demo" if mode == "demo" else "snapshot"
        source.update(mode=mode or "unknown", asof=asof, status=status, detail="已保存报告的来源声明，非实时连接健康检查。")
    payload["notes"] = daily.notes
    return _public(payload)


def create_app(paths: Paths | None = None, *, allowed_hosts: Sequence[str] | None = None) -> FastAPI:
    paths = paths or Paths.load()
    app = FastAPI(title="zhiku research workspace", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts or ["localhost", "127.0.0.1", "[::1]"]))

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        # Same-origin, read-only endpoints. No permissive CORS or remote scripts.
        if request.method not in {"GET", "HEAD"}:
            response: Response = JSONResponse({"detail": "只读工作台不接受写入请求。"}, status_code=405)
        elif request.headers.get("sec-fetch-site") == "cross-site":
            response = JSONResponse({"detail": "不允许跨站访问本地工作台。"}, status_code=403)
        else:
            response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'",
        })
        return response

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "zhiku-workspace", "read_only": True, "scope": "process_only", "market_data_checked": False}

    @app.get("/api/workspace")
    def workspace(snapshot: str = Query(default="latest", max_length=10)) -> dict[str, Any]:
        return workspace_payload(paths, snapshot)

    @app.get("/api/export")
    def export(snapshot: str = Query(default="latest", max_length=10)) -> Response:
        payload = workspace_payload(paths, snapshot)
        if payload["report"]["state"] != "ready":
            raise HTTPException(status_code=409, detail="没有通过校验的日报可导出。")
        # Same allowlisted projection as UI, never FileResponse on raw reports.
        filename = f"zhiku-{payload['service']['mode']}-{payload['report']['asof']}-workspace.json"
        return Response(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/")
    def index() -> Response:
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return JSONResponse({"detail": "前端静态资源尚未安装。"}, status_code=503)
        return FileResponse(page)

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False, follow_symlink=False), name="static")
    return app
