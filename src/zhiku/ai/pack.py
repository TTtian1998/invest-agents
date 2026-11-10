"""Evidence-bound review packs, immutable snapshots, and result validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from zhiku.ai.schema import (
    PACK_SCHEMA,
    RESULT_SCHEMA,
    Extraction,
    Judgment,
    RegimeAdvice,
    ReviewResult,
)
from zhiku.review.checklists import run_checklists
from zhiku.screens.fundamentals import Fundamentals

PACK_CONSTRAINTS = (
    "只使用本包里的候选、截面、清单和新闻摘录。不要编造包外数字。"
    "输出 zhiku.review_result.v1，并原样返回本包的 snapshot_id 和 asof。"
    "stance 只能是 研究跟进/观察/回避。每条主张必须有 cite，"
    "cite 应为本包字段名、字段路径或原文短句；缺数据写缺数据。"
    "不要输出本包以外的标的，不要改写公式分数或排名。详见 docs/ai_layer.md。"
)
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{64}$")


def review_snapshot_id(pack: dict[str, Any]) -> str:
    """Hash the entire visible evidence contract, independent of object key order."""
    content = {key: value for key, value in pack.items() if key != "snapshot_id"}
    canonical = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_review_pack(pack: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pack, dict) or pack.get("schema") != PACK_SCHEMA:
        raise ValueError(f"研判快照 schema 应为 {PACK_SCHEMA}")
    snapshot_id = pack.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("研判快照缺少有效 snapshot_id；请先运行 daily 生成快照")
    if snapshot_id != review_snapshot_id(pack):
        raise ValueError("研判快照内容与 snapshot_id 不匹配，不能使用被修改的证据")
    date.fromisoformat(str(pack.get("asof", "")))
    candidates = pack.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("研判快照 candidates 必须是列表")
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
            raise ValueError("研判快照候选格式无效")
        symbol = item["candidate"].get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in seen:
            raise ValueError("研判快照候选 symbol 为空或重复")
        seen.add(symbol)
        if not isinstance(item.get("snapshot"), dict) or item["snapshot"].get("symbol") != symbol:
            raise ValueError(f"{symbol} 候选与财务截面不匹配")
    return pack


def build_review_pack(
    *,
    asof: date,
    candidates: list[dict],
    rows: list[Fundamentals],
    masters: list[str],
    news: dict[str, Any] | None = None,
    macro: dict[str, Any] | None = None,
    trailing: dict[str, Any] | None = None,
    judge_n: int = 10,
) -> dict[str, Any]:
    by = {r.symbol: r for r in rows}
    packed_rows = []
    for i, cand in enumerate(candidates[:max(0, judge_n)], start=1):
        symbol = cand.get("symbol")
        row = by.get(symbol) if symbol is not None else None
        if row is None:
            continue
        cards, conflicts = run_checklists(row, masters)
        snap = row.model_dump(mode="json")
        # These are research fundamentals, never holdings/account information.
        snap.pop("fx_to_cny", None)
        packed_rows.append(
            {
                "rank": i,
                "candidate": {
                    "symbol": row.symbol,
                    "name": cand.get("name") or row.name,
                    "formula_score": cand.get("formula_score"),
                    "qlib_score": cand.get("qlib_score"),
                    "screens": cand.get("screens") or cand.get("screens_all") or {},
                },
                "snapshot": snap,
                "checklists": [
                    {
                        "master": c.master,
                        "stance": c.stance,
                        "school": c.school,
                        "items": [it.to_dict() for it in c.items],
                    }
                    for c in cards
                ],
                "disagreements": conflicts,
            }
        )
    excerpts = []
    for cluster in (news or {}).get("clusters") or (news or {}).get("items") or []:
        if not isinstance(cluster, dict):
            continue
        excerpts.append(
            {
                "symbol": cluster.get("symbol"),
                "headline": cluster.get("headline"),
                "summary": (cluster.get("summary") or cluster.get("raw_body") or "")[:500],
                "source": cluster.get("source") or cluster.get("source_kind"),
                "published": cluster.get("published"),
            }
        )
        if len(excerpts) >= 20:
            break
    # Detach nested inputs so later caller mutations cannot silently change the snapshot.
    pack = deepcopy({
        "schema": PACK_SCHEMA,
        "asof": asof.isoformat() if hasattr(asof, "isoformat") else str(asof),
        "constraints": PACK_CONSTRAINTS,
        "candidates": packed_rows,
        "news_excerpts": excerpts,
        "macro": {
            "geo": (macro or {}).get("geo"),
            "rate_expectation": (macro or {}).get("rate_expectation"),
            "window_banners": (macro or {}).get("window_banners"),
        },
        "trailing": trailing or {},
    })
    pack["snapshot_id"] = review_snapshot_id(pack)
    return validate_review_pack(pack)


def write_review_pack(path: Path, pack: dict[str, Any]) -> Path:
    """Atomically export an already validated snapshot without recomputing any inputs."""
    validate_review_pack(pack)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(pack, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def save_review_snapshot(reports: Path, pack: dict[str, Any]) -> Path:
    """Keep immutable evidence by ID; the date alias selects the latest daily snapshot."""
    validate_review_pack(pack)
    archive = reports / "review-snapshots" / f"{pack['snapshot_id']}.json"
    if archive.exists():
        saved = validate_review_pack(json.loads(archive.read_text(encoding="utf-8")))
        if saved != pack:
            raise ValueError("已有研判快照与当前证据不一致，拒绝覆盖")
    else:
        write_review_pack(archive, pack)
    write_review_pack(reports / f"{pack['asof']}-review-pack.json", pack)
    return archive


def load_review_snapshot(
    reports: Path, *, asof: date | str, snapshot_id: str | None = None
) -> dict[str, Any]:
    day = asof.isoformat() if isinstance(asof, date) else str(asof)
    date.fromisoformat(day)
    if snapshot_id is not None and not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("无效 snapshot_id")
    path = (
        reports / "review-snapshots" / f"{snapshot_id}.json"
        if snapshot_id is not None
        else reports / f"{day}-review-pack.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"没有 {day} 的已保存研判快照；请先运行 daily")
    pack = validate_review_pack(json.loads(path.read_text(encoding="utf-8")))
    if pack["asof"] != day or (snapshot_id is not None and pack["snapshot_id"] != snapshot_id):
        raise ValueError("研判快照日期或 snapshot_id 不匹配")
    archive = reports / "review-snapshots" / f"{pack['snapshot_id']}.json"
    if not archive.is_file():
        raise ValueError("研判快照缺少不可变存档；请重新运行 daily")
    archived = validate_review_pack(json.loads(archive.read_text(encoding="utf-8")))
    if archived != pack:
        raise ValueError("日期快照与不可变存档不一致")
    return pack


def review_result_from_obj(raw: dict[str, Any]) -> ReviewResult:
    if not isinstance(raw, dict):
        raise ValueError("研判结果必须是 JSON 对象")
    schema = raw.get("schema") or raw.get("schema_name")
    if schema != RESULT_SCHEMA:
        raise ValueError(f"schema 应为 {RESULT_SCHEMA}，实际 {schema}")
    payload = deepcopy(raw)
    for item in payload.get("judgments") or []:
        if isinstance(item, dict):
            if "long" in item and "long_case" not in item:
                item["long_case"] = item["long"]
            if "short" in item and "short_case" not in item:
                item["short_case"] = item["short"]
    result = ReviewResult.model_validate(payload)
    for judgment in result.judgments:
        if not judgment.long_case and not judgment.short_case:
            raise ValueError(f"{judgment.symbol} 缺少带 cite 的多空主张")
    return result


def load_review_result(path: Path) -> ReviewResult:
    return review_result_from_obj(json.loads(path.read_text(encoding="utf-8")))


def _citation_known(cite: str, evidence: Any) -> bool:
    """Check references, not the truth of free-form model reasoning."""
    cite = cite.strip()
    if not cite:
        return False
    if cite == "缺数据":
        return True
    references: set[str] = set()
    texts: list[str] = []

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                references.update((key, child))
                if key == "field" and isinstance(item, str):
                    references.update(item.split(","))
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and value:
            texts.append(value)

    walk(evidence)
    return cite in references or any(cite in text for text in texts)


def validate_review_result(result: ReviewResult, pack: dict[str, Any]) -> ReviewResult:
    """Bind a complete result to the exact candidates/evidence, never just the date."""
    validate_review_pack(pack)
    result = review_result_from_obj(result.to_dict())
    if not result.snapshot_id or result.snapshot_id != pack["snapshot_id"]:
        raise ValueError("研判结果 snapshot_id 与证据快照不匹配")
    if result.asof != pack["asof"]:
        raise ValueError("研判结果 asof 与证据快照不匹配")
    candidates = {item["candidate"]["symbol"]: item for item in pack["candidates"]}
    if candidates and not result.judgments:
        raise ValueError("快照包含候选，但研判结果没有有效候选判断")
    seen: set[str] = set()
    for judgment in result.judgments:
        if judgment.symbol not in candidates:
            raise ValueError(f"研判标的 {judgment.symbol} 不在快照候选中")
        if judgment.symbol in seen:
            raise ValueError(f"研判标的 {judgment.symbol} 重复")
        seen.add(judgment.symbol)
        item = candidates[judgment.symbol]
        candidate = item["candidate"]
        if judgment.formula_rank is not None and judgment.formula_rank != item["rank"]:
            raise ValueError(f"{judgment.symbol} formula_rank 与快照不匹配")
        if judgment.formula_score is not None and judgment.formula_score != candidate.get("formula_score"):
            raise ValueError(f"{judgment.symbol} formula_score 与快照不匹配")
        judgment.formula_rank = item["rank"]
        judgment.formula_score = candidate.get("formula_score")
        judgment.name = candidate.get("name") or ""
        evidence = {
            **item,
            "news_excerpts": [
                excerpt for excerpt in pack.get("news_excerpts", [])
                if not excerpt.get("symbol") or excerpt.get("symbol") == judgment.symbol
            ],
            "macro": pack.get("macro", {}),
            "trailing": pack.get("trailing", {}),
        }
        for claim in judgment.long_case + judgment.short_case:
            if not _citation_known(claim.cite, evidence):
                raise ValueError(f"{judgment.symbol} 引用不在快照证据中：{claim.cite}")
    for extraction in result.extractions:
        excerpts = [
            item for item in pack.get("news_excerpts", [])
            if not extraction.symbol or not item.get("symbol") or item.get("symbol") == extraction.symbol
        ]
        if extraction.symbol and extraction.symbol not in candidates and not any(
            item.get("symbol") == extraction.symbol for item in excerpts
        ):
            raise ValueError(f"抽取标的 {extraction.symbol} 不在快照证据中")
        if not _citation_known(extraction.cite, excerpts):
            raise ValueError(f"文本抽取引用不在快照中：{extraction.cite}")
    if result.regime is not None and not _citation_known(
        result.regime.cite, {"macro": pack.get("macro", {}), "trailing": pack.get("trailing", {})}
    ):
        raise ValueError(f"市场状态引用不在快照中：{result.regime.cite}")
    missing = sorted(set(candidates) - seen)
    if missing:
        note = f"本快照未完成研判的候选：{', '.join(missing)}"
        if note not in result.notes:
            result.notes.append(note)
    return result


def result_from_parts(
    *,
    asof: str,
    backend: str,
    extractions: list[Extraction],
    judgments: list[Judgment],
    regime: RegimeAdvice | None,
    notes: list[str] | None = None,
    snapshot_id: str = "",
) -> ReviewResult:
    return ReviewResult(
        asof=asof,
        snapshot_id=snapshot_id,
        backend=backend,
        extractions=extractions,
        judgments=judgments,
        regime=regime,
        notes=notes or [],
    )
