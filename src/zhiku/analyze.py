"""投研分析层：LLM 可选；无 key 时退化为规则模板，并标「未做深度分析」。

报告以分析为主体，原文只作为附录链接。不编造数字，不输出持仓数量/成本。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ClaimKind = Literal["事实", "解读", "推测"]
ANALYSIS_FIELDS = ("so_what", "priced_in", "impact", "counter_view", "next_check", "confidence")
DEGRADED_MARK = "未做深度分析"


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    kind: ClaimKind


class AnalysisCard(BaseModel):
    """入选新闻/宏观事件的结构化分析。低价值条目不生成卡片。"""

    model_config = ConfigDict(extra="ignore")

    subject: str
    important: bool = True
    so_what: str
    priced_in: str
    impact: str
    counter_view: str
    next_check: str
    confidence: float
    claims: list[Claim] = Field(default_factory=list)
    deep: bool = False
    engine: str = "rules"  # llm | rules
    mark: str = ""
    sources: list[dict[str, str]] = Field(default_factory=list)
    symbol: str | None = None
    catalyst: str = ""
    source_grade: int | None = None
    source_grade_label: str = ""
    institution_review: dict | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip_conf(cls, value: Any) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, num))

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        for name in ANALYSIS_FIELDS:
            val = getattr(self, name)
            if val is None or val == "":
                missing.append(name)
        if not self.claims:
            missing.append("claims")
        kinds = {c.kind for c in self.claims}
        for kind in ("事实", "解读", "推测"):
            if kind not in kinds:
                missing.append(f"claims.{kind}")
        return missing


def llm_key() -> str | None:
    return os.environ.get("ZHIKU_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or None


def llm_enabled() -> bool:
    if os.environ.get("ZHIKU_ANALYSIS_LLM", "1").lower() in {"0", "false", "no"}:
        return False
    return bool(llm_key())


def call_llm_json(prompt: str) -> dict | None:
    key = llm_key()
    if not key or not llm_enabled():
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    kwargs: dict[str, Any] = {"api_key": key}
    base = os.environ.get("ZHIKU_LLM_BASE_URL")
    if base:
        kwargs["base_url"] = base
    try:
        client = OpenAI(**kwargs)
        model = os.environ.get("ZHIKU_LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是投研分析助手。只输出一个 JSON 对象，不要编造未提供的数字。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 — 分析失败必须降级，不能打断雷达
        return None
    return _parse_json_object(text)


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def card_from_llm(payload: dict, *, fallback: AnalysisCard) -> AnalysisCard | None:
    try:
        claims_raw = payload.get("claims") or []
        claims = [Claim.model_validate(c) for c in claims_raw]
        card = AnalysisCard(
            subject=str(payload.get("subject") or fallback.subject),
            important=True,
            so_what=str(payload.get("so_what") or ""),
            priced_in=str(payload.get("priced_in") or ""),
            impact=str(payload.get("impact") or ""),
            counter_view=str(payload.get("counter_view") or ""),
            next_check=str(payload.get("next_check") or ""),
            confidence=payload.get("confidence", fallback.confidence),
            claims=claims,
            deep=True,
            engine="llm",
            mark="",
            sources=fallback.sources,
            symbol=fallback.symbol,
            catalyst=fallback.catalyst,
        )
    except Exception:  # noqa: BLE001
        return None
    if card.missing_fields():
        # LLM 缺字段时用规则模板补齐，但不再标「深度」
        merged = fallback.model_copy(
            update={k: getattr(card, k) or getattr(fallback, k) for k in ANALYSIS_FIELDS}
        )
        if card.claims:
            merged.claims = card.claims
        merged.engine = "llm+rules"
        merged.deep = False
        merged.mark = DEGRADED_MARK
        return merged
    return card


def ensure_claim_kinds(card: AnalysisCard) -> AnalysisCard:
    kinds = {c.kind for c in card.claims}
    if "事实" not in kinds:
        card.claims.append(Claim(text="以可核对的公开标题/日历为准，未引入未提供的数字。", kind="事实"))
    if "解读" not in kinds:
        card.claims.append(Claim(text=card.so_what[:120] or "结合价量与催化类型做规则解读。", kind="解读"))
    if "推测" not in kinds:
        card.claims.append(Claim(text=card.counter_view[:120] or "反方路径仍可能成立，需下一验证节点确认。", kind="推测"))
    return card
