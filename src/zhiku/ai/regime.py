"""市场状态：宏观锚点 + 近窗风格，建议更值得信哪类公式，事后检验。"""

from __future__ import annotations

from typing import Any

from zhiku.ai.backend import LLMBackend
from zhiku.ai.schema import RegimeAdvice

REGIME_SYS = (
    "任务=市场状态。根据所给宏观锚点和近窗收益差，建议当前风格（动量或反转、价值或成长）"
    "以及更值得观察的公式名。只输出 JSON：style, value_or_growth, prefer_screens, reason, cite。"
    "cite 必须是所给字段。这不是已发表的体制转换模型。"
)


def infer_style_flags(macro: dict[str, Any] | None, trailing: dict[str, Any] | None) -> dict[str, Any]:
    trail = trailing or {}
    mom = trail.get("universe_12_1")
    flags = {
        "geo": (macro or {}).get("geo") or {},
        "rate": (macro or {}).get("rate_expectation") or {},
        "universe_12_1": mom,
        "value_minus_growth": trail.get("value_minus_growth"),
    }
    if mom is not None:
        flags["mom_pos"] = bool(float(mom) > 0)
        flags["hint"] = "动量为正" if flags["mom_pos"] else "动量为负"
    return flags


def advise_regime(
    backend: LLMBackend,
    *,
    macro: dict[str, Any] | None = None,
    trailing: dict[str, Any] | None = None,
) -> RegimeAdvice:
    flags = infer_style_flags(macro, trailing)
    raw = backend.complete_json(REGIME_SYS, f"锚点与近窗={flags}")
    screens = raw.get("prefer_screens") or []
    if isinstance(screens, str):
        screens = [screens]
    return RegimeAdvice(
        style=str(raw.get("style") or "混合"),
        value_or_growth=str(raw.get("value_or_growth") or "缺数据"),
        prefer_screens=[str(s) for s in screens],
        reason=str(raw.get("reason") or ""),
        cite=str(raw.get("cite") or "缺数据"),
    )
