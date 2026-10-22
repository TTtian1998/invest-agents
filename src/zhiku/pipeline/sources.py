"""Explicit provenance for optional inputs; disabled never means demo data."""

from __future__ import annotations

from typing import Literal, cast

SourceMode = Literal["live", "demo", "disabled"]


def resolve_source_mode(provider_kind: str, *, enabled: bool = True) -> SourceMode:
    """Normalize the legacy fixture alias and honor an explicit disable switch.

    Unknown providers fail closed rather than accidentally selecting fixtures.
    The returned value describes the source actually used, not the requested one.
    """
    mode = "demo" if provider_kind == "fixture" else provider_kind
    if mode not in {"live", "demo", "disabled"}:
        raise ValueError(f"Unknown source mode: {provider_kind!r}; use live, demo, or disabled")
    return cast(SourceMode, mode) if enabled else "disabled"


def claim_source_mode(kb, provider_kind: str) -> SourceMode:
    """One knowledge base, one source mode: prevent mixed report/journal sinks.

    This marker is deliberately small. Migration uses a separate KB, not deletion
    of provenance or an automatic rewrite of existing live records.
    """
    from pathlib import Path
    mode = resolve_source_mode(provider_kind)
    root = Path(kb)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".zhiku-source-mode"
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(mode + "\n")
    except FileExistsError:
        existing = marker.read_text(encoding="utf-8").strip()
        if existing != mode:
            raise ValueError(f"知识库来源模式为 {existing!r}，不能以 {mode!r} 写入共享报告/日志。请使用独立 --kb / ZHIKU_KB；不要删除来源标记来混用历史。")
    return mode
