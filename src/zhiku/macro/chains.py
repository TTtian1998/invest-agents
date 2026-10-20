"""地缘冲突与大宗商品传导链。默认 YAML 可被用户覆盖。"""

from __future__ import annotations

from pathlib import Path

import yaml

from zhiku.macro.models import TransmissionChain

PACKAGE_CHAINS = Path(__file__).resolve().parent / "chains.yaml"


def load_chains(path: Path | None = None) -> list[TransmissionChain]:
    data: dict = {}
    if PACKAGE_CHAINS.is_file():
        loaded = yaml.safe_load(PACKAGE_CHAINS.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    if path is not None and path.is_file():
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(override, dict) and override.get("chains"):
            data = override
    rows = data.get("chains") if isinstance(data, dict) else data
    return [TransmissionChain.model_validate(r) for r in (rows or [])]


def match_text(text: str, chains: list[TransmissionChain] | None = None) -> list[TransmissionChain]:
    blob = (text or "").lower()
    if not blob.strip():
        return []
    chains = chains if chains is not None else load_chains()
    hit: list[TransmissionChain] = []
    for chain in chains:
        for kw in chain.keywords:
            if kw and kw.lower() in blob:
                hit.append(chain)
                break
    return hit


def chain_summary(chain: TransmissionChain) -> dict:
    return {
        "id": chain.id,
        "name": chain.name,
        "commodities": chain.commodities,
        "a_share": chain.a_share,
        "h_share": chain.h_share,
        "us": chain.us,
        "etfs": chain.etfs,
        "risk_on_off": chain.risk_on_off,
        "hist_path_note": chain.hist_path_note,
    }
