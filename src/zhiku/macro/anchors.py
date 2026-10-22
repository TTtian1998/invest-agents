"""锚定资产一览。缺价格写缺数据。"""

from __future__ import annotations

from zhiku.macro.models import ANCHOR_ASSETS, AnchorSnapshot


def snapshots_from_live(raw: dict | None) -> list[AnchorSnapshot]:
    """live 价量。缺的资产写缺数据，不拿 fixture 补。"""
    return snapshots_from_raw({"assets": raw or {}})


def snapshots_from_raw(raw: dict | None) -> list[AnchorSnapshot]:
    data = (raw or {}).get("assets") if isinstance(raw, dict) else None
    data = data or {}
    out: list[AnchorSnapshot] = []
    for asset, label in ANCHOR_ASSETS:
        row = data.get(asset) or {}
        price = row.get("price")
        chg = row.get("chg")
        missing = row.get("missing")
        if price is None and chg is None:
            missing = missing or "缺数据"
        out.append(
            AnchorSnapshot(
                asset=asset,
                label=label,
                price=price,
                chg=chg,
                missing=missing,
            )
        )
    return out
