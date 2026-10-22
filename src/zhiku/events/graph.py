"""公司上下游 + 地缘商品，合并为统一传导图。"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from zhiku.events.models import GraphImpact
from zhiku.macro.chains import load_chains
from zhiku.macro.models import TransmissionChain
from zhiku.news.models import ClusteredNews

PACKAGE_GRAPH = Path(__file__).resolve().parent / "supply_chains.yaml"


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    name: str = ""
    role: str = ""
    hop: int = 0
    direction: str = "同向"

    @field_validator("symbol", mode="before")
    @classmethod
    def _sym(cls, value) -> str:
        return str(value)


class SupplyChain(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    keywords: list[str] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    source_note: str = "公开披露/行业分类示例，占位代码非真实持仓"


def _from_csv(path: Path) -> list[SupplyChain]:
    grouped: dict[str, SupplyChain] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = str(row.get("chain_id") or row.get("id") or "").strip()
            if not cid:
                continue
            chain = grouped.get(cid)
            if chain is None:
                kws = str(row.get("keywords") or "")
                chain = SupplyChain(
                    id=cid,
                    name=str(row.get("chain_name") or cid),
                    keywords=[p.strip() for p in kws.replace("|", ",").split(",") if p.strip()],
                    nodes=[],
                    source_note=str(row.get("source_note") or "CSV 维护的示例链"),
                )
                grouped[cid] = chain
            sym = str(row.get("symbol") or "").strip()
            if not sym:
                continue
            hop_raw = row.get("hop") or 0
            try:
                hop = int(hop_raw)
            except (TypeError, ValueError):
                hop = 0
            chain.nodes.append(
                GraphNode(
                    symbol=sym,
                    name=str(row.get("name") or ""),
                    role=str(row.get("role") or ""),
                    hop=hop,
                    direction=str(row.get("direction") or "同向"),
                )
            )
    return list(grouped.values())


def load_supply_chains(path: Path | None = None) -> list[SupplyChain]:
    data: dict = {}
    if PACKAGE_GRAPH.is_file():
        data = yaml.safe_load(PACKAGE_GRAPH.read_text(encoding="utf-8")) or {}
    rows = data.get("chains") if isinstance(data, dict) else data
    chains = [SupplyChain.model_validate(r) for r in (rows or [])]
    if path is None or not path.is_file():
        return chains
    if path.suffix.lower() == ".csv":
        extra = _from_csv(path)
        by = {c.id: c for c in chains}
        for c in extra:
            by[c.id] = c
        return list(by.values())
    override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(override, dict) and override.get("chains"):
        return [SupplyChain.model_validate(r) for r in override["chains"]]
    return chains


def _blob(cluster: ClusteredNews) -> str:
    return " ".join(
        [
            cluster.headline,
            cluster.symbol or "",
            *[i.headline + " " + i.summary for i in cluster.items],
        ]
    ).lower()


def impacts_for_cluster(
    cluster: ClusteredNews,
    supplies: list[SupplyChain],
    geos: list[TransmissionChain],
) -> list[GraphImpact]:
    blob = _blob(cluster)
    hits: list[GraphImpact] = []
    seen: set[tuple[str, str, int]] = set()
    for chain in supplies:
        keyed = any(k.lower() in blob for k in chain.keywords if k)
        node_hit = any(n.symbol == cluster.symbol or n.symbol.lower() in blob for n in chain.nodes)
        if not (keyed or node_hit):
            continue
        origin = next((n for n in chain.nodes if n.symbol == cluster.symbol), None)
        origin_hop = origin.hop if origin is not None else 0
        for node in chain.nodes:
            hop = abs(node.hop - origin_hop) if origin is not None else node.hop
            if hop == 0:
                continue
            if hop > 2:
                continue
            key = (chain.id, node.symbol, hop)
            if key in seen:
                continue
            seen.add(key)
            direction = node.direction
            if origin is not None and origin.direction == "反向" and node.direction == "同向":
                direction = "反向"
            hits.append(
                GraphImpact(
                    chain_id=chain.id,
                    chain_name=chain.name,
                    kind="company",
                    symbol=node.symbol,
                    name=node.name,
                    hop=hop,
                    direction=direction,
                    role=node.role,
                )
            )
    for geo in geos:
        if not any(k.lower() in blob for k in geo.keywords if k):
            continue
        labels = [*geo.commodities, *geo.a_share, *geo.us]
        for i, lab in enumerate(labels[:6]):
            hop = 1 if i < 3 else 2
            key = (geo.id, lab, hop)
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                GraphImpact(
                    chain_id=geo.id,
                    chain_name=geo.name,
                    kind="geo",
                    symbol="",
                    name=lab,
                    hop=hop,
                    direction="risk_off" if geo.risk_on_off == "risk_off" else "分化",
                    role="commodity" if lab in geo.commodities else "industry",
                )
            )
    return hits


def attach_graph(
    clusters: list[ClusteredNews],
    *,
    supply_path: Path | None,
    geo_path: Path | None,
) -> None:
    supplies = load_supply_chains(supply_path)
    geos = load_chains(geo_path)
    for cluster in clusters:
        hits = impacts_for_cluster(cluster, supplies, geos)
        cluster.graph_impacts = hits
        extra_ids = [h.chain_id for h in hits if h.chain_id not in cluster.chain_ids]
        if extra_ids:
            cluster.chain_ids = list(dict.fromkeys([*cluster.chain_ids, *extra_ids]))
