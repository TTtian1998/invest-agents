"""选股宇宙：fixture 走 YAML；live 走本地库。"""

from __future__ import annotations

from datetime import date

from pathlib import Path

from zhiku.config import Paths
from zhiku.events.pit import load_asof_universe, load_delist_map
from zhiku.pipeline.settings import PipelineSettings
from zhiku.screens.fundamentals import Fundamentals


def fundamentals_path(paths: Paths) -> Path:
    for cand in (
        paths.fixtures / "fundamentals.yaml",
        paths.kb / "fundamentals.yaml",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError("找不到 fundamentals.yaml（请放在 fixtures 或 ZHIKU_KB）")


def load_pipeline_universe(
    paths: Paths,
    on: date,
    *,
    provider_kind: str,
    settings: PipelineSettings,
) -> tuple[list[Fundamentals], dict[str, date], list[str]]:
    from zhiku.pipeline.sources import resolve_source_mode
    mode = resolve_source_mode(provider_kind)
    if mode == "disabled":
        return [], {}, ["数据源 disabled，股票池为空。"]
    if mode == "live":
        from zhiku.data.store import MarketStore
        from zhiku.data.universe_live import build_fundamentals

        store = MarketStore(paths.data_dir)
        extra = list(settings.data.extra_symbols)
        return build_fundamentals(store, on, universe=settings.data.universe, extra=extra, max_spot_age_days=settings.artifacts.price_max_age_days)
    rows = load_asof_universe(
        fundamentals_path=fundamentals_path(paths),
        pit_path=paths.fixtures / "pit_fundamentals.yaml",
        delist_path=paths.fixtures / "delisted.yaml",
        asof=on,
    )
    delist = load_delist_map(paths.fixtures / "delisted.yaml")
    return rows, delist, []
