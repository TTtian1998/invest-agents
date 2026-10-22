"""路径约定：真实规则与持仓在仓库外，由 ZHIKU_KB 指向。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB = Path("kb")
ENV_KB = "ZHIKU_KB"
ENV_FIXTURES = "ZHIKU_FIXTURES"
ENV_NOW = "ZHIKU_NOW"


def kb_root(explicit: str | Path | None = None) -> Path:
    """知识库根目录。默认 ``./kb``，可用环境变量 ``ZHIKU_KB`` 覆盖。"""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    raw = os.environ.get(ENV_KB, str(DEFAULT_KB))
    return Path(raw).expanduser().resolve()


def fixtures_dir(explicit: str | Path | None = None, *, kb: Path | None = None) -> Path:
    """离线行情目录。默认 ``<kb 的上一级>/fixtures``，例如 ``examples/kb`` → ``examples/fixtures``。"""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    raw = os.environ.get(ENV_FIXTURES)
    if raw:
        return Path(raw).expanduser().resolve()
    base = kb if kb is not None else kb_root()
    if base.name == "kb":
        return (base.parent / "fixtures").resolve()
    return (base / "fixtures").resolve()


@dataclass(frozen=True)
class Paths:
    """一次运行用到的全部路径。"""

    kb: Path
    rules: Path
    portfolio: Path
    positions: Path
    journal: Path
    events: Path
    exposure_state: Path
    reports: Path
    fixtures: Path
    cache: Path
    data_dir: Path
    models_dir: Path

    @classmethod
    def load(
        cls,
        kb: str | Path | None = None,
        fixtures: str | Path | None = None,
    ) -> Paths:
        root = kb_root(kb)
        fx = fixtures_dir(fixtures, kb=root)
        journal = root / "journal"
        reports = root / "reports"
        return cls(
            kb=root,
            rules=root / "rules",
            portfolio=root / "portfolio",
            positions=root / "portfolio" / "positions.yaml",
            journal=journal,
            events=journal / "trigger_events.jsonl",
            exposure_state=journal / "exposure_state.json",
            reports=reports,
            fixtures=fx,
            cache=root / "cache" / "market.sqlite",
            data_dir=root / "data",
            models_dir=root / "models",
        )

    def ensure_runtime_dirs(self) -> None:
        self.journal.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
