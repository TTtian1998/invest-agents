"""运行全部选股公式并汇总。"""

from __future__ import annotations

from dataclasses import dataclass, field

from zhiku.pipeline.settings import PipelineSettings
from zhiku.screens.dividend import screen_dividend
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import ScreenRow, screen_greenblatt
from zhiku.screens.multifactor import screen_multifactor
from zhiku.screens.piotroski import screen_piotroski
from zhiku.screens.quality import QualityResult, filter_tradable, filter_universe

SCREEN_FUNCS = {
    "greenblatt": lambda rows, cfg: screen_greenblatt(rows, cfg.greenblatt),
    "piotroski": lambda rows, cfg: screen_piotroski(rows, cfg.piotroski),
    "dividend": lambda rows, cfg: screen_dividend(rows, cfg.dividend),
    "multifactor": lambda rows, cfg: screen_multifactor(rows, cfg.multifactor),
}


@dataclass
class ScreenBundle:
    universe_size: int
    quality_passed: list[str]
    quality: list[QualityResult]
    screens: dict[str, list[ScreenRow]] = field(default_factory=dict)


def formula_scores(bundle: ScreenBundle) -> dict[str, dict[str, float]]:
    """symbol -> {screen: score}。"""
    out: dict[str, dict[str, float]] = {}
    for name, rows in bundle.screens.items():
        for row in rows:
            out.setdefault(row.symbol, {})[name] = row.score
    return out


def average_formula_score(per_screen: dict[str, float]) -> float:
    if not per_screen:
        return 0.0
    return sum(per_screen.values()) / len(per_screen)


def run_screens(
    rows: list[Fundamentals],
    settings: PipelineSettings,
    names: list[str] | None = None,
) -> ScreenBundle:
    passed, quality = filter_universe(rows, settings.quality, settings.universe)
    tradable, _ = filter_tradable(rows, settings.universe)
    wanted = names or list(SCREEN_FUNCS)
    screens: dict[str, list[ScreenRow]] = {}
    for name in wanted:
        fn = SCREEN_FUNCS.get(name)
        if fn is None:
            raise KeyError(f"未知选股公式 {name}，可选：{', '.join(SCREEN_FUNCS)}")
        cfg = getattr(settings, name, None)
        require_q = True if cfg is None else bool(getattr(cfg, "require_quality", True))
        pool = passed if require_q else tradable
        screens[name] = fn(pool, settings)
    return ScreenBundle(
        universe_size=len(rows),
        quality_passed=[r.symbol for r in passed],
        quality=quality,
        screens=screens,
    )
