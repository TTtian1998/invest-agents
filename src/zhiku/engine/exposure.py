"""成长仓暴露闸门：带滞回。汇率只读 YAML，不联网。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from zhiku.models import Market, MissingData, Portfolio, Quote

STATUS_FROZEN = "FROZEN"
STATUS_HOLD = "HOLD"
STATUS_OK = "OK"


def fx_to_cny(market: Market, fx: dict[str, float]) -> float | MissingData:
    if market is Market.CN:
        return 1.0
    if market is Market.US:
        for key in ("USD_CNY", "USDCNY", "USD"):
            if key in fx:
                return float(fx[key])
        return MissingData("YAML 未提供 USD_CNY 汇率，拒绝估算")
    if market is Market.HK:
        for key in ("HKD_CNY", "HKDCNY", "HKD"):
            if key in fx:
                return float(fx[key])
        return MissingData("YAML 未提供 HKD_CNY 汇率，拒绝估算")
    return MissingData(f"未知市场 {market}")


@dataclass
class PositionValue:
    symbol: str
    name: str
    market: Market
    qty: float
    price: float | None
    fx: float | None
    value_cny: float | None
    growth: bool
    add_blocked: bool
    missing: str | None = None


@dataclass
class ExposureSnapshot:
    status: str
    ratio: float | None
    growth_cny: float | None
    total_cny: float | None
    freeze_threshold: float
    unfreeze_threshold: float
    frozen: bool
    positions: list[PositionValue]
    notes: list[str]

    @property
    def status_cn(self) -> str:
        return {
            STATUS_FROZEN: "冻结（禁新开仓、禁加）",
            STATUS_HOLD: "维持冻结",
            STATUS_OK: "正常",
        }.get(self.status, self.status)


class ExposureGuard:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load_status(self) -> str | None:
        if not self.state_path.is_file():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        status = data.get("status")
        if status in {STATUS_FROZEN, STATUS_HOLD, STATUS_OK}:
            return str(status)
        return None

    def save(self, snap: ExposureSnapshot, asof: date) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": snap.status,
            "ratio": snap.ratio,
            "asof": asof.isoformat(),
            "frozen": snap.frozen,
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def next_status(self, ratio: float, freeze: float, unfreeze: float, previous: str | None) -> str:
        if ratio >= freeze:
            return STATUS_FROZEN
        if ratio < unfreeze:
            return STATUS_OK
        if previous in {STATUS_FROZEN, STATUS_HOLD}:
            return STATUS_HOLD
        return STATUS_OK

    def evaluate_ratio(self, ratio: float, *, freeze: float = 0.60, unfreeze: float = 0.50) -> ExposureSnapshot:
        previous = self.load_status()
        status = self.next_status(ratio, freeze, unfreeze, previous)
        return ExposureSnapshot(
            status=status,
            ratio=ratio,
            growth_cny=None,
            total_cny=None,
            freeze_threshold=freeze,
            unfreeze_threshold=unfreeze,
            frozen=status in {STATUS_FROZEN, STATUS_HOLD},
            positions=[],
            notes=[],
        )

    def evaluate(
        self,
        portfolio: Portfolio,
        quotes: dict[str, Quote | MissingData],
        *,
        persist: bool = True,
        asof: date | None = None,
    ) -> ExposureSnapshot:
        notes: list[str] = []
        rows: list[PositionValue] = []
        total = 0.0
        growth = 0.0
        priced = 0
        for pos in portfolio.positions:
            quote = quotes.get(pos.symbol)
            if quote is None:
                rows.append(
                    PositionValue(
                        pos.symbol, pos.name, pos.market, pos.qty, None, None, None,
                        pos.growth, pos.add_blocked, "缺数据：无最新价",
                    )
                )
                notes.append(f"{pos.symbol} 缺数据：无最新价")
                continue
            if isinstance(quote, MissingData):
                rows.append(
                    PositionValue(
                        pos.symbol, pos.name, pos.market, pos.qty, None, None, None,
                        pos.growth, pos.add_blocked, f"缺数据：{quote.reason}",
                    )
                )
                notes.append(f"{pos.symbol} 缺数据：{quote.reason}")
                continue
            rate = fx_to_cny(pos.market, portfolio.fx)
            if isinstance(rate, MissingData):
                rows.append(
                    PositionValue(
                        pos.symbol, pos.name, pos.market, pos.qty, quote.price, None, None,
                        pos.growth, pos.add_blocked, f"缺数据：{rate.reason}",
                    )
                )
                notes.append(f"{pos.symbol} 缺数据：{rate.reason}")
                continue
            value = pos.qty * quote.price * rate
            rows.append(
                PositionValue(
                    pos.symbol, pos.name, pos.market, pos.qty, quote.price, rate, value,
                    pos.growth, pos.add_blocked, None,
                )
            )
            total += value
            priced += 1
            if pos.growth:
                growth += value

        ratio: float | None
        if priced == 0 or total <= 0:
            ratio = None
            notes.append("缺数据：无有效市值，无法计算成长仓占比")
            status = self.load_status() or STATUS_OK
        else:
            ratio = growth / total
            status = self.next_status(
                ratio, portfolio.freeze_threshold, portfolio.unfreeze_threshold, self.load_status()
            )

        snap = ExposureSnapshot(
            status=status,
            ratio=ratio,
            growth_cny=growth if priced else None,
            total_cny=total if priced else None,
            freeze_threshold=portfolio.freeze_threshold,
            unfreeze_threshold=portfolio.unfreeze_threshold,
            frozen=status in {STATUS_FROZEN, STATUS_HOLD},
            positions=rows,
            notes=notes,
        )
        if persist:
            day = asof or date.today()
            self.save(snap, day)
        return snap
