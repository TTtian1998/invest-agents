"""利率预期：CME FedWatch 若不可用，用联邦基金期货近似并标注。"""

from __future__ import annotations

import os
from datetime import date

from zhiku.macro.models import RateExpectation
from zhiku.models import MissingData


def from_fixture(raw: dict | None) -> RateExpectation:
    if not raw:
        return RateExpectation(source="缺数据", note="无利率预期样本")
    return RateExpectation.model_validate(raw)


def approximate_from_futures(
    futures_price: float | None,
    current_rate: float | None,
    *,
    meeting: str,
    asof: str,
) -> RateExpectation:
    """100 − 期货价 ≈ 隐含利率；相对当前利率的 25bp 距离近似加/降息概率。

    这不是 CME FedWatch 官方概率，报告必须写明「近似」。
    """
    if futures_price is None or current_rate is None:
        return RateExpectation(
            meeting=meeting,
            asof=asof,
            source="缺数据",
            note="缺数据：无联邦基金期货价格或当前利率，不编造隐含概率",
        )
    implied = 100.0 - futures_price
    gap_bp = (implied - current_rate) * 100.0
    hike = max(0.0, min(1.0, gap_bp / 25.0)) if gap_bp > 0 else 0.0
    cut = max(0.0, min(1.0, -gap_bp / 25.0)) if gap_bp < 0 else 0.0
    hold = max(0.0, 1.0 - hike - cut)
    return RateExpectation(
        meeting=meeting,
        asof=asof,
        implied_rate=round(implied, 4),
        current_rate=current_rate,
        hike_25_prob=round(hike, 4),
        cut_25_prob=round(cut, 4),
        hold_prob=round(hold, 4),
        source="yfinance 联邦基金期货近似（非 CME FedWatch 官方）",
        note="用 ZQ=F / FF 期货价格 100-price 作隐含利率，按 25bp 线性映射概率，仅供对照。",
    )


def live_fedwatch_approx(on: date, *, require_env: bool = True) -> RateExpectation | MissingData:
    """可选联网。失败返回 MissingData，调用方写「缺数据」。live 路径传 require_env=False。"""
    if require_env and os.environ.get("ZHIKU_MACRO_LIVE", "").lower() not in {"1", "true", "yes"}:
        return MissingData("未开启 ZHIKU_MACRO_LIVE，跳过联邦基金期货")
    try:
        import yfinance as yf
    except ImportError:
        return MissingData("未安装 yfinance")
    ticker = os.environ.get("ZHIKU_FF_FUTURES", "ZQ=F")
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception as exc:  # noqa: BLE001
        return MissingData(f"联邦基金期货拉取失败: {exc}")
    if hist is None or hist.empty:
        return MissingData("联邦基金期货无行情")
    close = float(hist["Close"].dropna().iloc[-1])
    current = os.environ.get("ZHIKU_EFFR")
    current_rate = float(current) if current else None
    if current_rate is None:
        implied = 100.0 - close
        return RateExpectation(
            meeting=f"{on.isoformat()} 下次 FOMC",
            asof=on.isoformat(),
            implied_rate=round(implied, 4),
            source="yfinance 联邦基金期货近似（非 CME FedWatch 官方）",
            note="缺数据：未配置 ZHIKU_EFFR，无法映射加/降息概率",
        )
    return approximate_from_futures(
        close, current_rate, meeting=f"{on.isoformat()} 下次 FOMC", asof=on.isoformat()
    )
