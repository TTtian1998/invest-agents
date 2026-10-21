"""日历外重大冲击：规则识别应急模式，输出应对清单，可选禁加与推送。"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from zhiku.events.models import Emergency
from zhiku.news.models import ClusteredNews
from zhiku.pipeline.settings import EventSettings

PACKAGE_PLAYBOOK = Path(__file__).resolve().parent / "playbook.yaml"


def keyword_hit(blob: str, key: str) -> bool:
    """中文按子串；英文按词边界，避免 war 命中 forward。"""
    if not key:
        return False
    text = (blob or "").lower()
    needle = key.lower()
    if any("\u4e00" <= ch <= "\u9fff" for ch in key):
        return needle in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text))


def load_playbook(path: Path | None = None) -> list[str]:
    target = path if path is not None and path.is_file() else PACKAGE_PLAYBOOK
    if not target.is_file():
        return []
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return [str(x) for x in (raw.get("playbook") or [])]


def load_shock_fixture(path: Path) -> dict:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def live_shock(*, store=None) -> dict:
    """VIX 与沪深300 价格确认。失败返回空 dict，不回落 fixture。"""
    out: dict = {}
    try:
        import yfinance as yf
    except ImportError:
        out["missing"] = "缺数据：未安装 yfinance，无法取 VIX/指数"
        return out
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        close = vix["Close"].dropna() if vix is not None and not vix.empty else None
        if close is not None and len(close) >= 2 and float(close.iloc[-2]) != 0:
            out["vix_change"] = float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0
            out["vix_level"] = float(close.iloc[-1])
        else:
            out["vix_missing"] = "缺数据：VIX 行情不足"
    except Exception as exc:  # noqa: BLE001
        out["vix_missing"] = f"缺数据：VIX {exc}"
    idx = None
    try:
        hist = yf.Ticker("000300.SS").history(period="5d")
        close = hist["Close"].dropna() if hist is not None and not hist.empty else None
        if close is not None and len(close) >= 2 and float(close.iloc[-2]) != 0:
            idx = float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0
            out["index_symbol"] = "000300.SS"
    except Exception as exc:  # noqa: BLE001
        out["index_missing"] = f"缺数据：000300.SS {exc}"
    if idx is None and store is not None:
        for symbol, adjust in (("510300", "none"), ("510300", "qfq"), ("000300", "qfq")):
            last = store.last_bar(symbol, adjust)
            if not last or last.get("date") is None or last.get("close") is None:
                continue
            from datetime import date as _date

            raw_d = last["date"]
            last_d = _date.fromisoformat(str(raw_d)[:10])
            prev = store.last_bar(symbol, adjust, before=last_d)
            if prev and prev.get("close") and float(prev["close"]) != 0:
                idx = float(last["close"]) / float(prev["close"]) - 1.0
                out["index_symbol"] = symbol
                break
    if idx is not None:
        out["index_intraday"] = idx
    elif "index_missing" not in out:
        out["index_missing"] = "缺数据：无指数涨跌"
    return out


def detect_emergency(
    *,
    clusters: list[ClusteredNews],
    settings: EventSettings,
    shock: dict | None = None,
    holdings: set[str] | None = None,
    playbook: list[str] | None = None,
) -> Emergency:
    reasons: list[str] = []
    shock = shock or {}
    vix = shock.get("vix_change")
    index_drop = shock.get("index_intraday")
    try:
        vix_f = float(vix) if vix is not None else None
    except (TypeError, ValueError):
        vix_f = None
    try:
        idx_f = float(index_drop) if index_drop is not None else None
    except (TypeError, ValueError):
        idx_f = None
    price_reasons: list[str] = []
    if vix_f is not None and vix_f >= settings.vix_spike:
        price_reasons.append(f"VIX 单日涨幅 {vix_f:.0%} 超过阈值 {settings.vix_spike:.0%}")
    if idx_f is not None and idx_f <= -abs(settings.index_intraday_drop):
        price_reasons.append(f"指数跌幅 {idx_f:.0%} 超过阈值 {settings.index_intraday_drop:.0%}")
    keys = tuple(settings.emergency_keywords)
    hits: list[str] = []
    sources: set[str] = set()
    for cluster in clusters:
        blob = (
            cluster.headline + " " + " ".join(i.summary + " " + i.headline for i in cluster.items)
        )
        if any(keyword_hit(blob, k) for k in keys):
            srcs = {i.source for i in cluster.items} | {
                i.x_handle for i in cluster.items if i.x_handle
            }
            sources |= {s for s in srcs if s}
            hits.append(cluster.headline)
    keyword_reason = None
    if len(sources) >= settings.emergency_min_sources and hits:
        keyword_reason = f"{len(sources)} 个快讯源同时出现冲突/制裁/监管关键词"
    require_price = bool(getattr(settings, "require_price_confirm", True))
    price_confirmed = bool(price_reasons)
    if price_reasons:
        reasons.extend(price_reasons)
        if keyword_reason:
            reasons.append(keyword_reason)
    elif keyword_reason and not require_price:
        reasons.append(keyword_reason)
    elif keyword_reason and require_price:
        reasons.append(keyword_reason + "（仅关键词，价格未确认，不进入应急）")
    active = bool(price_reasons) or (bool(reasons) and not require_price)
    if require_price:
        active = bool(price_reasons)
    book = list(playbook or settings.playbook)
    affected = sorted(holdings) if holdings else []
    hedges = [
        "美股：认沽期权或反向 ETF（权限缺数据）",
        "A 股：反向 ETF 或降低新开仓（融券权限缺数据）",
        "不输出任何真实账户或数量",
    ]
    metrics = {
        "vix_change": vix_f,
        "vix_threshold": settings.vix_spike,
        "index_intraday": idx_f,
        "index_threshold": -abs(settings.index_intraday_drop),
        "index_symbol": shock.get("index_symbol"),
        "vix_missing": shock.get("vix_missing") or shock.get("missing"),
        "index_missing": shock.get("index_missing"),
        "require_price_confirm": require_price,
    }
    if not active and keyword_reason and require_price:
        # 保留说明，但不启动应急
        display_reasons = [keyword_reason + "（仅关键词，价格未确认，不进入应急）"]
        if vix_f is not None:
            display_reasons.append(f"VIX 变动 {vix_f:.1%}（阈值 {settings.vix_spike:.0%}）")
        elif shock.get("vix_missing") or shock.get("missing"):
            display_reasons.append(str(shock.get("vix_missing") or shock.get("missing")))
        if idx_f is not None:
            display_reasons.append(f"指数变动 {idx_f:.1%}（阈值 {-abs(settings.index_intraday_drop):.0%}）")
        elif shock.get("index_missing"):
            display_reasons.append(str(shock["index_missing"]))
        reasons = display_reasons
    return Emergency(
        active=active,
        reasons=reasons,
        playbook=book if active else [],
        affected=affected if active else [],
        hedges=hedges if active else [],
        freeze_adds=bool(active and settings.freeze_adds_on_emergency),
        price_confirmed=price_confirmed,
        keyword_hits=hits[:8],
        metrics=metrics,
    )
