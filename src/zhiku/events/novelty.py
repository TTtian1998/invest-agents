"""新颖度、first_seen、发布时段、半衰期衰减。"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from pathlib import Path

from zhiku.calendar import EASTERN, SHANGHAI, add_trading_days, now_shanghai
from zhiku.events.models import NoveltyCard, SessionLabel
from zhiku.models import Market, MissingData
from zhiku.news.models import ClusteredNews
from zhiku.pipeline.settings import EventSettings

_WORD = re.compile(r"[a-z0-9]+", re.I)


def shingles(text: str, n: int = 2) -> set[str]:
    s = re.sub(r"\s+", "", (text or "").lower())
    if len(s) < n:
        tokens = set(_WORD.findall(s))
        return tokens or ({s} if s else set())
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def novelty_against(headline: str, previous: list[str]) -> float:
    if not previous:
        return 1.0
    best = max(jaccard(headline, p) for p in previous)
    return max(0.0, 1.0 - best)


def session_of(published: datetime | None, market: Market | None) -> tuple[SessionLabel, str]:
    if published is None:
        return "缺数据", ""
    mkt = market or Market.CN
    if mkt is Market.US:
        local = published.astimezone(EASTERN) if published.tzinfo else published.replace(tzinfo=EASTERN)
        t = local.time()
        if time(4, 0) <= t < time(9, 30):
            return "盘前", "US"
        if time(9, 30) <= t < time(16, 0):
            return "盘中", "US"
        if time(16, 0) <= t < time(20, 0):
            return "盘后", "US"
        return "隔夜", "US"
    tz = SHANGHAI
    local = published.astimezone(tz) if published.tzinfo else published.replace(tzinfo=tz)
    t = local.time()
    close = time(15, 0) if mkt is Market.CN else time(16, 10)
    if t < time(9, 15):
        return "盘前", mkt.value
    if t < close:
        return "盘中", mkt.value
    return "盘后", mkt.value


def earliest_executable(
    session: SessionLabel,
    market: Market | None,
    on: date,
    *,
    limit_hit: bool | None = None,
) -> str:
    mkt = market or Market.CN
    limit_note = ""
    if limit_hit is True:
        limit_note = "；触及涨跌停，当日不可按方向执行"
    elif limit_hit is None and mkt is Market.CN:
        limit_note = "；涨跌停标志缺数据"
    if mkt is Market.CN:
        nxt = add_trading_days(mkt, on, 1)
        nxt_s = nxt.isoformat() if not isinstance(nxt, MissingData) else "缺数据：下一交易日"
        if session == "盘中":
            return f"今日可买；最早可卖 {nxt_s}（A 股 T+1）{limit_note}"
        return f"最早可买 {nxt_s} 开盘；可卖再下一交易日（A 股 T+1）{limit_note}"
    if session in {"盘后", "隔夜", "盘前"}:
        nxt = add_trading_days(mkt, on, 1)
        nxt_s = nxt.isoformat() if not isinstance(nxt, MissingData) else "缺数据：下一交易日"
        return f"最早可执行：{nxt_s} 开盘{limit_note}"
    return f"当前时段可执行{limit_note}"


def decay_factor(age_hours: float | None, half_life: float | None) -> float:
    if age_hours is None or half_life is None or half_life <= 0:
        return 1.0
    return 0.5 ** (age_hours / half_life)


def estimate_half_life_hours(avg_t1: float | None, avg_t5: float | None, default: float) -> float:
    """用 T+1 / T+5 超额衰减粗估半衰期（小时）。缺样本用配置默认。"""
    if avg_t1 is None or avg_t1 == 0 or avg_t5 is None:
        return default
    if abs(avg_t5) >= abs(avg_t1):
        return default * 1.5
    ratio = abs(avg_t5) / abs(avg_t1)
    if ratio <= 0:
        return max(6.0, default * 0.5)
    # 4 个交易日 ≈ 96h 从 t1 衰减到 t5
    import math

    hl = 96.0 * math.log(2) / max(1e-6, -math.log(ratio))
    return max(6.0, min(168.0, hl))


def load_first_seen(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fp = str(row.get("fingerprint") or "")
        if fp and fp not in out:
            out[fp] = row
    return out


def append_first_seen(path: Path, fingerprint: str, ts: datetime, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_first_seen(path)
    if fingerprint in existing:
        return
    payload = {
        "fingerprint": fingerprint,
        "first_seen": ts.isoformat(),
        "source": source,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def attach_novelty(
    clusters: list[ClusteredNews],
    *,
    on: date,
    settings: EventSettings,
    journal: Path | None,
    previous_headlines: list[str] | None = None,
    hist_avgs: dict[str, tuple[float | None, float | None]] | None = None,
    now: datetime | None = None,
) -> None:
    clock = now or now_shanghai()
    seen = load_first_seen(journal) if journal is not None else {}
    prev = list(previous_headlines or [])
    hist_avgs = hist_avgs or {}
    for cluster in clusters:
        published = next((i.published for i in cluster.items if i.published), None)
        source = cluster.items[0].source if cluster.items else ""
        rec = seen.get(cluster.fingerprint)
        first_seen = published
        first_source = source
        if rec:
            raw = rec.get("first_seen")
            try:
                first_seen = datetime.fromisoformat(str(raw)) if raw else first_seen
            except ValueError:
                first_seen = first_seen
            first_source = str(rec.get("source") or first_source)
        elif journal is not None:
            ts = published or clock
            append_first_seen(journal, cluster.fingerprint, ts, source)
            first_seen = ts
        session, session_mkt = session_of(published, cluster.market)
        default_hl = float(settings.half_life_hours.get(cluster.catalyst) or settings.half_life_hours.get("default") or 36)
        t1, t5 = hist_avgs.get(cluster.catalyst, (None, None))
        hl = estimate_half_life_hours(t1, t5, default_hl)
        age = None
        if first_seen is not None:
            fs = first_seen if first_seen.tzinfo else first_seen.replace(tzinfo=SHANGHAI)
            age = max(0.0, (clock - fs.astimezone(SHANGHAI)).total_seconds() / 3600.0)
        decay = decay_factor(age, hl)
        stale = bool(age is not None and age >= hl * float(settings.stale_after_halflives or 1.0))
        novelty = novelty_against(cluster.headline, prev)
        if cluster.freshness == "旧闻翻炒":
            novelty = min(novelty, 0.2)
        prev.append(cluster.headline)
        limit_hit = None
        card = NoveltyCard(
            first_seen=first_seen,
            first_source=first_source,
            novelty=round(novelty, 4),
            session=session,
            session_market=session_mkt,
            earliest_executable=earliest_executable(session, cluster.market, on, limit_hit=limit_hit),
            age_hours=None if age is None else round(age, 2),
            half_life_hours=hl,
            decay=round(decay, 4),
            stale=stale,
        )
        cluster.novelty = card.novelty
        cluster.first_seen = card.first_seen
        cluster.first_source = card.first_source
        cluster.session = card.session
        cluster.session_market = card.session_market
        cluster.earliest_executable = card.earliest_executable
        cluster.decay = card.decay
        cluster.stale = card.stale
        cluster.half_life_hours = card.half_life_hours
        cluster.novelty_card = card
