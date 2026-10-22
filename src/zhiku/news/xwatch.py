"""X 财经大V：YAML 关注名单、API v2 采集、共振/反向风险、周命中率。"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from zhiku.config import Paths
from zhiku.data.cache import SQLiteCache
from zhiku.models import MissingData
from zhiku.news.grades import GRADE_LABELS, review_institution
from zhiku.news.httputil import cached_json, http_get
from zhiku.news.models import ClusteredNews, InstitutionReview, NewsItem, NewsTag
from zhiku.pipeline.settings import NewsSettings

XCategory = Literal["newswire", "macro_fed", "institutional", "flow_trader", "contrarian"]
CATEGORIES: tuple[XCategory, ...] = (
    "newswire",
    "macro_fed",
    "institutional",
    "flow_trader",
    "contrarian",
)
CATEGORY_CN: dict[str, str] = {
    "newswire": "快讯",
    "macro_fed": "美联储/宏观",
    "institutional": "机构大佬",
    "flow_trader": "交易/资金流",
    "contrarian": "反向参考",
}
TOKEN_ENVS = ("TWITTER_BEARER_TOKEN", "X_BEARER_TOKEN")
API_ROOT = "https://api.twitter.com/2"
SKIP_NO_TOKEN = "未配置 X Bearer token（TWITTER_BEARER_TOKEN / X_BEARER_TOKEN），已跳过 X 大V 采集。"
CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
CN_CODE = re.compile(r"\b(\d{6})\b")
HK_CODE = re.compile(r"\b(\d{1,5}\.HK)\b", re.IGNORECASE)
KNOWN_TICKERS = ("AAPL", "510300", "600519", "000333", "0700.HK", "2800.HK")
BULL_KW = (
    "看多",
    "加仓",
    "买入",
    "增持",
    "bullish",
    "constructive",
    "adding",
    "upgrade",
    "buy into",
    "call buying",
    "core holding",
)
BEAR_KW = (
    "看空",
    "减仓",
    "卖出",
    "减持",
    "bearish",
    "reducing",
    "trimming",
    "downgrade",
    "short",
)
HAWK_KW = ("hawkish", "偏鹰", "higher-for-longer", "higher for longer")
DOVE_KW = ("dovish", "偏鸽", "cut rates", "降息")
MATCH_KEYS = (
    "fomc",
    "federal reserve",
    "fed leaves",
    "fed issues",
    "cpi",
    "pce",
    "nonfarm",
    "议息",
    "降准",
    "lpr",
    "buyback",
    "业绩快报",
    "profit beat",
    "000333",
)


class XSkipped(RuntimeError):
    """未配置 token 或名单不可用，采集跳过。"""


class XAccount(BaseModel):
    model_config = ConfigDict(extra="ignore")

    handle: str
    category: XCategory
    name: str = ""
    known_holdings: list[str] = Field(default_factory=list)
    conflicts: str = ""
    notes: str = ""
    weight: float = 1.0
    enabled: bool = True

    @field_validator("handle", mode="before")
    @classmethod
    def _strip_at(cls, value: Any) -> str:
        return str(value or "").strip().lstrip("@")

    @property
    def handle_key(self) -> str:
        return self.handle.lower()


class XWatchlist(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accounts: list[XAccount] = Field(default_factory=list)
    max_posts_per_account: int = 8
    lookback_hours: int = 36
    throttle_seconds: float = 0.8
    resonance_min_accounts: int = 3
    extreme_share: float = 0.80
    warnings: list[str] = Field(default_factory=list)

    def by_handle(self) -> dict[str, XAccount]:
        return {a.handle_key: a for a in self.accounts if a.enabled}

    def handles(self) -> list[str]:
        return [a.handle for a in self.accounts if a.enabled]


class XCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    handle: str
    date: date
    symbol: str = ""
    direction: str = ""
    headline: str = ""
    t1: float | None = None
    outcome: str | None = None  # hit / miss / pending
    category: str = ""


def bearer_token() -> str | None:
    for key in TOKEN_ENVS:
        raw = os.environ.get(key)
        if raw and raw.strip():
            return raw.strip()
    return None


def example_watchlist_path() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "config" / "x_accounts.example.yaml",
        here.parent / "x_accounts.example.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_watchlist_path(paths: Paths | None = None, settings: NewsSettings | None = None) -> Path | None:
    rel = (settings.x_accounts_file if settings else None) or "config/x_accounts.yaml"
    if paths is not None:
        kb_path = paths.kb / rel
        if kb_path.is_file():
            return kb_path
        alt = paths.kb / "config" / "x_accounts.yaml"
        if alt.is_file():
            return alt
    return example_watchlist_path()


def load_watchlist(
    path: Path | None = None,
    *,
    paths: Paths | None = None,
    settings: NewsSettings | None = None,
) -> XWatchlist:
    target = path or resolve_watchlist_path(paths, settings)
    if target is None or not target.is_file():
        return XWatchlist(warnings=["缺数据：无 X 关注名单 YAML。"])
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return XWatchlist(warnings=[f"关注名单无法解析：{target}"])
    warnings: list[str] = []
    accounts: list[XAccount] = []
    for row in raw.get("accounts") or []:
        if not isinstance(row, dict):
            continue
        cat = str(row.get("category") or "")
        if cat not in CATEGORIES:
            warnings.append(f"跳过未知分类 {cat}（账号 {row.get('handle')}）")
            continue
        try:
            accounts.append(XAccount.model_validate(row))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"跳过无效账号 {row.get('handle')}: {exc}")
    data = {k: v for k, v in raw.items() if k != "accounts"}
    data["accounts"] = [a.model_dump() for a in accounts]
    data["warnings"] = warnings
    return XWatchlist.model_validate(data)


def extract_tickers(text: str, extra: list[str] | None = None) -> list[str]:
    blob = text or ""
    found: list[str] = []
    for tag in extra or []:
        if tag:
            found.append(str(tag).upper())
    for m in CASHTAG.findall(blob):
        found.append(m.upper())
    for m in CN_CODE.findall(blob):
        found.append(m)
    for m in HK_CODE.findall(blob):
        found.append(m.upper())
    for known in KNOWN_TICKERS:
        if re.search(rf"\b{re.escape(known)}\b", blob, re.IGNORECASE):
            found.append(known)
    out: list[str] = []
    seen: set[str] = set()
    for t in found:
        key = t.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(t if t.endswith(".HK") else (t.upper() if t.isalpha() else t))
    return out


def infer_direction(text: str) -> str:
    blob = (text or "").lower()
    bull = any(k.lower() in blob for k in BULL_KW)
    bear = any(k.lower() in blob for k in BEAR_KW)
    hawk = any(k.lower() in blob for k in HAWK_KW)
    dove = any(k.lower() in blob for k in DOVE_KW)
    if bull and not bear:
        return "看多"
    if bear and not bull:
        return "看空"
    if hawk and not dove:
        return "偏鹰"
    if dove and not hawk:
        return "偏鸽"
    return "中性"


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def item_from_post(
    *,
    post_id: str,
    handle: str,
    text: str,
    account: XAccount | None,
    created_at: Any = None,
    url: str = "",
    cashtags: list[str] | None = None,
) -> NewsItem:
    from zhiku.news.classify import infer_catalyst
    from zhiku.news.grades import annotate_item

    category: XCategory = account.category if account is not None else "newswire"
    tickers = extract_tickers(text, cashtags)
    direction = infer_direction(text)
    headline = re.sub(r"\s+", " ", text).strip()[:200]
    source_kind = "rating" if category == "institutional" else "social"
    item = NewsItem(
        id=f"x-{handle}-{post_id}",
        source=f"x:{handle}",
        source_kind=source_kind,
        symbol=tickers[0] if tickers else None,
        headline=headline or f"@{handle}",
        url=url or f"https://x.com/{handle}/status/{post_id}",
        summary=text[:400],
        published=_parse_dt(created_at),
        catalyst=infer_catalyst(text),
        x_handle=handle,
        x_category=category,
        tickers=tickers,
        x_direction=direction,
    )
    if category == "institutional":
        item.source_grade = 3
        item.source_grade_label = GRADE_LABELS[3]
    annotate_item(item)
    if category == "institutional":
        item.source_grade = 3
        item.source_grade_label = GRADE_LABELS[3]
        item.source_kind = "rating"
    return item


def load_fixture_posts(path: Path, watchlist: XWatchlist) -> list[NewsItem]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("posts") if isinstance(raw, dict) else raw
    by = watchlist.by_handle()
    items: list[NewsItem] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        handle = str(row.get("handle") or "").lstrip("@")
        text = str(row.get("text") or row.get("headline") or "")
        if not handle or not text:
            continue
        items.append(
            item_from_post(
                post_id=str(row.get("id") or len(items)),
                handle=handle,
                text=text,
                account=by.get(handle.lower()),
                created_at=row.get("created_at") or row.get("published"),
                url=str(row.get("url") or ""),
                cashtags=list(row.get("cashtags") or []),
            )
        )
    return items


def _x_sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds)


def _cached(
    cache: SQLiteCache | None,
    key: str,
    on: date,
    fetcher,
    *,
    min_interval: float,
) -> Any | MissingData:
    return cached_json(cache, "twitter", key, on, fetcher, min_interval=min_interval)


def _users_by_usernames(
    handles: list[str],
    token: str,
    on: date,
    cache: SQLiteCache | None,
    throttle_seconds: float,
) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for i in range(0, len(handles), 20):
        batch = handles[i : i + 20]
        names = ",".join(batch)
        url = f"{API_ROOT}/users/by?usernames={quote(names)}&user.fields=name,username"

        def fetcher(u=url) -> str | MissingData:
            _x_sleep(throttle_seconds)
            return http_get(u, headers={"Authorization": f"Bearer {token}"})

        data = _cached(cache, f"users-{'-'.join(batch)}", on, fetcher, min_interval=0)
        if isinstance(data, MissingData):
            raise RuntimeError(data.reason)
        for row in (data or {}).get("data") or []:
            uname = str(row.get("username") or "")
            if uname:
                mapping[uname.lower()] = row
    return mapping


def _user_tweets(
    user_id: str,
    token: str,
    on: date,
    cache: SQLiteCache | None,
    *,
    max_results: int,
    lookback_hours: int,
    throttle_seconds: float,
) -> list[dict]:
    max_results = max(5, min(100, int(max_results)))
    start = datetime.combine(on, datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=lookback_hours)
    start_s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{API_ROOT}/users/{user_id}/tweets?max_results={max_results}"
        f"&tweet.fields=created_at,entities,lang,public_metrics"
        f"&start_time={start_s}"
        f"&exclude=retweets,replies"
    )

    def fetcher() -> str | MissingData:
        _x_sleep(throttle_seconds)
        return http_get(url, headers={"Authorization": f"Bearer {token}"})

    data = _cached(cache, f"tweets-{user_id}", on, fetcher, min_interval=0)
    if isinstance(data, MissingData):
        raise RuntimeError(data.reason)
    if isinstance(data, dict):
        return list(data.get("data") or [])
    return []


def items_from_api_payload(
    tweets: list[dict],
    *,
    handle: str,
    account: XAccount | None,
) -> list[NewsItem]:
    items: list[NewsItem] = []
    for row in tweets:
        text = str(row.get("text") or "")
        if not text:
            continue
        raw_ent = row.get("entities")
        entities: dict = raw_ent if isinstance(raw_ent, dict) else {}
        tags = [str(t.get("tag")) for t in (entities.get("cashtags") or []) if isinstance(t, dict) and t.get("tag")]
        tid = str(row.get("id") or len(items))
        items.append(
            item_from_post(
                post_id=tid,
                handle=handle,
                text=text,
                account=account,
                created_at=row.get("created_at"),
                url=f"https://x.com/{handle}/status/{tid}",
                cashtags=tags,
            )
        )
    return items


def fetch_live_posts(
    on: date,
    settings: NewsSettings,
    cache: SQLiteCache | None = None,
    *,
    watchlist: XWatchlist | None = None,
    paths: Paths | None = None,
) -> list[NewsItem]:
    token = bearer_token()
    if not token:
        raise XSkipped(SKIP_NO_TOKEN)
    wl = watchlist or load_watchlist(paths=paths, settings=settings)
    handles = wl.handles()
    if not handles:
        raise XSkipped("X 关注名单为空，已跳过采集。")
    throttle = float(settings.x_throttle_seconds if settings.x_throttle_seconds is not None else wl.throttle_seconds)
    max_posts = settings.x_max_posts or wl.max_posts_per_account
    lookback = settings.x_lookback_hours or wl.lookback_hours
    users = _users_by_usernames(handles, token, on, cache, throttle)
    by = wl.by_handle()
    items: list[NewsItem] = []
    missing: list[str] = []
    for handle in handles:
        row = users.get(handle.lower())
        if not row or not row.get("id"):
            missing.append(handle)
            continue
        tweets = _user_tweets(
            str(row["id"]),
            token,
            on,
            cache,
            max_results=max_posts,
            lookback_hours=lookback,
            throttle_seconds=throttle,
        )
        items.extend(items_from_api_payload(tweets, handle=handle, account=by.get(handle.lower())))
    if missing and not items:
        raise RuntimeError("X 用户查找失败：" + ",".join(missing[:8]))
    return items


def _overlap(a: str, b: str) -> bool:
    la, lb = a.lower(), b.lower()
    if any(k in la and k in lb for k in MATCH_KEYS):
        return True
    tokens_a = {t for t in re.findall(r"[a-zA-Z]{4,}|\d{6}", la)}
    tokens_b = {t for t in re.findall(r"[a-zA-Z]{4,}|\d{6}", lb)}
    shared = tokens_a & tokens_b
    return len(shared) >= 2


def _drop_tag(tags: list[NewsTag], name: NewsTag) -> list[NewsTag]:
    return [t for t in tags if t != name]


def _ensure_tag(tags: list[NewsTag], name: NewsTag) -> list[NewsTag]:
    if name in tags:
        return tags
    return [name, *tags]


def match_newswire(clusters: list[ClusteredNews]) -> None:
    """快讯必须匹配到一级或二级信源后才能标「事实」。"""
    grade12 = [c for c in clusters if (c.source_grade or 9) <= 2]
    for cluster in clusters:
        wires = [i for i in cluster.items if i.x_category == "newswire"]
        if not wires:
            continue
        has_t12 = any((i.source_grade or 9) <= 2 and not i.x_handle for i in cluster.items)
        matched = has_t12
        official_url = cluster.official_url
        if not matched:
            blob = " ".join(i.headline + " " + i.summary for i in wires)
            tickers = list(cluster.tickers)
            for item in wires:
                tickers.extend(item.tickers or [])
            for other in grade12:
                if other is cluster:
                    continue
                other_blob = other.headline + " " + " ".join(i.headline for i in other.items)
                hit = _overlap(blob, other_blob)
                if not hit and other.symbol and other.symbol in tickers:
                    hit = True
                if hit:
                    matched = True
                    official_url = other.official_url or next((i.url for i in other.items if i.url), "")
                    break
        cluster.newswire_matched = matched
        if official_url:
            cluster.official_url = official_url
            for item in wires:
                if not item.official_url:
                    item.official_url = official_url
        if matched:
            cluster.tags = _drop_tag(_ensure_tag(cluster.tags, "事实"), "未经证实")
        else:
            cluster.tags = _drop_tag(cluster.tags, "事实")
            cluster.tags = _ensure_tag(cluster.tags, "未经证实")
            if "情绪" not in cluster.tags:
                cluster.tags = _ensure_tag(cluster.tags, "情绪")


def annotate_resonance(
    clusters: list[ClusteredNews],
    *,
    min_accounts: int = 3,
    extreme_share: float = 0.80,
) -> list[dict]:
    """多账号同向 → 共振；情绪极度一致 → 反向风险。"""
    by_key: dict[str, list[tuple[ClusteredNews, NewsItem, str]]] = {}
    for cluster in clusters:
        for item in cluster.items:
            if not item.x_handle:
                continue
            direction = item.x_direction or infer_direction(item.headline + item.summary)
            if direction in {"中性", ""}:
                continue
            tickers = item.tickers or cluster.tickers or extract_tickers(item.headline + " " + item.summary)
            keys = tickers or [f"theme:{direction}"]
            for key in keys:
                by_key.setdefault(key, []).append((cluster, item, direction))
    reports: list[dict] = []
    touched: set[str] = set()
    for key, rows in by_key.items():
        handles = list(dict.fromkeys(r[1].x_handle for r in rows))
        if len(handles) < min_accounts:
            continue
        dirs = [r[2] for r in rows]
        majority = max(set(dirs), key=dirs.count)
        share = dirs.count(majority) / max(1, len(dirs))
        reverse = share >= extreme_share
        reports.append(
            {
                "key": key,
                "handles": handles,
                "direction": majority,
                "share": round(share, 3),
                "resonance": True,
                "reverse_risk": reverse,
                "hint": (
                    f"「{key}」{len(handles)} 个账号观点同向（{majority}，占比 {share:.0%}），标共振。"
                    + ("情绪极度一致，提示反向拥挤风险。" if reverse else "")
                ),
            }
        )
        for cluster, _item, _d in rows:
            cluster.resonance = True
            if reverse:
                cluster.reverse_risk = True
            if cluster.fingerprint not in touched:
                if "市场解读" not in cluster.tags:
                    cluster.tags = [*cluster.tags, "市场解读"]
                touched.add(cluster.fingerprint)
    return reports


def in_fomc_window(macro_bundle: Any, on: date) -> bool:
    if macro_bundle is None:
        return False
    windows = list(getattr(macro_bundle, "windows", None) or getattr(macro_bundle, "active_windows", None) or [])
    for w in windows:
        if isinstance(w, dict):
            et = str(w.get("event_type") or "")
            start, end = w.get("start"), w.get("end")
        else:
            et = str(getattr(w, "event_type", "") or "")
            start, end = getattr(w, "start", None), getattr(w, "end", None)
        if et not in {"FOMC", "FOMC_DOTS", "FOMC_SPEECH"}:
            continue
        if start is None or end is None:
            continue
        s = start if isinstance(start, date) else date.fromisoformat(str(start)[:10])
        e = end if isinstance(end, date) else date.fromisoformat(str(end)[:10])
        if s <= on <= e:
            return True
    return False


def apply_fomc_boost(clusters: list[ClusteredNews], *, active: bool) -> None:
    if not active:
        return
    for cluster in clusters:
        if any(i.x_category == "macro_fed" for i in cluster.items) or cluster.x_category == "macro_fed":
            cluster.fomc_boost = True


def fill_tickers(clusters: list[ClusteredNews]) -> None:
    for cluster in clusters:
        tickers: list[str] = list(cluster.tickers)
        for item in cluster.items:
            for t in item.tickers or extract_tickers(item.headline + " " + item.summary):
                if t not in tickers:
                    tickers.append(t)
        cluster.tickers = tickers
        handles = [i.x_handle for i in cluster.items if i.x_handle]
        if handles:
            cluster.x_handles = list(dict.fromkeys(handles))
            cats = [i.x_category for i in cluster.items if i.x_category]
            cluster.x_category = cats[0] if cats else cluster.x_category
        if not cluster.symbol and tickers:
            cluster.symbol = tickers[0]


def apply_institutional_review(
    clusters: list[ClusteredNews],
    watchlist: XWatchlist,
    track: dict[str, dict] | None = None,
) -> None:
    by = watchlist.by_handle()
    for cluster in clusters:
        inst_items = [i for i in cluster.items if i.x_category == "institutional"]
        if not inst_items and cluster.source_grade != 3:
            continue
        if inst_items and (cluster.source_grade or 9) > 3:
            cluster.source_grade = 3
            cluster.source_grade_label = GRADE_LABELS[3]
        handle = (inst_items[0].x_handle if inst_items else "") or ""
        acct = by.get(handle.lower())
        extra_track = dict(track or {})
        if acct is not None:
            extra_track.setdefault(
                cluster.items[0].source if cluster.items else handle,
                {},
            )
        review = review_institution(cluster, extra_track, account=acct)
        cluster.institution_review = review


def fomc_weight(cluster: ClusteredNews, settings: NewsSettings) -> float:
    if cluster.fomc_boost:
        return float(settings.x_fomc_boost or 1.35)
    return 1.0


def account_weight(cluster: ClusteredNews, weights: dict[str, dict[str, float]]) -> float:
    table = weights.get("x_handle") or {}
    handles = cluster.x_handles or [i.x_handle for i in cluster.items if i.x_handle]
    if not handles:
        return 1.0
    vals = [float(table.get(h.lower(), table.get(h, 1.0))) for h in handles]
    return sum(vals) / len(vals)


def key_calls_from_clusters(clusters: list[ClusteredNews], on: date) -> list[XCall]:
    """带方向的个股判断进入每周命中率。"""
    out: list[XCall] = []
    seen: set[tuple[str, str, str]] = set()
    for cluster in clusters:
        for item in cluster.items:
            if not item.x_handle:
                continue
            direction = item.x_direction or infer_direction(item.headline)
            if direction not in {"看多", "看空"}:
                continue
            tickers = item.tickers or cluster.tickers
            if not tickers:
                continue
            for symbol in tickers:
                key = (item.x_handle.lower(), symbol, direction)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    XCall(
                        handle=item.x_handle,
                        date=on,
                        symbol=symbol,
                        direction=direction,
                        headline=item.headline,
                        category=item.x_category,
                        outcome="pending",
                    )
                )
    return out


def load_x_calls(path: Path) -> list[XCall]:
    if not path.is_file():
        return []
    if path.suffix == ".jsonl":
        rows: list[XCall] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(XCall.model_validate(json.loads(line)))
            except Exception:  # noqa: BLE001
                continue
        return rows
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("calls") if isinstance(raw, dict) else raw
    out: list[XCall] = []
    for row in data or []:
        if not isinstance(row, dict):
            continue
        if True in row and "date" not in row:
            row = dict(row)
            row["date"] = row.pop(True)
        try:
            out.append(XCall.model_validate(row))
        except Exception:  # noqa: BLE001
            continue
    return out


def append_x_calls(path: Path, calls: list[XCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {(c.handle.lower(), c.date.isoformat(), c.symbol, c.direction) for c in load_x_calls(path)}
    with path.open("a", encoding="utf-8") as fh:
        for call in calls:
            key = (call.handle.lower(), call.date.isoformat(), call.symbol, call.direction)
            if key in existing:
                continue
            fh.write(call.model_dump_json() + "\n")
            existing.add(key)


def classify_call(call: XCall) -> str:
    if call.outcome in {"hit", "miss", "pending"}:
        if call.outcome != "pending":
            return call.outcome
    if call.t1 is None or call.direction not in {"看多", "看空"}:
        return "pending"
    if call.direction == "看多":
        return "hit" if call.t1 > 0 else "miss"
    return "hit" if call.t1 < 0 else "miss"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def summarize_x_hits(calls: list[XCall]) -> dict:
    by: dict[str, dict[str, Any]] = {}
    for call in calls:
        slot = by.setdefault(
            call.handle,
            {"handle": call.handle, "hit": 0, "miss": 0, "pending": 0, "total": 0, "category": call.category},
        )
        label = classify_call(call)
        slot[label] = _as_int(slot.get(label)) + 1
        slot["total"] = _as_int(slot.get("total")) + 1
        if call.category:
            slot["category"] = call.category
    rows: list[dict] = []
    for handle, slot in sorted(by.items()):
        hit_n = _as_int(slot.get("hit"))
        miss_n = _as_int(slot.get("miss"))
        resolved = hit_n + miss_n
        rate = (hit_n / resolved) if resolved else None
        rows.append({**slot, "handle": handle, "hit_rate": rate})
    overall_h = sum(r["hit"] for r in rows)
    overall_m = sum(r["miss"] for r in rows)
    resolved = overall_h + overall_m
    return {
        "accounts": rows,
        "hit": overall_h,
        "miss": overall_m,
        "pending": sum(r["pending"] for r in rows),
        "hit_rate": (overall_h / resolved) if resolved else None,
    }


def update_x_weights(
    weights: dict[str, dict[str, float]],
    hits: dict,
    *,
    low: float = 0.45,
    high: float = 0.55,
) -> dict[str, dict[str, float]]:
    table = dict(weights.get("x_handle") or {})
    for row in hits.get("accounts") or []:
        rate = row.get("hit_rate")
        if rate is None:
            continue
        handle = str(row.get("handle") or "").lower()
        if not handle:
            continue
        cur = table.get(handle, 1.0)
        if rate < low:
            cur *= 0.9
        elif rate > high:
            cur *= 1.05
        table[handle] = max(0.3, min(1.5, cur))
    out = dict(weights)
    out["x_handle"] = table
    return out


def annotate_x_clusters(
    clusters: list[ClusteredNews],
    watchlist: XWatchlist,
    *,
    settings: NewsSettings,
    macro_bundle: Any = None,
    on: date | None = None,
    track: dict[str, dict] | None = None,
) -> list[dict]:
    fill_tickers(clusters)
    match_newswire(clusters)
    apply_institutional_review(clusters, watchlist, track)
    if on is not None:
        apply_fomc_boost(clusters, active=in_fomc_window(macro_bundle, on))
    resonance = annotate_resonance(
        clusters,
        min_accounts=settings.x_resonance_min or watchlist.resonance_min_accounts,
        extreme_share=settings.x_extreme_share or watchlist.extreme_share,
    )
    return resonance


def kol_feed(clusters: list[ClusteredNews]) -> list[dict]:
    rows: list[dict] = []
    for cluster in clusters:
        x_items = [i for i in cluster.items if i.x_handle]
        if not x_items:
            continue
        review: InstitutionReview | None = cluster.institution_review
        for item in x_items:
            rows.append(
                {
                    "handle": item.x_handle,
                    "category": item.x_category,
                    "category_cn": CATEGORY_CN.get(item.x_category, item.x_category),
                    "headline": item.headline,
                    "tickers": item.tickers or cluster.tickers,
                    "direction": item.x_direction,
                    "tags": cluster.tags,
                    "newswire_matched": cluster.newswire_matched,
                    "resonance": cluster.resonance,
                    "reverse_risk": cluster.reverse_risk,
                    "fomc_boost": cluster.fomc_boost,
                    "source_grade": item.source_grade or cluster.source_grade,
                    "source_grade_label": item.source_grade_label or cluster.source_grade_label,
                    "url": item.url,
                    "institution_review": review.to_dict() if review and item.x_category == "institutional" else None,
                }
            )
    return rows
