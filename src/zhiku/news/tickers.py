"""从标题/摘要挂 A 股代码。只匹配已知股票池名称或 6 位代码，不编造。"""

from __future__ import annotations

import re

from zhiku.models import Market
from zhiku.news.models import ClusteredNews

CN_CODE = re.compile(r"(?<!\d)([036]\d{5})(?!\d)")


def attach_cn_tickers(clusters: list[ClusteredNews], name_map: dict[str, str] | None) -> int:
    """name_map: 中文名 → 代码（通常来自当日股票池）。返回新挂上代码的簇数。"""
    if not clusters:
        return 0
    names = {k: v for k, v in (name_map or {}).items() if k and len(k) >= 2}
    # 长名优先，避免「中国」误伤
    ordered = sorted(names.items(), key=lambda kv: len(kv[0]), reverse=True)
    attached = 0
    for cluster in clusters:
        blob = cluster.headline + " " + " ".join(
            (i.summary or "") + " " + (i.headline or "") for i in cluster.items
        )
        found: list[str] = list(cluster.tickers)
        hit_name = ""
        for m in CN_CODE.findall(blob):
            if m not in found:
                found.append(m)
        for name, symbol in ordered:
            if name in blob:
                if symbol not in found:
                    found.append(symbol)
                if not hit_name:
                    hit_name = name
        if found != list(cluster.tickers):
            cluster.tickers = found
        if not cluster.symbol and found:
            cluster.symbol = found[0]
            if hit_name:
                cluster.name = cluster.name or hit_name
            if cluster.market is None:
                cluster.market = Market.CN
            attached += 1
        elif cluster.symbol and cluster.market is None and cluster.symbol in set(names.values()):
            cluster.market = Market.CN
    return attached
