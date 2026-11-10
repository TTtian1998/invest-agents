"""确定性规则审核。日报不在这里触发大师 LLM 或 TradingAgents。"""

from __future__ import annotations

from zhiku.review.masters import AuditCard, audit_symbol
from zhiku.screens.fundamentals import Fundamentals


def audit_top(
    rows: list[Fundamentals],
    masters: list[str],
    top_n: int = 15,
) -> list[AuditCard]:
    return [audit_symbol(row, masters) for row in rows[:top_n]]
