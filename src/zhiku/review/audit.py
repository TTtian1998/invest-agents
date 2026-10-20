"""AI 审核：大师 agent + TradingAgents 辩论（都可跳过）。"""

from __future__ import annotations

from zhiku.debate.tradingagents_adapter import DebateNotInstalled, run_debate
from zhiku.review.masters import AuditCard, audit_symbol
from zhiku.screens.fundamentals import Fundamentals


def audit_top(
    rows: list[Fundamentals],
    masters: list[str],
    top_n: int = 15,
) -> list[AuditCard]:
    cards: list[AuditCard] = []
    for row in rows[:top_n]:
        card = audit_symbol(row, masters)
        try:
            card.debate = run_debate(row.symbol)
        except DebateNotInstalled:
            card.debate = "TradingAgents 未安装，跳过多空辩论。"
        except Exception as exc:  # noqa: BLE001
            card.debate = f"辩论层跳过: {exc}"
        cards.append(card)
    return cards
