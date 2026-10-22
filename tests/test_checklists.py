"""规则审核：引用字段，价值/趋势分歧，无 LLM 不跳过。"""

from __future__ import annotations

from zhiku.review.checklists import disagreements, run_checklists
from zhiku.review.masters import audit_symbol
from tests.test_screens import _row


def test_audit_without_llm_uses_rules_and_cites_fields() -> None:
    row = _row(symbol="000333", name="示例家电个股", pe=10, pb=1.2, pe_percentile=0.2, eps_growth=0.2)
    card = audit_symbol(row, ["graham", "buffett", "livermore"])
    assert card.skipped is False
    assert card.opinions
    assert all(o.checks for o in card.opinions)
    g = next(o for o in card.opinions if o.master == "graham")
    assert "pe" in g.rationale.lower() or "PE" in g.rationale or "pe" in g.rationale
    assert all(c["answer"] in {"是", "否", "缺数据"} for c in g.checks)


def test_value_trend_disagreement() -> None:
    cheap_down = _row(
        symbol="VAL",
        pe=6,
        pb=0.8,
        pe_percentile=0.1,
        debt_asset=0.3,
        ncav=80,
        mktcap=90,
        momentum_12_1=-0.3,
        dist_to_high=-0.5,
        amount_avg=1e7,
    )
    cards, conflicts = run_checklists(cheap_down, ["graham", "livermore"])
    by = {c.master: c for c in cards}
    assert by["graham"].stance in {"买入", "观察"}
    assert by["livermore"].stance in {"回避", "观察"}
    # 强制分歧：价值买入 + 趋势回避
    by["graham"].stance = "买入"
    by["livermore"].stance = "回避"
    d = disagreements([by["graham"], by["livermore"]])
    assert d and d[0]["kind"] == "价值 vs 趋势"


def test_oneil_marks_institution_and_market_missing() -> None:
    row = _row(symbol="X")
    cards, _ = run_checklists(row, ["oneil"])
    missing = [i for i in cards[0].items if i.answer == "缺数据"]
    assert any("机构" in i.question or i.field == "n/a" for i in missing)
