"""投资大师 agent 适配。

划分（巴菲特 / 芒格 / Graham / Lynch）对齐 virattt/ai-hedge-fund 的 agent 角色，
**未复制**其源码。提示词只使用公开投资框架的要点：
Graham《证券分析》安全边际；Buffett owner earnings / 护城河；Munger 能力圈与逆向；
Lynch 常识与增长。调用需 OPENAI_API_KEY（或 ZHIKU_LLM_KEY）；没 key 则跳过。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from zhiku.review.checklists import MasterChecklist, run_checklists
from zhiku.screens.fundamentals import Fundamentals

MASTERS = {
    "buffett": "巴菲特：强调可理解的生意、持续自由现金流、诚实管理层与合理价格。",
    "munger": "芒格：能力圈、逆向思考、避免蠢行，关注治理与激励。",
    "graham": "格雷厄姆：安全边际、资产价值和利润稳定性，避免概念炒作。",
    "lynch": "林奇：用常识理解公司如何赚钱，增长是否可被订单/份额验证。",
    "fisher": "费雪：长期高回报与研发。",
    "marks": "马克斯：周期位置与二阶思维。",
    "livermore": "利弗莫尔：顺势与止损。",
    "oneil": "欧奈尔：CAN SLIM。",
    "drukenmiller": "德鲁肯米勒：流动性、集中、认错。",
    "dalio": "达利欧：杠杆与风险平价约束。",
    "taleb": "塔勒布：避免脆弱与左尾。",
    "duan": "段永平：商业模式与不懂不做。",
    "zhanglei": "张磊：长期主义与生意本质。",
    "greenblatt": "格林布拉特：高 ROC 与高盈利收益率。",
    "soros": "索罗斯：反身性（缺叙事则缺数据）。",
}

SKIP_REASON_NO_KEY = "未配置 OPENAI_API_KEY / ZHIKU_LLM_KEY，跳过 AI 审核"
SKIP_REASON_NO_LIB = "未安装 openai 库（extra [masters]），跳过 AI 审核"


@dataclass
class MasterOpinion:
    master: str
    stance: str  # 买入 / 观察 / 回避 / 跳过
    rationale: str
    skipped: bool = False
    checks: list[dict] = field(default_factory=list)
    school: str = ""


@dataclass
class AuditCard:
    symbol: str
    name: str
    conclusion: str
    opinions: list[MasterOpinion] = field(default_factory=list)
    debate: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    disagreements: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "conclusion": self.conclusion,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "debate": self.debate,
            "disagreements": list(self.disagreements),
            "opinions": [
                {
                    "master": o.master,
                    "stance": o.stance,
                    "rationale": o.rationale,
                    "skipped": o.skipped,
                    "school": o.school,
                    "checks": o.checks,
                }
                for o in self.opinions
            ],
        }


def llm_key() -> str | None:
    return os.environ.get("ZHIKU_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or None


def _call_openai(prompt: str) -> str | None:
    key = llm_key()
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=key)
    model = os.environ.get("ZHIKU_LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _snapshot_text(row: Fundamentals) -> str:
    payload = row.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _from_checklist(card: MasterChecklist) -> MasterOpinion:
    return MasterOpinion(
        master=card.master,
        stance=card.stance,
        rationale=card.rationale(),
        skipped=False,
        checks=[c.to_dict() for c in card.items],
        school=card.school,
    )


def audit_symbol(
    row: Fundamentals, masters: list[str], *, llm_supplement: bool = False
) -> AuditCard:
    """规则清单默认纯确定性；历史大师补充仅显式请求时启用。"""
    cards, conflicts = run_checklists(row, masters)
    opinions = [_from_checklist(c) for c in cards]
    extra_note = None
    key = llm_key() if llm_supplement else None
    if key:
        try:
            import openai  # noqa: F401
        except ImportError:
            extra_note = SKIP_REASON_NO_LIB
            key = None
    if key:
        for i, master in enumerate(masters):
            desc = MASTERS.get(master, master)
            prompt = (
                f"你扮演{desc}\n"
                "下面是虚构研究用的财务快照（不是真实账户）。"
                "已有规则结论，请用不超过 40 字补充一句，不要改数字。\n"
                f"规则结论：{opinions[i].stance}\n"
                f"{_snapshot_text(row)}"
            )
            try:
                raw = _call_openai(prompt)
            except Exception as exc:  # noqa: BLE001
                extra_note = f"LLM 失败: {exc}"
                break
            if raw:
                opinions[i].rationale = opinions[i].rationale + "｜补充：" + raw[:80]
    elif extra_note is None:
        extra_note = "使用确定性规则清单；统一 AI 研判单独执行"

    conclusion = _vote(opinions)
    return AuditCard(
        symbol=row.symbol,
        name=row.name,
        conclusion=conclusion,
        opinions=opinions,
        skipped=False,
        skip_reason=extra_note,
        disagreements=conflicts,
    )


def _vote(opinions: list[MasterOpinion]) -> str:
    active = [o.stance for o in opinions if not o.skipped]
    if not active:
        return "跳过"
    if active.count("回避") >= 2:
        return "回避"
    if active.count("买入") >= 2:
        return "买入"
    return "观察"
