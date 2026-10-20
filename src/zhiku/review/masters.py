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

from zhiku.screens.fundamentals import Fundamentals

MASTERS = {
    "buffett": "巴菲特：强调可理解的生意、持续自由现金流、诚实管理层与合理价格。",
    "munger": "芒格：能力圈、逆向思考、避免蠢行，关注治理与激励。",
    "graham": "格雷厄姆：安全边际、资产价值和利润稳定性，避免概念炒作。",
    "lynch": "林奇：用常识理解公司如何赚钱，增长是否可被订单/份额验证。",
}

SKIP_REASON_NO_KEY = "未配置 OPENAI_API_KEY / ZHIKU_LLM_KEY，跳过 AI 审核"
SKIP_REASON_NO_LIB = "未安装 openai 库（extra [masters]），跳过 AI 审核"


@dataclass
class MasterOpinion:
    master: str
    stance: str  # 买入 / 观察 / 回避 / 跳过
    rationale: str
    skipped: bool = False


@dataclass
class AuditCard:
    symbol: str
    name: str
    conclusion: str
    opinions: list[MasterOpinion] = field(default_factory=list)
    debate: str | None = None
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "conclusion": self.conclusion,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "debate": self.debate,
            "opinions": [
                {
                    "master": o.master,
                    "stance": o.stance,
                    "rationale": o.rationale,
                    "skipped": o.skipped,
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


def audit_symbol(row: Fundamentals, masters: list[str]) -> AuditCard:
    key = llm_key()
    if not key:
        return AuditCard(
            symbol=row.symbol,
            name=row.name,
            conclusion="跳过",
            skipped=True,
            skip_reason=SKIP_REASON_NO_KEY,
            opinions=[
                MasterOpinion(m, "跳过", SKIP_REASON_NO_KEY, skipped=True) for m in masters
            ],
        )
    try:
        import openai  # noqa: F401
    except ImportError:
        return AuditCard(
            symbol=row.symbol,
            name=row.name,
            conclusion="跳过",
            skipped=True,
            skip_reason=SKIP_REASON_NO_LIB,
            opinions=[
                MasterOpinion(m, "跳过", SKIP_REASON_NO_LIB, skipped=True) for m in masters
            ],
        )

    opinions: list[MasterOpinion] = []
    for master in masters:
        desc = MASTERS.get(master, master)
        prompt = (
            f"你扮演{desc}\n"
            "下面是虚构研究用的财务快照（不是真实账户）。"
            "请给出结论（买入/观察/回避）和不超过 80 字的中文理由。"
            "只输出 JSON：{{\"stance\": \"...\", \"rationale\": \"...\"}}\n"
            f"{_snapshot_text(row)}"
        )
        try:
            raw = _call_openai(prompt)
        except Exception as exc:  # noqa: BLE001
            opinions.append(MasterOpinion(master, "跳过", f"调用失败: {exc}", skipped=True))
            continue
        if not raw:
            opinions.append(MasterOpinion(master, "跳过", "空响应", skipped=True))
            continue
        stance, rationale = "观察", raw
        try:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(raw[start : end + 1])
                stance = str(data.get("stance") or "观察")
                rationale = str(data.get("rationale") or raw)
        except json.JSONDecodeError:
            pass
        opinions.append(MasterOpinion(master, stance, rationale, skipped=False))

    conclusion = _vote(opinions)
    return AuditCard(symbol=row.symbol, name=row.name, conclusion=conclusion, opinions=opinions)


def _vote(opinions: list[MasterOpinion]) -> str:
    active = [o.stance for o in opinions if not o.skipped]
    if not active:
        return "跳过"
    if active.count("回避") >= 2:
        return "回避"
    if active.count("买入") >= 2:
        return "买入"
    return "观察"
