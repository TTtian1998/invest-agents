"""大师审核：每条是/否/缺数据，必须引用字段。价值派与趋势派相反时记分歧。"""

from __future__ import annotations

from dataclasses import dataclass, field

from zhiku.models import MissingData
from zhiku.screens.classics import graham_number_ratio, ncav_to_mkt, peg
from zhiku.screens.fundamentals import Fundamentals
from zhiku.screens.greenblatt import earnings_yield, roc
from zhiku.screens.piotroski import fscore


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "缺数据"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pct:
        return f"{x:.1%}"
    return f"{x:.4g}"


def _ans(ok: bool | None) -> str:
    if ok is None:
        return "缺数据"
    return "是" if ok else "否"


@dataclass
class CheckItem:
    question: str
    field: str
    answer: str
    evidence: str

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "field": self.field,
            "answer": self.answer,
            "evidence": self.evidence,
        }


@dataclass
class MasterChecklist:
    master: str
    school: str
    items: list[CheckItem] = field(default_factory=list)
    stance: str = "观察"

    def rationale(self) -> str:
        bits = []
        for it in self.items:
            bits.append(f"{it.question}→{it.answer}（{it.field}={it.evidence}）")
        return "；".join(bits) if bits else "无检查项"

    def to_dict(self) -> dict:
        return {
            "master": self.master,
            "school": self.school,
            "stance": self.stance,
            "rationale": self.rationale(),
            "checks": [c.to_dict() for c in self.items],
        }


def _stance(items: list[CheckItem]) -> str:
    known = [i for i in items if i.answer != "缺数据"]
    if len(known) < 3:
        return "观察"
    yes = sum(1 for i in known if i.answer == "是")
    no = sum(1 for i in known if i.answer == "否")
    if yes >= 2 * len(known) / 3:
        return "买入"
    if no >= 2 * len(known) / 3:
        return "回避"
    return "观察"


def _item(question: str, field: str, ok: bool | None, evidence: str) -> CheckItem:
    return CheckItem(question, field, _ans(ok), evidence)


def checklist_for(master: str, row: Fundamentals) -> MasterChecklist:
    fn = _BUILDERS.get(master)
    if fn is None:
        card = MasterChecklist(master, "other")
        card.items.append(CheckItem("未配置清单", "n/a", "缺数据", "无此大师规则模板"))
        card.stance = "观察"
        return card
    card = fn(row)
    card.stance = _stance(card.items)
    return card


def graham_checks(row: Fundamentals) -> MasterChecklist:
    gnr = graham_number_ratio(row)
    n2m = ncav_to_mkt(row)
    items = [
        _item("价格低于格雷厄姆数字", "pe,pb", None if gnr is None else gnr < 1.0, f"ratio={_fmt(gnr)} PE={_fmt(row.pe)} PB={_fmt(row.pb)}"),
        _item("NCAV/市值 > 2/3", "ncav,mktcap", None if n2m is None else n2m > 2.0 / 3.0, f"ncav/mkt={_fmt(n2m)}"),
        _item("PE 不在截面高位", "pe_percentile", None if row.pe_percentile is None else row.pe_percentile <= 0.4, _fmt(row.pe_percentile, True)),
        _item("负债率低于 50%", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.5, _fmt(row.debt_asset, True)),
        _item("非 ST", "st", not row.st, str(row.st)),
        _item("股息或利润为正", "dividend_yield,net_income", None if row.dividend_yield is None and row.net_income is None else ((row.dividend_yield or 0) > 0 or (row.net_income or 0) > 0), f"yield={_fmt(row.dividend_yield, True)} ni={_fmt(row.net_income)}"),
    ]
    return MasterChecklist("graham", "value", items)


def buffett_checks(row: Fundamentals) -> MasterChecklist:
    raw, _, _ = fscore(row)
    ocf_ni = None
    if row.ocf is not None and row.net_income not in (None, 0):
        ocf_ni = row.ocf > row.net_income
    items = [
        _item("三年 ROE 均高于 12%", "roe_3y", None if not row.roe_3y else all(x >= 0.12 for x in row.roe_3y), str(row.roe_3y or row.roe)),
        _item("经营现金流为正", "ocf", None if row.ocf is None else row.ocf > 0, _fmt(row.ocf)),
        _item("经营现金流大于净利润", "ocf,net_income", ocf_ni, f"ocf={_fmt(row.ocf)} ni={_fmt(row.net_income)}"),
        _item("负债率低于 70%", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.7, _fmt(row.debt_asset, True)),
        _item("F-Score ≥ 6", "piotroski", raw >= 6, str(raw)),
        _item("PE 分位不在顶部", "pe_percentile", None if row.pe_percentile is None else row.pe_percentile <= 0.8, _fmt(row.pe_percentile, True)),
        _item("非 ST / 非退市风险", "st,delist_risk", (not row.st) and (not row.delist_risk), f"st={row.st} delist={row.delist_risk}"),
    ]
    return MasterChecklist("buffett", "quality", items)


def munger_checks(row: Fundamentals) -> MasterChecklist:
    gm_ok = None
    if row.gross_margin_3y:
        gm_ok = min(row.gross_margin_3y) > 0.2
    elif row.gross_margin is not None:
        gm_ok = row.gross_margin > 0.2
    items = [
        _item("毛利率保持在 20% 以上", "gross_margin_3y", gm_ok, str(row.gross_margin_3y or row.gross_margin)),
        _item("未新发股份", "shares_issued", None if row.shares_issued is None else (not row.shares_issued), str(row.shares_issued)),
        _item("资产负债率下降或低于 60%", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.6, _fmt(row.debt_asset, True)),
        _item("ROE 为正", "roe", None if row.roe is None else row.roe > 0, _fmt(row.roe, True)),
        _item("非题材小票（市值≥80 亿）", "mktcap", None if row.mktcap is None else row.mktcap >= 8e9, _fmt(row.mktcap)),
        _item("能力圈：行业字段非空", "industry", bool(row.industry), row.industry or "空"),
    ]
    return MasterChecklist("munger", "quality", items)


def lynch_checks(row: Fundamentals) -> MasterChecklist:
    p = peg(row)
    items = [
        _item("PEG < 1", "pe,eps_growth", None if p is None else p < 1.0, f"PEG={_fmt(p)} PE={_fmt(row.pe)} g={_fmt(row.eps_growth, True)}"),
        _item("盈利增速为正", "eps_growth", None if row.eps_growth is None else row.eps_growth > 0, _fmt(row.eps_growth, True)),
        _item("PE 低于 25", "pe", None if row.pe is None else 0 < row.pe < 25, _fmt(row.pe)),
        _item("能指出行业（常识前提）", "industry", bool(row.industry), row.industry or "空"),
        _item("日均成交足够（可跟踪）", "amount_avg", None if row.amount_avg is None else row.amount_avg >= 1e8, _fmt(row.amount_avg)),
        _item("非 ST", "st", not row.st, str(row.st)),
    ]
    return MasterChecklist("lynch", "growth", items)


def fisher_checks(row: Fundamentals) -> MasterChecklist:
    roe_ok = None if row.roe_3y_mean is None else row.roe_3y_mean >= 0.15
    items = [
        _item("三年 ROE 均值 ≥ 15%", "roe_3y", roe_ok, _fmt(row.roe_3y_mean, True)),
        _item("毛利率同比未恶化", "gross_margin", None if row.gross_margin is None or row.gross_margin_prev is None else row.gross_margin >= row.gross_margin_prev, f"{_fmt(row.gross_margin, True)} vs {_fmt(row.gross_margin_prev, True)}"),
        _item("有研发投入记录", "rd_to_sales", None if row.rd_to_sales is None else row.rd_to_sales > 0, _fmt(row.rd_to_sales, True)),
        _item("资产周转未下降", "asset_turnover", None if row.asset_turnover is None or row.asset_turnover_prev is None else row.asset_turnover >= row.asset_turnover_prev, f"{_fmt(row.asset_turnover)} vs {_fmt(row.asset_turnover_prev)}"),
        _item("非 ST", "st", not row.st, str(row.st)),
    ]
    return MasterChecklist("fisher", "quality", items)


def marks_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("估值不在截面最贵 20%", "pe_percentile", None if row.pe_percentile is None else row.pe_percentile <= 0.8, _fmt(row.pe_percentile, True)),
        _item("波动不是截面极端", "volatility", None if row.volatility is None else row.volatility < 0.5, _fmt(row.volatility, True)),
        _item("负债未到危险区", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.75, _fmt(row.debt_asset, True)),
        _item("近一年动量不是过热（<80%）", "momentum_12_1", None if row.momentum_12_1 is None else row.momentum_12_1 < 0.8, _fmt(row.momentum_12_1, True)),
        _item("非退市风险", "delist_risk", not row.delist_risk, str(row.delist_risk)),
    ]
    return MasterChecklist("marks", "cycle", items)


def livermore_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("12−1 月动量为正", "momentum_12_1", None if row.momentum_12_1 is None else row.momentum_12_1 > 0, _fmt(row.momentum_12_1, True)),
        _item("价格距 52 周高点 15% 以内", "dist_to_high", None if row.dist_to_high is None else row.dist_to_high >= -0.15, _fmt(row.dist_to_high, True)),
        _item("非 ST", "st", not row.st, str(row.st)),
        _item("成交额足够以止损", "amount_avg", None if row.amount_avg is None else row.amount_avg >= 3e8, _fmt(row.amount_avg)),
        _item("波动可执行（<60%）", "volatility", None if row.volatility is None else row.volatility < 0.6, _fmt(row.volatility, True)),
    ]
    return MasterChecklist("livermore", "trend", items)


def oneil_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("C：ROA 改善", "roa", None if row.roa is None or row.roa_prev is None else row.roa > row.roa_prev, f"{_fmt(row.roa, True)} vs {_fmt(row.roa_prev, True)}"),
        _item("A：盈利增速为正", "eps_growth", None if row.eps_growth is None else row.eps_growth > 0, _fmt(row.eps_growth, True)),
        _item("L：动量为正或行业龙头", "momentum_12_1,industry_leader", row.industry_leader or (row.momentum_12_1 is not None and row.momentum_12_1 > 0), f"leader={row.industry_leader} mom={_fmt(row.momentum_12_1, True)}"),
        _item("S：未新发股份", "shares_issued", None if row.shares_issued is None else (not row.shares_issued), str(row.shares_issued)),
        _item("I：机构持仓", "n/a", None, "无机构持仓字段"),
        _item("M：大盘择时", "n/a", None, "无大盘择时字段"),
    ]
    return MasterChecklist("oneil", "trend", items)


def drukenmiller_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("趋势未破（动量≥0）", "momentum_12_1", None if row.momentum_12_1 is None else row.momentum_12_1 >= 0, _fmt(row.momentum_12_1, True)),
        _item("流动性：成交额达标", "amount_avg", None if row.amount_avg is None else row.amount_avg >= 3e8, _fmt(row.amount_avg)),
        _item("不是退市风险票", "delist_risk", not row.delist_risk, str(row.delist_risk)),
        _item("估值尚未极端（PE 分位<90%）", "pe_percentile", None if row.pe_percentile is None else row.pe_percentile < 0.9, _fmt(row.pe_percentile, True)),
        _item("宏观流动性", "n/a", None, "无利率/社融字段"),
    ]
    return MasterChecklist("drukenmiller", "trading", items)


def dalio_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("杠杆不高", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.7, _fmt(row.debt_asset, True)),
        _item("波动不是最高档", "volatility", None if row.volatility is None else row.volatility < 0.45, _fmt(row.volatility, True)),
        _item("市值足够分散风险", "mktcap", None if row.mktcap is None else row.mktcap >= 8e9, _fmt(row.mktcap)),
        _item("非 ST", "st", not row.st, str(row.st)),
        _item("跨资产对冲腿", "n/a", None, "股票池内做不了全天候"),
    ]
    return MasterChecklist("dalio", "cycle", items)


def taleb_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("避免高杠杆脆弱性", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.6, _fmt(row.debt_asset, True)),
        _item("避免退市/ST 左尾", "st,delist_risk", (not row.st) and (not row.delist_risk), f"st={row.st} delist={row.delist_risk}"),
        _item("不是极端高估值", "pe_percentile", None if row.pe_percentile is None else row.pe_percentile < 0.85, _fmt(row.pe_percentile, True)),
        _item("波动可知", "volatility", row.volatility is not None, _fmt(row.volatility, True)),
        _item("事项驱动价差", "n/a", None, "无并购/分拆价差"),
    ]
    return MasterChecklist("taleb", "cycle", items)


def duan_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("商业模式：ROE 稳定偏高", "roe_3y", None if not row.roe_3y else min(row.roe_3y) >= 0.12, str(row.roe_3y)),
        _item("现金流为正", "ocf", None if row.ocf is None else row.ocf > 0, _fmt(row.ocf)),
        _item("负债可接受", "debt_asset", None if row.debt_asset is None else row.debt_asset < 0.65, _fmt(row.debt_asset, True)),
        _item("行业非空（懂不懂仍要人判断）", "industry", bool(row.industry), row.industry or "空"),
        _item("非 ST", "st", not row.st, str(row.st)),
    ]
    return MasterChecklist("duan", "quality", items)


def zhanglei_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("长期盈利：三年 ROE 均正", "roe_3y", None if not row.roe_3y else all(x > 0 for x in row.roe_3y), str(row.roe_3y)),
        _item("毛利率未崩", "gross_margin", None if row.gross_margin is None else row.gross_margin > 0.15, _fmt(row.gross_margin, True)),
        _item("经营现金流为正", "ocf", None if row.ocf is None else row.ocf > 0, _fmt(row.ocf)),
        _item("人才与组织", "n/a", None, "无治理/人才字段"),
        _item("非退市风险", "delist_risk", not row.delist_risk, str(row.delist_risk)),
    ]
    return MasterChecklist("zhanglei", "quality", items)


def greenblatt_checks(row: Fundamentals) -> MasterChecklist:
    r = roc(row)
    ey = earnings_yield(row)
    r_ok = None if isinstance(r, MissingData) else (float(r) > 0)
    ey_ok = None if isinstance(ey, MissingData) else (float(ey) > 0)
    items = [
        _item("ROC 为正", "ebit,nwc,nfa", r_ok, str(r)),
        _item("Earnings Yield 为正", "ebit,ev", ey_ok, str(ey)),
        _item("非 ST", "st", not row.st, str(row.st)),
        _item("EV 可用", "ev", row.ev is not None and row.ev > 0, _fmt(row.ev)),
        _item("EBIT 可用", "ebit", row.ebit is not None, _fmt(row.ebit)),
    ]
    return MasterChecklist("greenblatt", "value", items)


def soros_checks(row: Fundamentals) -> MasterChecklist:
    items = [
        _item("价格与基本面同向（动量与 ROE 都正或都负）", "momentum_12_1,roe", None if row.momentum_12_1 is None or row.roe is None else (row.momentum_12_1 > 0) == (row.roe > 0), f"mom={_fmt(row.momentum_12_1, True)} roe={_fmt(row.roe, True)}"),
        _item("估值分位可知", "pe_percentile", row.pe_percentile is not None, _fmt(row.pe_percentile, True)),
        _item("非 ST", "st", not row.st, str(row.st)),
        _item("反身性叙事", "n/a", None, "无文本叙事字段"),
        _item("成交活跃", "amount_avg", None if row.amount_avg is None else row.amount_avg >= 1e8, _fmt(row.amount_avg)),
    ]
    return MasterChecklist("soros", "other", items)


_BUILDERS = {
    "graham": graham_checks,
    "buffett": buffett_checks,
    "munger": munger_checks,
    "lynch": lynch_checks,
    "fisher": fisher_checks,
    "marks": marks_checks,
    "livermore": livermore_checks,
    "oneil": oneil_checks,
    "drukenmiller": drukenmiller_checks,
    "dalio": dalio_checks,
    "taleb": taleb_checks,
    "duan": duan_checks,
    "zhanglei": zhanglei_checks,
    "greenblatt": greenblatt_checks,
    "soros": soros_checks,
}

VALUE_SCHOOLS = {"value", "quality"}
TREND_SCHOOLS = {"trend"}


def disagreements(cards: list[MasterChecklist]) -> list[dict[str, str]]:
    out = []
    value_buy = [c for c in cards if c.school in VALUE_SCHOOLS and c.stance == "买入"]
    value_sell = [c for c in cards if c.school in VALUE_SCHOOLS and c.stance == "回避"]
    trend_buy = [c for c in cards if c.school in TREND_SCHOOLS and c.stance == "买入"]
    trend_sell = [c for c in cards if c.school in TREND_SCHOOLS and c.stance == "回避"]
    if value_buy and trend_sell:
        out.append({
            "kind": "价值 vs 趋势",
            "summary": (
                f"价值/质量派买入（{','.join(c.master for c in value_buy)}），"
                f"趋势派回避（{','.join(c.master for c in trend_sell)}）"
            ),
        })
    if value_sell and trend_buy:
        out.append({
            "kind": "价值 vs 趋势",
            "summary": (
                f"价值/质量派回避（{','.join(c.master for c in value_sell)}），"
                f"趋势派买入（{','.join(c.master for c in trend_buy)}）"
            ),
        })
    return out


def run_checklists(row: Fundamentals, masters: list[str]) -> tuple[list[MasterChecklist], list[dict[str, str]]]:
    cards = [checklist_for(m, row) for m in masters]
    return cards, disagreements(cards)
