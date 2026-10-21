"""Streamlit 看板：今日候选、单票详情、回测、触发线风控、命中率。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from zhiku.config import Paths
from zhiku.engine.exposure import ExposureGuard
from zhiku.engine.rules import RulesSchemaError, load_portfolio, load_symbol_books
from zhiku.notify.render import latest_daily_json, latest_json
from zhiku.review.events import EventLog
from zhiku.review.hitrate import review_week
from zhiku.calendar import now_shanghai
from zhiku.data.fixture import FixtureProvider


def _load_daily(paths: Paths) -> dict | None:
    path = latest_daily_json(paths.reports)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _emergency_panel(news: dict | None) -> None:
    """应急模式始终可见：未触发也写明状态，避免产品页空白。"""
    st.subheader("应急模式")
    em = (news or {}).get("emergency") or {}
    if em.get("active"):
        st.error("应急模式已开启：" + "；".join(em.get("reasons") or []))
        st.write("应对清单：")
        for p in em.get("playbook") or []:
            st.write(f"- {p}")
        if em.get("affected"):
            st.caption(
                "受影响持仓代码："
                + "、".join(em.get("affected") or [])
                + "（不展示数量/成本）"
            )
        if em.get("hedges"):
            st.write("对冲工具（示意，无账户权限推断）：")
            for h in em.get("hedges") or []:
                st.write(f"- {h}")
        if em.get("freeze_adds"):
            st.warning("触发线与新开仓：禁加")
    else:
        st.info("当前未进入应急模式（VIX / 指数盘中跌幅 / 多源冲突关键词均未超阈值）。")
        playbook = em.get("playbook") or [
            "先降风险，不扩大暴露",
            "不抄底",
            "等官方表态",
            "列出受影响持仓代码（不输出数量/成本）和对冲工具",
        ]
        st.caption("预设应对清单（触发后自动带出）：")
        for p in playbook:
            st.write(f"- {p}")


def _frame(df: pd.DataFrame, **kwargs):
    n = len(df.index) if df is not None else 0
    height = min(38 + 36 * max(n, 1), 720)
    st.dataframe(df, hide_index=True, use_container_width=True, height=height, **kwargs)


def _crowding_panel(news: dict | None) -> None:
    st.subheader("拥挤度（反身性）")
    crowd = (news or {}).get("crowding_stats") or {}
    if crowd:
        rows = []
        for bucket, stats in crowd.items():
            rows.append(
                {
                    "分组": bucket,
                    "样本": stats.get("n"),
                    "T+1胜率": stats.get("winrate_t1"),
                    "T+5胜率": stats.get("winrate_t5"),
                }
            )
            _frame(pd.DataFrame(rows))
        st.caption("拥挤时好消息标「利好出尽风险」并降权；冷清加权。缺拥挤输入写缺数据，不编造。")
    else:
        st.info("缺数据：还没有按拥挤度分组的事件研究样本。")


def _examples_kb() -> Path | None:
    here = Path(__file__).resolve()
    cand = here.parents[3] / "examples" / "kb"
    if cand.is_dir():
        return cand
    return None


def main() -> None:
    st.set_page_config(page_title="zhiku 看板", layout="wide")
    st.title("zhiku 投研看板")
    st.caption(
        "研究与提醒工具，不构成投资建议。页面中的持仓、价位、财务均为虚构示例或来自仓库外 ZHIKU_KB。"
    )

    paths = Paths.load()
    daily = _load_daily(paths)
    news = None
    news_path = latest_json(paths.reports, "news")
    if news_path is not None:
        news = json.loads(news_path.read_text(encoding="utf-8"))
    macro = (daily or {}).get("macro") if daily else None
    macro_path = latest_json(paths.reports, "macro")
    if macro is None and macro_path is not None:
        macro = json.loads(macro_path.read_text(encoding="utf-8"))
    tabs = st.tabs(
        [
            "今日候选",
            "单票详情",
            "回测",
            "今日舆情",
            "大V动态",
            "多空机会",
            "宏观与事件",
            "应急与拥挤",
            "触发线风控",
            "每周命中率",
        ]
    )

    with tabs[0]:
        if not daily:
            st.info("还没有每日报告。请先运行 `zk --provider fixture daily --date 2026-10-20`。")
        else:
            st.write(f"截面日期：**{daily.get('asof')}**")
            rows = daily.get("candidates") or []
            if rows:
                df = pd.DataFrame(rows)
                show = df[
                    ["symbol", "name", "total", "formula_score", "qlib_score", "source"]
                ].copy()
                _frame(show)
            else:
                st.warning("今日无候选。")
            notes = daily.get("notes") or []
            if notes:
                with st.expander("运行备注"):
                    for note in notes:
                        st.caption(note)
            rejected = daily.get("quality_rejected") or []
            if rejected:
                with st.expander("未过优质门槛"):
                    st.json(rejected)

    with tabs[1]:
        if not daily or not daily.get("candidates"):
            st.info("无候选可展示。")
        else:
            labels = [f"{c['symbol']} {c['name']}" for c in daily["candidates"]]
            pick = st.selectbox("选择标的", labels)
            item = daily["candidates"][labels.index(pick)]
            st.subheader(f"{item['symbol']} {item['name']}")
            st.write(f"总分 {item['total']:.1f}（{item['source']}）")
            st.json(item.get("screens") or {})
            audits = [a for a in daily.get("audits") or [] if a.get("symbol") == item["symbol"]]
            if audits:
                a = audits[0]
                st.write(f"AI 结论：**{a.get('conclusion')}**")
                if a.get("skip_reason"):
                    st.caption(a["skip_reason"])
                for op in a.get("opinions") or []:
                    st.write(f"- {op.get('master')}：{op.get('stance')} — {op.get('rationale')}")
                if a.get("debate"):
                    st.caption(a["debate"])

    with tabs[2]:
        if not daily or not daily.get("backtests"):
            st.info("无回测结果。")
        else:
            for name, bt in daily["backtests"].items():
                st.subheader(name)
                cols = st.columns(5)
                cols[0].metric(
                    "年化", "缺数据" if bt.get("annualized") is None else f"{bt['annualized']:.2%}"
                )
                cols[1].metric(
                    "最大回撤",
                    "缺数据" if bt.get("max_drawdown") is None else f"{bt['max_drawdown']:.2%}",
                )
                sharpe = bt.get("sharpe")
                if sharpe is None:
                    sharpe_s = "缺数据"
                elif abs(float(sharpe)) > 10:
                    sharpe_s = f"{float(sharpe):.1f}（样本短）"
                else:
                    sharpe_s = f"{float(sharpe):.2f}"
                cols[2].metric("夏普", sharpe_s)
                cols[3].metric(
                    "胜率", "缺数据" if bt.get("win_rate") is None else f"{bt['win_rate']:.1%}"
                )
                status = bt.get("gate_status") or ("通过" if bt.get("passed_gate") else "未通过")
                cols[4].metric("闸门", status)
                cov = bt.get("coverage_ratio")
                if cov is not None:
                    st.caption(
                        f"覆盖 {bt.get('n_covered')}/{bt.get('n_months')} 月（空仓 {bt.get('n_empty')}）。"
                        "对照行须为 T+1 开盘同池等权。今日静态 Top-N 表已作废（前视）。"
                    )
                eq = bt.get("equity") or []
                if eq:
                    edf = pd.DataFrame(eq, columns=["date", "nav"])
                    edf["date"] = pd.to_datetime(edf["date"])
                    st.line_chart(edf.set_index("date"))
                for n in bt.get("notes") or []:
                    st.caption(n)

    with tabs[3]:
        if not news:
            st.info("还没有舆情报告。请先运行 `zk --provider fixture news --date 2026-10-20`。")
        else:
            st.caption(
                f"分析引擎：{news.get('analysis_engine')}；低价值未展开 {news.get('low_value_count', 0)} 条"
            )
            em = news.get("emergency") or {}
            if em.get("active"):
                st.error("应急模式：" + "；".join(em.get("reasons") or []) + "（详见「应急与拥挤」）")
            for a in news.get("analyses") or []:
                st.subheader(a.get("subject") or "")
                st.caption(a.get("source_grade_label") or "")
                if a.get("mark"):
                    st.warning(a["mark"])
                st.write(f"**意味着什么** {a.get('so_what')}")
                st.write(f"**是否消化** {a.get('priced_in')}")
                st.write(f"**影响** {a.get('impact')}")
                st.write(f"**反方** {a.get('counter_view')}")
                conf = a.get("confidence")
                conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf or "缺数据")
                st.write(f"**下一验证** {a.get('next_check')} ｜置信度 {conf_s}")
                if a.get("institution_review"):
                    st.json(a["institution_review"])
            with st.expander("信源分级一览"):
                for c in news.get("clusters") or []:
                    extra = []
                    if c.get("surprise_z") is not None:
                        extra.append(f"z={c.get('surprise_z')}")
                    elif c.get("surprise_missing"):
                        extra.append(c.get("surprise_missing"))
                    if c.get("session"):
                        extra.append(str(c.get("session")))
                    if c.get("stale"):
                        extra.append("已过时")
                    if c.get("crowding_note"):
                        extra.append(c.get("crowding_note"))
                    tail = "｜" + "；".join(extra) if extra else ""
                    st.write(f"- {c.get('source_grade_label')} {c.get('headline')}{tail}")

    with tabs[4]:
        if not news:
            st.info("还没有舆情报告。请先运行 `zk --provider fixture news --date 2026-10-20`。")
        else:
            skipped = news.get("kol_skipped") or ""
            if skipped:
                st.warning(skipped)
            for r in news.get("kol_resonance") or []:
                st.write(r.get("hint") or "")
                if r.get("reverse_risk"):
                    st.caption("情绪极度一致，提示反向拥挤风险。")
            feed = news.get("kol_feed") or []
            if not feed:
                st.write("无")
            else:
                rows = [
                    {
                        "账号": p.get("handle"),
                        "分类": p.get("category_cn"),
                        "方向": p.get("direction"),
                        "代码": ", ".join(p.get("tickers") or []),
                        "摘要": p.get("headline"),
                        "共振": p.get("resonance"),
                        "反向风险": p.get("reverse_risk"),
                        "快讯已匹配": p.get("newswire_matched"),
                    }
                    for p in feed
                ]
                _frame(pd.DataFrame(rows))
            hits = news.get("x_hits") or {}
            if hits.get("accounts"):
                st.subheader("账号命中率（动态权重）")
                _frame(pd.DataFrame(hits["accounts"]))

    with tabs[5]:
        if not news:
            st.info("无多空机会。")
        else:
            em = news.get("emergency") or {}
            if em.get("active"):
                st.error("应急模式：机会已降为只观察/禁加")
            for tier in ("质量档", "交易档", "只观察"):
                rows = (news.get("by_tier") or {}).get(tier) or []
                st.subheader(tier)
                if not rows:
                    st.write("无")
                else:
                    show_cols = [
                        c
                        for c in (
                            "symbol",
                            "name",
                            "direction",
                            "score",
                            "invalidation",
                            "time_stop_on",
                            "surprise_z",
                            "novelty",
                            "crowding",
                            "stale",
                            "earliest_executable",
                        )
                        if rows and c in rows[0]
                    ]
                    _frame(pd.DataFrame(rows)[show_cols] if show_cols else pd.DataFrame(rows))
                    st.caption(
                        "失效条件与时间止损见 invalidation / time_stop_on。已过时或应急禁加会降为只观察。"
                    )

    with tabs[6]:
        if not macro:
            st.info("还没有宏观报告。请先运行 `zk --provider fixture macro --date 2026-10-20`。")
        else:
            geo = macro.get("geo") or {}
            st.metric("地缘温度", geo.get("temperature") or "缺数据")
            st.write("事件窗口：")
            for w in macro.get("window_banners") or []:
                st.warning(w)
            st.write("未来 7 天")
            cal = macro.get("calendar_7d") or []
            if cal:
                _frame(pd.DataFrame(cal))
            else:
                st.write("缺数据")
            st.write("已发布预期差")
            _frame(pd.DataFrame(macro.get("released") or []))
            st.write("锚定资产")
            _frame(pd.DataFrame(macro.get("anchors") or []))
            for a in macro.get("analyses") or []:
                st.subheader(a.get("subject") or "")
                if a.get("mark"):
                    st.caption(a["mark"])
                st.write(a.get("so_what") or "")
            if macro.get("chains_fired"):
                st.write("传导链")
                st.json(macro["chains_fired"])

    with tabs[7]:
        if not news:
            st.info("还没有舆情报告。请先运行 `zk --provider fixture news --date 2026-10-20`。")
        else:
            _emergency_panel(news)
            _crowding_panel(news)
            notes = [
                c.get("crowding_note")
                for c in news.get("clusters") or []
                if c.get("crowding_note")
            ]
            if notes:
                st.caption("标的拥挤备注（虚构样本）：" + "；".join(dict.fromkeys(notes)))

    with tabs[8]:
        st.write("读取 `ZHIKU_KB` 下的规则与持仓；若不存在则回退到仓库内虚构示例。")
        rule_paths = paths
        source = "ZHIKU_KB"
        if not paths.rules.is_dir() or not any(paths.rules.glob("*.yaml")):
            example = _examples_kb()
            if example is not None:
                rule_paths = Paths.load(kb=example, fixtures=example.parent / "fixtures")
                source = "examples（虚构）"
        st.caption(f"当前规则来源：{source}（示例知识库，非真实账户）")
        try:
            books = load_symbol_books(rule_paths.rules)
            _frame(
                pd.DataFrame(
                    [
                        {
                            "symbol": b.symbol,
                            "name": b.name,
                            "market": b.market.value,
                            "禁加": b.add_blocked,
                            "规则数": len(b.rules),
                        }
                        for b in books
                    ]
                )
            )
        except (RulesSchemaError, FileNotFoundError) as exc:
            st.error(str(exc))
        if rule_paths.positions.is_file():
            try:
                portfolio = load_portfolio(rule_paths.positions)
                provider = (
                    FixtureProvider(rule_paths.fixtures) if rule_paths.fixtures.is_dir() else None
                )
                quotes = (
                    provider.latest_quotes([p.symbol for p in portfolio.positions])
                    if provider
                    else {}
                )
                snap = ExposureGuard(rule_paths.exposure_state).evaluate(
                    portfolio, quotes, persist=False, asof=now_shanghai().date()
                )
                ratio_s = f"{snap.ratio:.1%}" if snap.ratio is not None else "缺数据"
                st.write(f"成长仓状态：**{snap.status}** 占比 {ratio_s}")
                blocked = [p.symbol for p in snap.positions if p.add_blocked]
                if blocked:
                    st.warning("禁加: " + ", ".join(blocked))
            except RulesSchemaError as exc:
                st.error(str(exc))

    with tabs[9]:
        events = EventLog(paths.events).read() if paths.events.is_file() else []
        if not events:
            example = _examples_kb()
            if example is not None:
                alt = Paths.load(kb=example, fixtures=example.parent / "fixtures")
                if alt.events.is_file():
                    events = EventLog(alt.events).read()
                    paths = alt
        if not events:
            st.info("还没有触发事件。可先跑 `zk --provider fixture close --date 2026-10-20`。")
        else:
            end = now_shanghai().date()
            if daily and daily.get("asof"):
                end = pd.Timestamp(daily["asof"]).date()
            provider = FixtureProvider(paths.fixtures)
            week = review_week(events, provider, end)
            st.write(f"{week.start} — {week.end}")
            if week.groups:
                _frame(
                    pd.DataFrame(
                        [
                            {
                                "市场": g.market,
                                "规则类型": g.rule_type,
                                "样本": g.total,
                                "兑现": g.hit,
                                "误报": g.false_alarm,
                                "命中率": g.hit_rate,
                            }
                            for g in week.groups
                        ]
                    )
                )
            for lesson in week.lessons:
                st.caption(lesson)
            if news and news.get("crowding_stats"):
                _crowding_panel(news)
            x_hits = (news or {}).get("x_hits") if news else None
            if x_hits and x_hits.get("accounts"):
                st.subheader("大V命中率")
                _frame(pd.DataFrame(x_hits["accounts"]))


if __name__ == "__main__":
    main()
