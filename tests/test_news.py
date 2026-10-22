"""新闻雷达：信源分级、分析降级、字段完整性。全部离线虚构样本。"""

from __future__ import annotations

from datetime import date

from zhiku.analyze import ANALYSIS_FIELDS, DEGRADED_MARK, llm_enabled
from zhiku.config import Paths
from zhiku.news.aggregate import cluster_items
from zhiku.news.analyze import is_important
from zhiku.news.grades import REVIEW_FIELDS, grade_for, grade_weight, review_institution
from zhiku.news.models import ClusteredNews, NewsItem
from zhiku.news.opportunities import build_opportunities
from zhiku.news.runner import run_news
from zhiku.news.sources import load_fixture_news
from zhiku.pipeline.settings import NewsSettings, load_pipeline_settings
from zhiku.screens.fundamentals import Fundamentals


def test_grade_official_vs_social_and_hot() -> None:
    g1, label1, w1 = grade_for("fed", "official")
    g3, _, w3 = grade_for("ratings", "rating")
    g4, _, w4 = grade_for("eastmoney_hot", "social")
    g2, _, _ = grade_for("eastmoney", "media")
    assert g1 == 1 and "一级" in label1
    assert g3 == 3
    assert g4 == 4
    assert g2 == 2
    assert w1 > w3 > w4


def test_people_familiar_not_fact_and_tier3_review(examples_dir) -> None:
    items, _ = load_fixture_news(examples_dir / "fixtures" / "news.yaml")
    clusters = cluster_items(items, {"000333", "AAPL", "600519"})
    by_head = {c.headline: c for c in clusters}

    leak = next(c for c in clusters if "据知情人士" in c.headline)
    assert leak.source_grade == 2
    assert leak.unofficial is True
    assert "未经官方证实" in leak.tags
    assert "事实" not in leak.tags
    assert leak.exclusive is True

    rating = next(c for c in clusters if "目标价" in c.headline)
    assert rating.source_grade == 3
    assert "事实" not in rating.tags
    assert "市场解读" in rating.tags
    review = review_institution(rating, {"ratings": {"hits": 3, "misses": 2}})
    for field in REVIEW_FIELDS:
        assert getattr(review, field), field

    official = next(c for c in clusters if c.source_grade == 1 and "FOMC" in c.headline)
    media = next(c for c in clusters if c.headline.startswith("Fed issues FOMC"))
    assert media.official_url or official.items[0].url
    assert media.official_url == official.items[0].url or media.official_url.endswith("example-fomc")
    del by_head


def test_tier3_cannot_be_quality_opportunity() -> None:
    cluster = ClusteredNews(
        fingerprint="x",
        symbol="000333",
        name="示例家电个股",
        headline="示例券商上调目标价",
        catalyst="rating_up",
        tags=["市场解读"],
        source_grade=3,
        source_grade_label="三级·投资机构",
        items=[
            NewsItem(
                id="r",
                source="ratings",
                source_kind="rating",
                symbol="000333",
                headline="示例券商上调目标价",
                catalyst="rating_up",
                source_grade=3,
            )
        ],
    )
    cluster.institution_review = review_institution(cluster, {"ratings": {"hits": 1, "misses": 1}})
    fund = Fundamentals(symbol="000333", name="示例家电个股", industry="家电", industry_leader=True)
    opps = build_opportunities(
        [cluster],
        funds={"000333": fund},
        quality_passed={"000333"},
        screen_scores={"000333": {"greenblatt": 80}},
        qlib_scores={},
        holding_symbols={"000333"},
        prices={"000333": 10.0},
        hist={"rating_up": 0.4},
        weights={"source": {}, "catalyst": {}},
        settings=NewsSettings(),
    )
    assert opps
    assert all(o.tier != "质量档" for o in opps)
    assert opps[0].institution_review is not None
    for field in REVIEW_FIELDS:
        assert opps[0].institution_review.get(field)


def test_grade_weight_affects_score() -> None:
    official = ClusteredNews(
        fingerprint="a",
        symbol="000333",
        headline="官方快报",
        catalyst="earnings_beat",
        tags=["事实"],
        source_grade=1,
        items=[],
    )
    rating = ClusteredNews(
        fingerprint="b",
        symbol="000333",
        headline="券商观点",
        catalyst="rating_up",
        tags=["市场解读"],
        source_grade=3,
        items=[],
    )
    assert grade_weight(official) > grade_weight(rating)


def test_run_news_degrades_without_llm_and_complete_fields(tmp_kb, examples_dir, monkeypatch) -> None:
    monkeypatch.delenv("ZHIKU_LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ZHIKU_ANALYSIS_LLM", "0")
    assert llm_enabled() is False
    paths = Paths.load(kb=tmp_kb, fixtures=examples_dir / "fixtures")
    bundle = run_news(paths, date(2026, 10, 20), provider_kind="fixture")
    assert bundle.analyses
    assert bundle.analysis_engine == "rules"
    assert any(DEGRADED_MARK in n or "未做深度分析" in n for n in bundle.notes + [a.get("mark", "") for a in bundle.analyses])
    for card in bundle.analyses:
        for field in ANALYSIS_FIELDS:
            assert card.get(field) not in (None, ""), field
        kinds = {c["kind"] for c in card.get("claims") or []}
        assert {"事实", "解读", "推测"} <= kinds
        assert card.get("source_grade_label")
        assert card.get("subject") != card.get("so_what")
        assert "未做深度分析" in (card.get("mark") or "") or card.get("deep") is False
    rating_cards = [a for a in bundle.analyses if a.get("source_grade") == 3 or "目标价" in (a.get("subject") or "")]
    assert rating_cards
    review = rating_cards[0].get("institution_review") or {}
    for field in REVIEW_FIELDS:
        assert review.get(field), field
    assert bundle.low_value_count >= 1
    grades = {c["source_grade"] for c in bundle.clusters if c.get("source_grade")}
    assert 1 in grades and 3 in grades and 4 in grades
    leak = next(c for c in bundle.clusters if c.get("unofficial"))
    assert leak.get("official_followup") in {"pending", "confirmed", "denied"}
    settings = load_pipeline_settings()
    assert settings.news.sources.official_rss is True


def test_empty_qualified_set_prints_wu_in_template() -> None:
    from zhiku.notify.render import render
    from zhiku.news.runner import NewsBundle

    md = render(
        "premarket_ah.md.j2",
        news=NewsBundle(asof="2026-10-20", kind="premarket_ah"),
        disclaimer="仅为机会与逻辑，不构成投资建议",
    )
    assert "无" in md
    assert "入选分析" in md


def test_is_important_skips_pure_social() -> None:
    hot = ClusteredNews(
        fingerprint="h",
        symbol="600519",
        headline="人气榜",
        catalyst="sentiment_hot",
        tags=["情绪", "未经证实"],
        groups=["其它"],
        source_grade=4,
        items=[],
    )
    assert is_important(hot) is False
