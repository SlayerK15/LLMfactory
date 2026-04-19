from __future__ import annotations

from collection_system.adapters.search.cc_cdx import _url_relevance_score


def test_url_relevance_score_prefers_slug_and_keywords() -> None:
    high = _url_relevance_score(
        "https://example.com/blog/ai-agents-design-patterns",
        slug="ai-agents-design-patterns",
        keywords=["agents", "design", "patterns"],
    )
    low = _url_relevance_score(
        "https://example.com/",
        slug="ai-agents-design-patterns",
        keywords=["agents", "design", "patterns"],
    )
    assert high > low


def test_url_relevance_score_zero_for_unrelated_url() -> None:
    score = _url_relevance_score(
        "https://example.com/about",
        slug="kubernetes-observability",
        keywords=["kubernetes", "observability"],
    )
    assert score == 0
