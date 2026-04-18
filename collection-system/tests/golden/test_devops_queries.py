"""Golden test: topic 'DevOps' should produce diverse, relevant queries."""
from __future__ import annotations

import os

import pytest

_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
_skip = pytest.mark.skipif(not _GROQ_KEY, reason="GROQ_API_KEY not set")

_EXPECTED_THEMES = [
    "CI",
    "CD",
    "pipeline",
    "container",
    "Docker",
    "Kubernetes",
    "monitoring",
    "infrastructure",
    "terraform",
    "ansible",
    "deployment",
    "git",
    "automation",
]


@_skip
@pytest.mark.slow
@pytest.mark.asyncio
async def test_devops_query_count():
    """
    'DevOps' with max_depth=2 and max_queries=100 should produce >= 50 queries
    via real Groq LLM after relevance filtering at threshold=0.4.
    """
    from collection_system.adapters.llm.groq_adapter import GroqAdapter
    from collection_system.agents.graph import run_query_agent

    llm = GroqAdapter(api_key=_GROQ_KEY)
    queries = await run_query_agent(
        topic="DevOps",
        run_id="golden-test-count",
        llm=llm,
        max_depth=2,
        max_queries=100,
        relevance_threshold=0.4,
    )

    assert len(queries) >= 50, f"Expected >= 50 queries, got {len(queries)}"


@_skip
@pytest.mark.slow
@pytest.mark.asyncio
async def test_devops_query_diversity():
    """
    Queries should cover at least 6 distinct DevOps themes from the expected set.
    Checks that the query expansion is genuinely diverse, not just paraphrases.
    """
    from collection_system.adapters.llm.groq_adapter import GroqAdapter
    from collection_system.agents.graph import run_query_agent

    llm = GroqAdapter(api_key=_GROQ_KEY)
    queries = await run_query_agent(
        topic="DevOps",
        run_id="golden-test-diversity",
        llm=llm,
        max_depth=2,
        max_queries=150,
        relevance_threshold=0.3,
    )

    all_text = " ".join(q.text.lower() for q in queries)
    covered = [theme for theme in _EXPECTED_THEMES if theme.lower() in all_text]

    assert len(covered) >= 6, (
        f"Expected >= 6 themes covered, got {len(covered)}: {covered}\n"
        f"Total queries: {len(queries)}"
    )
