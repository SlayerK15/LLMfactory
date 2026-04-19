"""Unit tests for the LangGraph query agent."""
from __future__ import annotations

import asyncio
import time

import pytest

from collection_system.agents.nodes import expand_node
from collection_system.agents.graph import run_query_agent
from collection_system.core.models import Query


@pytest.mark.asyncio
async def test_agent_produces_queries_from_topic(fake_llm):
    """Running the agent with FakeLLM should produce >0 scored queries."""
    queries = await run_query_agent(
        topic="DevOps",
        run_id="test-run-1",
        llm=fake_llm,
        max_depth=1,
        max_queries=20,
        relevance_threshold=0.0,
    )
    assert len(queries) > 0
    for q in queries:
        assert q.run_id == "test-run-1"
        assert 0.0 <= q.relevance_score <= 1.0


@pytest.mark.asyncio
async def test_agent_respects_max_queries(fake_llm):
    """The agent must not return more than max_queries."""
    queries = await run_query_agent(
        topic="Kubernetes",
        run_id="test-run-2",
        llm=fake_llm,
        max_depth=3,
        max_queries=5,
        relevance_threshold=0.0,
    )
    assert len(queries) <= 5


@pytest.mark.asyncio
async def test_agent_applies_threshold(fake_llm):
    """With threshold=0.9 and FakeLLM returning 0.8, zero queries should pass."""
    queries = await run_query_agent(
        topic="Observability",
        run_id="test-run-3",
        llm=fake_llm,
        max_depth=1,
        max_queries=20,
        relevance_threshold=0.9,
    )
    assert queries == []


@pytest.mark.asyncio
async def test_agent_ranks_descending(fake_llm):
    """Output must be sorted by relevance_score desc."""
    queries = await run_query_agent(
        topic="SRE",
        run_id="test-run-4",
        llm=fake_llm,
        max_depth=2,
        max_queries=20,
        relevance_threshold=0.0,
    )
    scores = [q.relevance_score for q in queries]
    assert scores == sorted(scores, reverse=True)


class _TrackingLLM:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def expand_topic(self, topic: str, parent: str | None, depth: int) -> list[str]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.05)
            return [f"{parent} child 1", f"{parent} child 2"]
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_expand_node_parallelizes_parent_expansion():
    llm = _TrackingLLM()
    pending = [
        Query(run_id="run-1", text=f"parent {i}", depth=0, source="root")
        for i in range(4)
    ]
    state = {
        "topic": "DevOps",
        "run_id": "run-1",
        "max_depth": 2,
        "max_queries": 20,
        "relevance_threshold": 0.0,
        "current_depth": 0,
        "pending_expansion": pending,
        "all_queries": list(pending),
        "flat_queries": [],
        "scored_batches": [],
        "ranked_queries": [],
        "error": None,
    }

    started = time.perf_counter()
    result = await expand_node(state, llm)
    elapsed = time.perf_counter() - started

    assert llm.max_active > 1
    assert len(result["pending_expansion"]) == 8
    assert elapsed < 0.15
