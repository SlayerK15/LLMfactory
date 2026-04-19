"""
LangGraph node functions.
Each node receives AgentState and returns a partial state update dict.
LLM-dependent nodes take an explicit `llm` param; graph.py wraps them in closures.
"""
from __future__ import annotations

import asyncio

import structlog

from collection_system.agents.state import AgentState
from collection_system.core.models import Query
from collection_system.core.ports import LLMPort

log = structlog.get_logger()

SCORE_BATCH_SIZE = 20
EXPAND_CONCURRENCY = 16


async def expand_node(state: AgentState, llm: LLMPort) -> dict:
    """
    Expand all pending_expansion queries by one depth level.
    Stops early if max_queries would be exceeded.
    """
    pending = state["pending_expansion"]
    depth = state["current_depth"]
    topic = state["topic"]
    max_queries = state["max_queries"]
    all_queries = list(state["all_queries"])
    remaining_slots = max(0, max_queries - len(all_queries))
    parents_to_expand = pending[:remaining_slots]
    sem = asyncio.Semaphore(EXPAND_CONCURRENCY)

    async def _expand_parent(parent_query: Query) -> list[str]:
        async with sem:
            return await llm.expand_topic(
                topic=topic,
                parent=parent_query.text,
                depth=depth,
            )

    new_queries: list[Query] = []
    child_batches = await asyncio.gather(
        *[_expand_parent(parent_query) for parent_query in parents_to_expand]
    )
    for parent_query, children_texts in zip(parents_to_expand, child_batches):
        for text in children_texts:
            if len(all_queries) + len(new_queries) >= max_queries:
                break
            new_queries.append(
                Query(
                    run_id=parent_query.run_id,
                    text=text,
                    parent_id=parent_query.id,
                    depth=depth + 1,
                    source="expansion",
                )
            )

    log.info("expand_node", depth=depth, new=len(new_queries), total=len(all_queries) + len(new_queries))
    return {
        "current_depth": depth + 1,
        "pending_expansion": new_queries,
        "all_queries": all_queries + new_queries,
    }


async def flatten_node(state: AgentState) -> dict:
    """Collect leaf queries (no children) from all_queries into flat_queries."""
    all_queries = state["all_queries"]
    parent_ids = {q.parent_id for q in all_queries if q.parent_id is not None}
    leaves = [q for q in all_queries if q.id not in parent_ids]
    log.info("flatten_node", total=len(all_queries), leaves=len(leaves))
    return {"flat_queries": leaves}


async def score_node(state: AgentState, llm: LLMPort) -> dict:
    """
    Score all flat_queries against the root topic in batches of SCORE_BATCH_SIZE.
    Returns scored_batches: list of float lists, one per batch.
    """
    flat = state["flat_queries"]
    topic = state["topic"]
    scored_batches: list[list[float]] = []

    for i in range(0, len(flat), SCORE_BATCH_SIZE):
        batch = flat[i : i + SCORE_BATCH_SIZE]
        scores = await llm.score_relevance([q.text for q in batch], topic)
        scored_batches.append(scores)
        log.debug("score_node batch", batch=i // SCORE_BATCH_SIZE, size=len(batch))

    return {"scored_batches": scored_batches}


async def score_aggregate_node(state: AgentState) -> dict:
    """Flatten scored_batches and write relevance_score back onto each flat query."""
    flat = state["flat_queries"]
    all_scores = [s for batch in state["scored_batches"] for s in batch]

    updated: list[Query] = []
    for i, q in enumerate(flat):
        score = all_scores[i] if i < len(all_scores) else 0.5
        updated.append(q.model_copy(update={"relevance_score": score}))

    log.info("score_aggregate_node", queries=len(updated))
    return {"flat_queries": updated}


async def filter_node(state: AgentState) -> dict:
    """Apply relevance_threshold and sort descending. Caps at max_queries."""
    flat = state["flat_queries"]
    threshold = state["relevance_threshold"]
    max_queries = state["max_queries"]

    filtered = [q for q in flat if q.relevance_score >= threshold]
    ranked = sorted(filtered, key=lambda q: q.relevance_score, reverse=True)[:max_queries]
    log.info("filter_node", before=len(flat), after=len(ranked), threshold=threshold)
    return {"ranked_queries": ranked}
