"""
LangGraph StateGraph for the query generation agent.

Graph flow:
  START → expand (loops) → flatten → score → aggregate → filter → END
"""
from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from collection_system.agents.nodes import (
    expand_node,
    filter_node,
    flatten_node,
    score_aggregate_node,
    score_node,
)
from collection_system.agents.state import AgentState
from collection_system.core.models import Query
from collection_system.core.ports import LLMPort

log = structlog.get_logger()


def _should_continue_expanding(state: AgentState) -> str:
    """Conditional edge: keep expanding or move to flatten."""
    if (
        state["current_depth"] >= state["max_depth"]
        or len(state["all_queries"]) >= state["max_queries"]
        or not state["pending_expansion"]
    ):
        return "flatten"
    return "expand"


def build_query_graph(llm: LLMPort):
    """
    Build and compile the query generation graph.
    llm is injected via closure so the graph is testable with fake adapters.
    Returns a compiled LangGraph that can be invoked with ainvoke().
    """

    async def _expand(state: AgentState) -> dict:
        return await expand_node(state, llm)

    async def _score(state: AgentState) -> dict:
        return await score_node(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("expand", _expand)
    graph.add_node("flatten", flatten_node)
    graph.add_node("score", _score)
    graph.add_node("aggregate", score_aggregate_node)
    graph.add_node("filter", filter_node)

    graph.add_edge(START, "expand")
    graph.add_conditional_edges(
        "expand",
        _should_continue_expanding,
        {"expand": "expand", "flatten": "flatten"},
    )
    graph.add_edge("flatten", "score")
    graph.add_edge("score", "aggregate")
    graph.add_edge("aggregate", "filter")
    graph.add_edge("filter", END)

    return graph.compile()


async def run_query_agent(
    topic: str,
    run_id: str,
    llm: LLMPort,
    max_depth: int = 3,
    max_queries: int = 600,
    relevance_threshold: float = 0.5,
) -> list[Query]:
    """
    Run the full query expansion + scoring + filtering pipeline.
    Returns list[Query] sorted by relevance_score descending.
    """
    compiled = build_query_graph(llm)

    root = Query(run_id=run_id, text=topic, parent_id=None, depth=0, source="root")

    initial_state: AgentState = {
        "topic": topic,
        "run_id": run_id,
        "max_depth": max_depth,
        "max_queries": max_queries,
        "relevance_threshold": relevance_threshold,
        "current_depth": 0,
        "pending_expansion": [root],
        "all_queries": [root],
        "flat_queries": [],
        "scored_batches": [],
        "ranked_queries": [],
        "error": None,
    }

    log.info("run_query_agent.start", topic=topic, run_id=run_id, max_depth=max_depth)
    result = await compiled.ainvoke(initial_state)
    ranked: list[Query] = result["ranked_queries"]
    log.info("run_query_agent.done", topic=topic, queries=len(ranked))
    return ranked
