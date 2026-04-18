"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from collection_system.core.models import Query


class AgentState(TypedDict):
    # Run context
    topic: str
    run_id: str
    max_depth: int
    max_queries: int
    relevance_threshold: float

    # Expansion progress
    current_depth: int
    pending_expansion: list[Query]   # queue for ExpandNode
    all_queries: list[Query]          # accumulated across all depths

    # Post-flatten
    flat_queries: list[Query]

    # Scoring (fan-out batches)
    scored_batches: list[list[float]]

    # Final output
    ranked_queries: list[Query]

    # Error propagation
    error: str | None
