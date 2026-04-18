"""LLM adapter that fails over from primary to fallback providers."""
from __future__ import annotations

import structlog

from collection_system.core.errors import LLMRateLimitError
from collection_system.core.ports import LLMPort

log = structlog.get_logger()


class FailoverLLMAdapter:
    """Use primary provider, but switch to fallback when rate-limited."""

    def __init__(self, primary: LLMPort, fallback: LLMPort) -> None:
        self._primary = primary
        self._fallback = fallback

    async def expand_topic(self, topic: str, parent: str | None, depth: int) -> list[str]:
        try:
            return await self._primary.expand_topic(topic=topic, parent=parent, depth=depth)
        except LLMRateLimitError:
            log.warning("llm.failover", operation="expand_topic", fallback="enabled")
            return await self._fallback.expand_topic(topic=topic, parent=parent, depth=depth)

    async def score_relevance(self, queries: list[str], topic: str) -> list[float]:
        try:
            return await self._primary.score_relevance(queries=queries, topic=topic)
        except LLMRateLimitError:
            log.warning("llm.failover", operation="score_relevance", fallback="enabled")
            return await self._fallback.score_relevance(queries=queries, topic=topic)

    async def health_check(self) -> bool:
        primary_ok = await self._primary.health_check()
        if primary_ok:
            return True
        return await self._fallback.health_check()
