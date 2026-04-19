"""
Composite search adapter — fans out to all enabled backends and merges results.
Deduplicates by URL hash across backends.
Respects per-backend rate limits + circuit breakers.
"""
from __future__ import annotations

import asyncio
import re

import pybreaker
import structlog

from collection_system.adapters.search.base import url_hash
from collection_system.core.errors import CircuitOpenError, SearchError
from collection_system.core.models import DiscoveredURL, Query
from collection_system.core.ports import RateLimit, SearchPort
from collection_system.infra.circuit_breaker import CircuitBreakerRegistry
from collection_system.infra.rate_limiter import RateLimiterRegistry

log = structlog.get_logger()


class CompositeSearchAdapter:
    """
    Routes each query to all enabled SearchPort adapters in parallel.
    Results are merged and deduplicated by normalised URL hash.

    Rate limiting and circuit breaking are applied per backend — if one
    backend trips its circuit, the others continue serving requests.
    """

    def __init__(
        self,
        backends: list[SearchPort],
        rate_limiters: RateLimiterRegistry | None = None,
        breakers: CircuitBreakerRegistry | None = None,
    ) -> None:
        self._backends = backends
        self._rates = rate_limiters or RateLimiterRegistry()
        self._breakers = breakers or CircuitBreakerRegistry()

        # Auto-register rate limiters + breakers for each backend if absent.
        for b in backends:
            if self._rates.get(b.name) is None:
                rl: RateLimit = b.rate_limit
                self._rates.register(
                    b.name, rate=float(rl.requests), per=rl.per_seconds
                )
            if self._breakers.get(b.name) is None:
                self._breakers.register(b.name)

    @property
    def name(self) -> str:
        return "composite"

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=100, per_seconds=1.0)

    async def _call_one(
        self,
        backend: SearchPort,
        query: Query,
        per_backend_limit: int,
    ) -> list[DiscoveredURL]:
        breaker = self._breakers.get(backend.name)
        if breaker is not None and breaker.current_state == "open":
            log.debug("composite.circuit_open", backend=backend.name)
            raise CircuitOpenError(backend.name)

        await self._rates.acquire(backend.name)
        try:
            return await backend.discover_urls(query, per_backend_limit)
        except pybreaker.CircuitBreakerError as exc:
            raise CircuitOpenError(backend.name) from exc

    async def discover_urls(self, query: Query, limit: int = 20) -> list[DiscoveredURL]:
        if not self._backends:
            return []

        merged, failed_backends = await self._discover_once(query, limit)
        if merged:
            return merged

        if failed_backends == len(self._backends):
            raise SearchError(f"all search backends failed for query={query.text!r}")

        relaxed_text = self._relax_query(query.text)
        if relaxed_text and relaxed_text != query.text:
            relaxed_query = query.model_copy(update={"text": relaxed_text})
            merged, relaxed_failed_backends = await self._discover_once(
                relaxed_query, limit
            )
            if merged:
                log.info(
                    "composite.query_relaxed_recovered",
                    original=query.text,
                    relaxed=relaxed_text,
                    returned=len(merged),
                )
                return merged
            if relaxed_failed_backends == len(self._backends):
                raise SearchError(
                    "all search backends failed for both original and relaxed query"
                )
        return []

    async def _discover_once(
        self,
        query: Query,
        limit: int,
    ) -> tuple[list[DiscoveredURL], int]:
        # Request each backend slightly over fair-share so unique-after-dedup ≈ limit.
        per_backend = max(5, (limit * 2) // max(1, len(self._backends)))
        tasks = [self._call_one(backend, query, per_backend) for backend in self._backends]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        seen: set[str] = set()
        merged: list[DiscoveredURL] = []
        failed_backends = 0
        for backend, batch in zip(self._backends, batches, strict=False):
            if isinstance(batch, BaseException):
                failed_backends += 1
                log.warning(
                    "composite.backend_failed",
                    backend=backend.name,
                    error=str(batch),
                )
                continue
            for du in batch:
                h = url_hash(du.url)
                if h in seen:
                    continue
                seen.add(h)
                merged.append(du)
                if len(merged) >= limit:
                    break
            if len(merged) >= limit:
                break

        log.info(
            "composite.discover_urls",
            query=query.text,
            backends=len(self._backends),
            returned=len(merged),
            failed_backends=failed_backends,
        )
        return merged, failed_backends

    def _relax_query(self, text: str) -> str:
        """
        Build a broader fallback query from informative tokens.
        Example: "AI infrastructure observability patterns" -> "ai infrastructure observability".
        """
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        kept: list[str] = []
        for token in tokens:
            if len(token) <= 2:
                continue
            if token in {"with", "from", "into", "about", "guide", "tutorial"}:
                continue
            kept.append(token)
            if len(kept) == 3:
                break
        return " ".join(kept)

    async def health_check(self) -> bool:
        results = await asyncio.gather(
            *[b.health_check() for b in self._backends],
            return_exceptions=True,
        )
        return any(r is True for r in results)
