"""Unit tests for CompositeSearchAdapter — dedup, fan-out, circuit-breaker skips."""
from __future__ import annotations

from uuid import uuid4

import pytest

from collection_system.adapters.search.composite import CompositeSearchAdapter
from collection_system.core.constants import SearchBackend
from collection_system.core.errors import CircuitOpenError
from collection_system.core.models import DiscoveredURL, Query
from collection_system.core.ports import RateLimit
from collection_system.infra.circuit_breaker import CircuitBreakerRegistry
from collection_system.infra.rate_limiter import RateLimiterRegistry


# ---------------------------------------------------------------------------
# Fixtures / stub backends
# ---------------------------------------------------------------------------


class StubBackend:
    """Minimal SearchPort stub that returns a canned URL list."""

    def __init__(
        self,
        name: str,
        urls: list[str],
        *,
        backend: SearchBackend = SearchBackend.CC_CDX,
        raise_exc: Exception | None = None,
        rate: RateLimit | None = None,
    ) -> None:
        self._name = name
        self._urls = urls
        self._backend = backend
        self._raise = raise_exc
        self._rate = rate or RateLimit(requests=100, per_seconds=1.0)
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def rate_limit(self) -> RateLimit:
        return self._rate

    async def discover_urls(self, query: Query, limit: int = 20) -> list[DiscoveredURL]:
        self.call_count += 1
        if self._raise is not None:
            raise self._raise
        return [
            DiscoveredURL(
                run_id=query.run_id,
                query_id=query.id,
                url=u,
                domain=u.split("/")[2] if "//" in u else u,
                source_backend=self._backend,
            )
            for u in self._urls[:limit]
        ]

    async def health_check(self) -> bool:
        return self._raise is None


@pytest.fixture
def query() -> Query:
    return Query(run_id=str(uuid4()), text="devops pipelines", depth=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_fans_out_to_all_backends(query: Query) -> None:
    b1 = StubBackend("b1", ["https://a.com/x", "https://a.com/y"])
    b2 = StubBackend("b2", ["https://b.com/x", "https://b.com/y"])
    composite = CompositeSearchAdapter(backends=[b1, b2])

    results = await composite.discover_urls(query, limit=10)

    assert b1.call_count == 1
    assert b2.call_count == 1
    assert len(results) == 4


@pytest.mark.asyncio
async def test_composite_dedups_overlapping_urls(query: Query) -> None:
    """Same URL returned by two backends counts once."""
    b1 = StubBackend("b1", ["https://shared.com/x", "https://a.com/x"])
    b2 = StubBackend("b2", ["https://shared.com/x", "https://b.com/y"])
    composite = CompositeSearchAdapter(backends=[b1, b2])

    results = await composite.discover_urls(query, limit=10)
    urls = [d.url for d in results]

    assert urls.count("https://shared.com/x") == 1
    assert len(results) == 3


@pytest.mark.asyncio
async def test_composite_respects_global_limit(query: Query) -> None:
    b1 = StubBackend("b1", [f"https://a.com/{i}" for i in range(20)])
    b2 = StubBackend("b2", [f"https://b.com/{i}" for i in range(20)])
    composite = CompositeSearchAdapter(backends=[b1, b2])

    results = await composite.discover_urls(query, limit=5)

    assert len(results) == 5


@pytest.mark.asyncio
async def test_composite_tolerates_one_backend_failure(query: Query) -> None:
    """One backend exploding does not fail the whole call."""
    b_ok = StubBackend("ok", ["https://ok.com/x", "https://ok.com/y"])
    b_bad = StubBackend("bad", [], raise_exc=RuntimeError("boom"))
    composite = CompositeSearchAdapter(backends=[b_ok, b_bad])

    results = await composite.discover_urls(query, limit=10)

    assert len(results) == 2
    assert all(d.url.startswith("https://ok.com") for d in results)


@pytest.mark.asyncio
async def test_composite_returns_empty_when_no_backends(query: Query) -> None:
    composite = CompositeSearchAdapter(backends=[])
    results = await composite.discover_urls(query, limit=10)
    assert results == []


@pytest.mark.asyncio
async def test_composite_skips_open_circuit(query: Query) -> None:
    """If a breaker is open for a backend, that backend is not called."""
    b1 = StubBackend("b1", ["https://a.com/x"])
    b2 = StubBackend("b2", ["https://b.com/x"])

    breakers = CircuitBreakerRegistry()
    breakers.register("b1")
    breakers.register("b2")
    # Force b1's breaker open by tripping it past fail_max
    cb = breakers.get("b1")
    assert cb is not None
    for _ in range(10):
        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("trip")))
        except Exception:
            pass
    assert cb.current_state == "open"

    composite = CompositeSearchAdapter(backends=[b1, b2], breakers=breakers)
    results = await composite.discover_urls(query, limit=10)

    # b1 must have been skipped; b2 served the request
    assert b1.call_count == 0
    assert b2.call_count == 1
    assert len(results) == 1
    assert results[0].url == "https://b.com/x"


@pytest.mark.asyncio
async def test_composite_auto_registers_rate_limiters_and_breakers() -> None:
    """Ctor should ensure every backend has a limiter + breaker entry."""
    b1 = StubBackend("alpha", [], rate=RateLimit(requests=5, per_seconds=2.0))
    b2 = StubBackend("beta", [])

    rates = RateLimiterRegistry()
    breakers = CircuitBreakerRegistry()
    assert rates.get("alpha") is None
    assert breakers.get("alpha") is None

    CompositeSearchAdapter(backends=[b1, b2], rate_limiters=rates, breakers=breakers)

    assert rates.get("alpha") is not None
    assert rates.get("beta") is not None
    assert breakers.get("alpha") is not None
    assert breakers.get("beta") is not None


@pytest.mark.asyncio
async def test_composite_health_check_true_if_any_backend_healthy(query: Query) -> None:
    b_ok = StubBackend("ok", ["https://ok.com/x"])
    b_bad = StubBackend("bad", [], raise_exc=RuntimeError("down"))
    composite = CompositeSearchAdapter(backends=[b_ok, b_bad])

    assert await composite.health_check() is True


@pytest.mark.asyncio
async def test_composite_health_check_false_if_all_backends_down() -> None:
    b_bad1 = StubBackend("b1", [], raise_exc=RuntimeError("down"))
    b_bad2 = StubBackend("b2", [], raise_exc=RuntimeError("down"))
    composite = CompositeSearchAdapter(backends=[b_bad1, b_bad2])

    # StubBackend.health_check returns False when raise_exc is set
    assert await composite.health_check() is False
