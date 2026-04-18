"""Integration test: real SearXNG query. Requires local SearXNG running."""
from __future__ import annotations

import os

import httpx
import pytest

from collection_system.adapters.search.searxng import SearXNGAdapter
from collection_system.core.models import Query

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")


def _searxng_up() -> bool:
    try:
        resp = httpx.get(SEARXNG_URL, timeout=2.0)
        return resp.status_code < 500
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _searxng_up(), reason="SearXNG not running")
@pytest.mark.asyncio
async def test_searxng_returns_urls():
    adapter = SearXNGAdapter(base_url=SEARXNG_URL, rate_per_second=2.0)
    try:
        query = Query(run_id="it-searxng", text="python asyncio tutorial", depth=0)
        urls = await adapter.discover_urls(query, limit=5)
        assert len(urls) >= 1
        for u in urls:
            assert u.url.startswith(("http://", "https://"))
            assert u.source_backend.value == "SEARXNG"
    finally:
        await adapter.aclose()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _searxng_up(), reason="SearXNG not running")
@pytest.mark.asyncio
async def test_searxng_health_check():
    adapter = SearXNGAdapter(base_url=SEARXNG_URL)
    try:
        assert await adapter.health_check() is True
    finally:
        await adapter.aclose()
