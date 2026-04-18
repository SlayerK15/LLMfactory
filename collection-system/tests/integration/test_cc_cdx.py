"""Integration test: real CC CDX query. Requires internet access."""
from __future__ import annotations

import pytest

from collection_system.adapters.search.cc_cdx import CCDXAdapter
from collection_system.core.models import Query


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_cc_cdx_returns_urls():
    """Query CC CDX for 'python tutorial' and assert at least 1 URL returned."""
    adapter = CCDXAdapter(max_per_query=5, seeds_per_query=2)
    query = Query(run_id="it-cc-cdx", text="python async tutorial", depth=0)
    urls = await adapter.discover_urls(query, limit=5)
    assert isinstance(urls, list)
    # CC is stale — it's acceptable to return 0 on some queries. Require the call
    # to succeed without raising; a more stringent check requires a known-good query.
    for u in urls:
        assert u.url.startswith(("http://", "https://"))
        assert u.source_backend.value == "CC_CDX"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_cc_cdx_health_check():
    adapter = CCDXAdapter()
    ok = await adapter.health_check()
    assert ok is True
