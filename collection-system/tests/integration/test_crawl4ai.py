"""Integration test: real Crawl4AI scrape. Requires Playwright installed."""
from __future__ import annotations

import pytest

from collection_system.adapters.scraper.crawl4ai_adapter import Crawl4AIAdapter
from collection_system.core.constants import SearchBackend, URLStatus
from collection_system.core.models import DiscoveredURL, Failure, ScrapedDoc


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_crawl4ai_scrapes_example_com():
    """Scrape example.com and assert markdown is returned."""
    url = DiscoveredURL(
        run_id="it-crawl4ai",
        query_id="q1",
        url="https://example.com",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
        status=URLStatus.PENDING,
    )
    async with Crawl4AIAdapter(concurrency=1, per_url_timeout_s=30) as adapter:
        result = await adapter.scrape(url)

    if isinstance(result, Failure):
        pytest.skip(f"Scrape failed in environment: {result.error_msg}")
    assert isinstance(result, ScrapedDoc)
    assert len(result.markdown) > 0
    assert result.content_hash
    assert result.token_count >= 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_crawl4ai_scrape_batch_yields_results():
    urls = [
        DiscoveredURL(
            run_id="it-crawl4ai-batch",
            query_id="q1",
            url="https://example.com",
            domain="example.com",
            source_backend=SearchBackend.CC_CDX,
        ),
        DiscoveredURL(
            run_id="it-crawl4ai-batch",
            query_id="q1",
            url="https://example.org",
            domain="example.org",
            source_backend=SearchBackend.CC_CDX,
        ),
    ]
    results = []
    async with Crawl4AIAdapter(concurrency=2, per_url_timeout_s=30) as adapter:
        async for r in adapter.scrape_batch(urls, concurrency=2):
            results.append(r)

    assert len(results) == 2
