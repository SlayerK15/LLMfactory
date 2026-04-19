from __future__ import annotations

import pytest

from collection_system.adapters.scraper.resilient import ResilientScraperAdapter
from collection_system.core.constants import SearchBackend, Stage
from collection_system.core.models import DiscoveredURL, Failure, ScrapedDoc


class _AlwaysFailScraper:
    name = "primary"

    async def scrape(self, url: DiscoveredURL) -> Failure:
        return Failure(
            run_id=url.run_id,
            stage=Stage.SCRAPE,
            target=url.url,
            error_type="ScrapeFailed",
            error_msg="primary failed",
        )

    async def health_check(self) -> bool:
        return False


class _AlwaysPassScraper:
    name = "fallback"

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc:
        return ScrapedDoc(
            run_id=url.run_id,
            url_id=url.id,
            url=url.url,
            markdown="# recovered",
            content_hash=ScrapedDoc.compute_content_hash(url.url),
            token_count=10,
            extraction_confidence=0.6,
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def sample_url() -> DiscoveredURL:
    return DiscoveredURL(
        run_id="run-1",
        query_id="q-1",
        url="https://example.com/a",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
    )


@pytest.mark.asyncio
async def test_resilient_scraper_uses_fallback_on_primary_failure(sample_url):
    adapter = ResilientScraperAdapter(_AlwaysFailScraper(), _AlwaysPassScraper())

    result = await adapter.scrape(sample_url)

    assert isinstance(result, ScrapedDoc)
    assert result.extraction_confidence == 0.6


@pytest.mark.asyncio
async def test_resilient_scraper_healthcheck_passes_if_either_backend_works(sample_url):
    adapter = ResilientScraperAdapter(_AlwaysFailScraper(), _AlwaysPassScraper())

    ok = await adapter.health_check()

    assert ok is True
