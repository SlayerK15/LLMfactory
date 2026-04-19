"""Tests for the httpx fallback retry path in `_stage_scrape`."""
from __future__ import annotations

import pytest

from collection_system.core.constants import SearchBackend, Stage, URLStatus
from collection_system.core.models import (
    AdapterBundle,
    DiscoveredURL,
    Failure,
    RunConfig,
    RunManifest,
    ScrapedDoc,
)
from collection_system.core.pipeline import _stage_scrape

from tests.conftest import (
    FakeFallbackScraper,
    FakeFilesystem,
    FakeLLM,
    FakeScraper,
    FakeSearch,
    FakeStorage,
)


class _PrescriptedScraper(FakeScraper):
    """Scraper that returns a caller-controlled Failure for every URL."""

    def __init__(self, error_type: str) -> None:
        super().__init__()
        self._error_type = error_type

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        self.call_count += 1
        return Failure(
            run_id=url.run_id,
            stage=Stage.SCRAPE,
            target=url.url,
            error_type=self._error_type,
            error_msg="prescripted",
        )


class _PerUrlFailureFallback(FakeFallbackScraper):
    def __init__(self, error_type: str = "HTTPError", error_msg: str = "status 403") -> None:
        super().__init__()
        self._error_type = error_type
        self._error_msg = error_msg

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        self.call_count += 1
        return Failure(
            run_id=url.run_id,
            stage=Stage.SCRAPE,
            target=url.url,
            error_type=self._error_type,
            error_msg=self._error_msg,
        )


def _make_url(run_id: str, i: int) -> DiscoveredURL:
    return DiscoveredURL(
        run_id=run_id,
        query_id=f"q-{i}",
        url=f"https://example.com/page/{i}",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
        status=URLStatus.PENDING,
    )


def _bundle(scraper, fallback) -> AdapterBundle:
    return AdapterBundle(
        llm=FakeLLM(),
        search=FakeSearch(),
        scraper=scraper,
        storage=FakeStorage(),
        filesystem=FakeFilesystem(),
        fallback_scraper=fallback,
    )


def _bundle_with_curl(scraper, fallback, curl_fallback) -> AdapterBundle:
    return AdapterBundle(
        llm=FakeLLM(),
        search=FakeSearch(),
        scraper=scraper,
        storage=FakeStorage(),
        filesystem=FakeFilesystem(),
        fallback_scraper=fallback,
        curl_fallback_scraper=curl_fallback,
    )


@pytest.mark.asyncio
async def test_fallback_recovers_timeout_failure():
    """A TimeoutError from Crawl4AI must be retried via the fallback scraper."""
    config = RunConfig(topic="kubernetes", doc_count=10)
    urls = [_make_url(config.run_id, i) for i in range(3)]
    primary = _PrescriptedScraper(error_type="TimeoutError")
    fallback = FakeFallbackScraper()  # returns a ScrapedDoc
    adapters = _bundle(primary, fallback)
    manifest = RunManifest(run_id=config.run_id, config=config)

    docs, failures = await _stage_scrape(
        config, urls, adapters, manifest, sink=lambda _e: None
    )

    assert fallback.call_count == 3, "fallback must be attempted for each timeout"
    assert len(docs) == 3
    assert len(failures) == 0


@pytest.mark.asyncio
async def test_fallback_not_used_for_empty_content():
    """EmptyContent isn't a navigation-class error — no retry."""
    config = RunConfig(topic="kubernetes", doc_count=10)
    urls = [_make_url(config.run_id, i) for i in range(2)]
    primary = _PrescriptedScraper(error_type="EmptyContent")
    fallback = FakeFallbackScraper()
    adapters = _bundle(primary, fallback)
    manifest = RunManifest(run_id=config.run_id, config=config)

    docs, failures = await _stage_scrape(
        config, urls, adapters, manifest, sink=lambda _e: None
    )

    assert fallback.call_count == 0
    assert len(docs) == 0
    assert len(failures) == 2
    assert {f.error_type for f in failures} == {"EmptyContent"}


@pytest.mark.asyncio
async def test_fallback_double_failure_records_primary_error():
    """When the fallback ALSO fails, the recorded failure is the fallback's."""
    config = RunConfig(topic="kubernetes", doc_count=10)
    urls = [_make_url(config.run_id, 0)]
    primary = _PrescriptedScraper(error_type="RuntimeError")
    fallback_failure = Failure(
        run_id=config.run_id,
        stage=Stage.SCRAPE,
        target=urls[0].url,
        error_type="HTTPError",
        error_msg="status 403",
    )
    fallback = FakeFallbackScraper(response=fallback_failure)
    adapters = _bundle(primary, fallback)
    manifest = RunManifest(run_id=config.run_id, config=config)

    docs, failures = await _stage_scrape(
        config, urls, adapters, manifest, sink=lambda _e: None
    )

    assert fallback.call_count == 1
    assert len(docs) == 0
    assert len(failures) == 1
    # Pipeline keeps the original Crawl4AI failure when the fallback also
    # fails — its error message is more informative than the fallback's.
    assert failures[0].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_curl_fallback_recovers_http_failure():
    config = RunConfig(topic="kubernetes", doc_count=10)
    urls = [_make_url(config.run_id, i) for i in range(2)]
    primary = _PrescriptedScraper(error_type="RuntimeError")
    httpx_fallback = _PerUrlFailureFallback()
    curl_fallback = FakeFallbackScraper()
    adapters = _bundle_with_curl(primary, httpx_fallback, curl_fallback)
    manifest = RunManifest(run_id=config.run_id, config=config)

    docs, failures = await _stage_scrape(
        config, urls, adapters, manifest, sink=lambda _e: None
    )

    assert httpx_fallback.call_count == 2
    assert curl_fallback.call_count == 2
    assert len(docs) == 2
    assert len(failures) == 0


@pytest.mark.asyncio
async def test_curl_fallback_failure_keeps_original_error():
    config = RunConfig(topic="kubernetes", doc_count=10)
    urls = [_make_url(config.run_id, 0)]
    primary = _PrescriptedScraper(error_type="TimeoutError")
    httpx_fallback = _PerUrlFailureFallback()
    curl_fallback = FakeFallbackScraper(
        response=Failure(
            run_id=config.run_id,
            stage=Stage.SCRAPE,
            target=urls[0].url,
            error_type="TimeoutError",
            error_msg="still blocked",
        )
    )
    adapters = _bundle_with_curl(primary, httpx_fallback, curl_fallback)
    manifest = RunManifest(run_id=config.run_id, config=config)

    docs, failures = await _stage_scrape(
        config, urls, adapters, manifest, sink=lambda _e: None
    )

    assert len(docs) == 0
    assert len(failures) == 1
    assert failures[0].error_type == "TimeoutError"
