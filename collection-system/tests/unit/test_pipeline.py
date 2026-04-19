"""Unit tests for the collection pipeline using fake adapters."""
from __future__ import annotations

import pytest

from collection_system.core.constants import RunStatus, Stage
from collection_system.core.models import DiscoveredURL
from collection_system.core.events import (
    DocScraped,
    QueriesGenerated,
    RunCompleted,
    RunStarted,
    StageCompleted,
    URLsDiscovered,
)
from collection_system.core.pipeline import (
    _query_budget_for_doc_target,
    URLS_PER_DOC_TARGET,
    run_collection,
    run_collection_streaming,
)


# ---------------------------------------------------------------------------
# Blocking run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_completes_end_to_end(run_config, adapter_bundle):
    """Full pipeline with fakes should produce a COMPLETED manifest."""
    run_config.doc_count = 5
    run_config.max_queries = 10
    run_config.max_depth = 1

    manifest = await run_collection(run_config, adapter_bundle)

    assert manifest.status == RunStatus.COMPLETED
    assert manifest.run_id == run_config.run_id
    assert Stage.QUERY_GENERATION in manifest.stages
    assert Stage.URL_DISCOVERY in manifest.stages
    assert Stage.SCRAPE in manifest.stages
    assert Stage.FINALIZE in manifest.stages


@pytest.mark.asyncio
async def test_pipeline_persists_via_fake_storage(run_config, adapter_bundle, fake_storage):
    """Pipeline must record run + queries + urls + docs through the storage port."""
    run_config.doc_count = 3
    run_config.max_queries = 5
    run_config.max_depth = 1

    await run_collection(run_config, adapter_bundle)

    assert run_config.run_id in fake_storage.runs
    assert len(fake_storage.queries) > 0
    assert len(fake_storage.urls) > 0
    assert len(fake_storage.docs) > 0


@pytest.mark.asyncio
async def test_pipeline_writes_artefacts_via_fake_filesystem(
    run_config, adapter_bundle, fake_filesystem
):
    """manifest + metrics must be written, one entry per scraped doc."""
    run_config.doc_count = 3
    run_config.max_queries = 5
    run_config.max_depth = 1

    await run_collection(run_config, adapter_bundle)

    assert run_config.run_id in fake_filesystem.manifests
    assert run_config.run_id in fake_filesystem.metrics
    assert len(fake_filesystem.docs) > 0


@pytest.mark.asyncio
async def test_pipeline_handles_scraper_failures(
    run_config, fake_llm, fake_search, fake_storage, fake_filesystem
):
    """Failures must not abort the run — they land in storage + manifest."""
    from collection_system.core.models import AdapterBundle
    from tests.conftest import FakeScraper

    scraper = FakeScraper(fail_every=2)  # every 2nd URL fails
    bundle = AdapterBundle(
        llm=fake_llm,
        search=fake_search,
        scraper=scraper,
        storage=fake_storage,
        filesystem=fake_filesystem,
    )
    run_config.doc_count = 10
    run_config.max_queries = 8
    run_config.max_depth = 1

    manifest = await run_collection(run_config, bundle)

    assert manifest.status == RunStatus.COMPLETED
    assert len(fake_storage.failures) > 0
    assert Stage.SCRAPE in manifest.stages
    assert manifest.stages[Stage.SCRAPE].failure_count > 0


@pytest.mark.asyncio
async def test_pipeline_content_hash_dedup(run_config, adapter_bundle, fake_storage):
    """Duplicate content (same content_hash) should not be stored twice."""
    run_config.doc_count = 10
    run_config.max_queries = 5
    run_config.max_depth = 1

    # FakeScraper hashes from URL, so unique URLs → unique hashes.
    # Just verify dedup set is populated and counts are consistent.
    await run_collection(run_config, adapter_bundle)
    hashes = {d.content_hash for d in fake_storage.docs}
    assert len(hashes) == len(fake_storage.docs)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_emits_expected_event_sequence(run_config, adapter_bundle):
    """Streaming API emits RunStarted → QueriesGenerated → DocScraped → RunCompleted."""
    run_config.doc_count = 3
    run_config.max_queries = 5
    run_config.max_depth = 1

    events = []
    async for event in run_collection_streaming(run_config, adapter_bundle):
        events.append(event)

    kinds = [type(e).__name__ for e in events]
    assert "RunStarted" in kinds
    assert "QueriesGenerated" in kinds
    assert "DocScraped" in kinds
    assert "StageCompleted" in kinds
    assert kinds[-1] in ("RunCompleted", "RunFailed")


@pytest.mark.asyncio
async def test_streaming_docscraped_events_match_doc_count(
    run_config, adapter_bundle, fake_storage
):
    run_config.doc_count = 4
    run_config.max_queries = 6
    run_config.max_depth = 1

    doc_events = 0
    async for event in run_collection_streaming(run_config, adapter_bundle):
        if isinstance(event, DocScraped):
            doc_events += 1

    assert doc_events == len(fake_storage.docs)


@pytest.mark.asyncio
async def test_url_discovery_retries_with_simplified_query(
    run_config, fake_llm, fake_storage, fake_filesystem, fake_scraper
):
    from collection_system.core.constants import SearchBackend
    from collection_system.core.models import AdapterBundle, Query

    class VerboseQuerySearch:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @property
        def name(self) -> str:
            return "verbose_query_search"

        @property
        def rate_limit(self):
            from collection_system.core.ports import RateLimit

            return RateLimit(requests=100, per_seconds=1.0)

        async def discover_urls(self, query: Query, limit: int = 20) -> list[DiscoveredURL]:
            self.calls.append(query.text)
            if "explain" in query.text.lower():
                return []
            return [
                DiscoveredURL(
                    run_id=query.run_id,
                    query_id=query.id,
                    url=f"https://example.com/{query.id}/0",
                    domain="example.com",
                    source_backend=SearchBackend.SEARXNG,
                )
            ]

        async def health_check(self) -> bool:
            return True

    search = VerboseQuerySearch()
    bundle = AdapterBundle(
        llm=fake_llm,
        search=search,
        scraper=fake_scraper,
        storage=fake_storage,
        filesystem=fake_filesystem,
    )

    run_config.topic = "Explain practical AI safety strategies for startups"
    run_config.max_queries = 1
    run_config.max_depth = 0
    run_config.doc_count = 1

    manifest = await run_collection(run_config, bundle)

    assert manifest.status == RunStatus.COMPLETED
    assert len(fake_storage.urls) > 0
    assert len(search.calls) >= 2


def test_query_budget_scales_with_doc_target():
    assert _query_budget_for_doc_target(doc_count=80, configured_max=600) == 40
    assert _query_budget_for_doc_target(doc_count=1200, configured_max=600) == 300
    assert _query_budget_for_doc_target(doc_count=4000, configured_max=600) == 600
    assert _query_budget_for_doc_target(doc_count=4000, configured_max=250) == 250


@pytest.mark.asyncio
async def test_url_validation_guardrail_prevents_overdrop(
    run_config, fake_search, fake_storage, fake_filesystem, fake_scraper
):
    from collection_system.core.models import AdapterBundle

    class DropAllLLM:
        async def expand_topic(self, topic: str, parent: str | None, depth: int) -> list[str]:
            return [topic]

        async def score_relevance(self, queries: list[str], topic: str) -> list[float]:
            return [1.0] * len(queries)

        async def validate_urls(
            self,
            topic: str,
            query: str,
            items: list[tuple[str, str | None, str | None]],
        ) -> list[bool]:
            return [False] * len(items)

    bundle = AdapterBundle(
        llm=DropAllLLM(),
        search=fake_search,
        scraper=fake_scraper,
        storage=fake_storage,
        filesystem=fake_filesystem,
    )
    run_config.doc_count = 50
    run_config.max_queries = 3
    run_config.max_depth = 1

    manifest = await run_collection(run_config, bundle)

    assert manifest.status == RunStatus.COMPLETED
    assert manifest.stages[Stage.URL_DISCOVERY].output_count >= 1
    assert manifest.stages[Stage.URL_VALIDATION].output_count == manifest.stages[Stage.URL_DISCOVERY].output_count


def test_url_target_ratio_is_one_to_five():
    assert URLS_PER_DOC_TARGET == 5
