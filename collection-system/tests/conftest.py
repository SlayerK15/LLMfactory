"""Shared fixtures and fake adapters for all test suites."""
from __future__ import annotations

import pytest

from collection_system.core.constants import SearchBackend
from collection_system.core.models import (
    DiscoveredURL,
    Failure,
    Query,
    RunConfig,
    RunManifest,
    RunSummary,
    ScrapedDoc,
    StageStats,
)
from collection_system.core.constants import RunStatus, Stage


# ---------------------------------------------------------------------------
# Fake adapters — implement ports with in-memory state, no I/O
# ---------------------------------------------------------------------------

class FakeLLM:
    """Returns deterministic responses for testing."""

    async def expand_topic(self, topic: str, parent: str | None, depth: int) -> list[str]:
        return [f"{topic} sub-topic {i}" for i in range(1, 4)]

    async def score_relevance(self, queries: list[str], topic: str) -> list[float]:
        return [0.8] * len(queries)

    async def validate_urls(
        self,
        topic: str,
        query: str,
        items: list[tuple[str, str | None, str | None]],
    ) -> list[bool]:
        return [True] * len(items)

    async def health_check(self) -> bool:
        return True


class FakeSearch:
    """Returns fake URLs without hitting any network."""

    def __init__(self, urls_per_query: int = 3) -> None:
        self.urls_per_query = urls_per_query
        self.call_count = 0

    @property
    def name(self) -> str:
        return "fake_search"

    @property
    def rate_limit(self):
        from collection_system.core.ports import RateLimit
        return RateLimit(requests=100, per_seconds=1.0)

    async def discover_urls(self, query: Query, limit: int = 20) -> list[DiscoveredURL]:
        self.call_count += 1
        return [
            DiscoveredURL(
                run_id=query.run_id,
                query_id=query.id,
                url=f"https://example.com/{query.id}/{i}",
                domain="example.com",
                source_backend=SearchBackend.CC_CDX,
            )
            for i in range(self.urls_per_query)
        ]

    async def health_check(self) -> bool:
        return True


class FakeScraper:
    """Returns fake scraped docs without hitting any network."""

    def __init__(self, fail_every: int | None = None) -> None:
        self.fail_every = fail_every
        self.call_count = 0

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        self.call_count += 1
        if self.fail_every and self.call_count % self.fail_every == 0:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="FakeError",
                error_msg="Intentional test failure",
            )
        return ScrapedDoc(
            run_id=url.run_id,
            url_id=url.id,
            url=url.url,
            markdown=f"# Fake doc for {url.url}\n\nContent here.",
            content_hash=ScrapedDoc.compute_content_hash(url.url),
            token_count=50,
        )

    async def scrape_batch(self, urls, concurrency):
        for url in urls:
            yield await self.scrape(url)

    async def health_check(self) -> bool:
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeFallbackScraper:
    """Returns a pre-baked response — used to exercise _stage_scrape's retry."""

    def __init__(self, response: ScrapedDoc | Failure | None = None) -> None:
        self._response = response
        self.call_count = 0

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        self.call_count += 1
        if self._response is not None:
            return self._response
        return ScrapedDoc(
            run_id=url.run_id,
            url_id=url.id,
            url=url.url,
            markdown=f"# fallback doc for {url.url}\n\n" + ("x " * 100),
            content_hash=ScrapedDoc.compute_content_hash(f"fallback-{url.url}"),
            token_count=100,
            extraction_confidence=0.6,
        )

    async def scrape_batch(self, urls, concurrency):
        for url in urls:
            yield await self.scrape(url)


class FakeFilesystem:
    """In-memory filesystem — no disk required."""

    def __init__(self) -> None:
        self.docs: dict[str, ScrapedDoc] = {}
        self.manifests: dict[str, RunManifest] = {}
        self.metrics: dict[str, dict] = {}

    async def write_doc(self, doc: ScrapedDoc):
        from pathlib import Path
        self.docs[doc.id] = doc
        return Path(f"/fake/{doc.run_id}/docs/{doc.id}.md")

    async def write_manifest(self, manifest: RunManifest) -> None:
        self.manifests[manifest.run_id] = manifest

    async def write_metrics(self, run_id: str, metrics: dict) -> None:
        self.metrics[run_id] = metrics

    async def read_doc(self, run_id: str, doc_id: str) -> ScrapedDoc:
        return self.docs[doc_id]

    def ensure_run_dirs(self, run_id: str) -> None:
        pass


class FakeStorage:
    """In-memory storage — no database required."""

    def __init__(self) -> None:
        self.runs: dict[str, RunConfig] = {}
        self.queries: list[Query] = []
        self.urls: list[DiscoveredURL] = []
        self.docs: list[ScrapedDoc] = []
        self.failures: list[Failure] = []
        self._url_hashes: set[str] = set()
        self._content_hashes: set[str] = set()

    async def save_run(self, config: RunConfig) -> None:
        self.runs[config.run_id] = config

    async def save_query(self, query: Query) -> None:
        self.queries.append(query)

    async def save_url(self, url: DiscoveredURL) -> None:
        self.urls.append(url)
        self._url_hashes.add(url.url_hash)

    async def save_doc(self, doc: ScrapedDoc) -> None:
        self.docs.append(doc)
        self._content_hashes.add(doc.content_hash)

    async def save_failure(self, failure: Failure) -> None:
        self.failures.append(failure)

    async def update_run_status(self, run_id, status, error_msg=None) -> None:
        pass

    async def save_stage_stats(self, run_id, stats) -> None:
        pass

    async def load_run(self, run_id: str) -> RunManifest:
        config = self.runs[run_id]
        return RunManifest(run_id=run_id, config=config)

    async def list_runs(self, limit: int = 50) -> list[RunSummary]:
        return []

    async def load_docs(self, run_id: str):
        for doc in self.docs:
            if doc.run_id == run_id:
                yield doc

    async def load_failures(self, run_id: str) -> list[Failure]:
        return [f for f in self.failures if f.run_id == run_id]

    async def run_exists(self, run_id: str) -> bool:
        return run_id in self.runs

    async def url_seen(self, run_id: str, url_hash: str) -> bool:
        return url_hash in self._url_hashes

    async def content_hash_seen(self, run_id: str, content_hash: str) -> bool:
        return content_hash in self._content_hashes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_search() -> FakeSearch:
    return FakeSearch()


@pytest.fixture
def fake_scraper() -> FakeScraper:
    return FakeScraper()


@pytest.fixture
def fake_storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def fake_filesystem() -> FakeFilesystem:
    return FakeFilesystem()


@pytest.fixture
def adapter_bundle(fake_llm, fake_search, fake_scraper, fake_storage, fake_filesystem):
    from collection_system.core.models import AdapterBundle

    return AdapterBundle(
        llm=fake_llm,
        search=fake_search,
        scraper=fake_scraper,
        storage=fake_storage,
        filesystem=fake_filesystem,
    )


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(topic="DevOps", doc_count=10, max_queries=20, max_depth=2)
