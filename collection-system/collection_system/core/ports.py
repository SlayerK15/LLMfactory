"""
Port contracts — Protocol interfaces for every swappable adapter.
Core logic imports only from here, never from adapters/.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from collection_system.core.constants import RunStatus, Stage
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


class RateLimit:
    def __init__(self, requests: int, per_seconds: float) -> None:
        self.requests = requests
        self.per_seconds = per_seconds

    def __repr__(self) -> str:
        return f"RateLimit({self.requests}/{self.per_seconds}s)"


@runtime_checkable
class LLMPort(Protocol):
    """Query generation and relevance scoring."""

    async def expand_topic(
        self,
        topic: str,
        parent: str | None,
        depth: int,
    ) -> list[str]:
        """Expand a topic into sub-queries. Returns list of query strings."""
        ...

    async def score_relevance(
        self,
        queries: list[str],
        topic: str,
    ) -> list[float]:
        """Score each query 0.0–1.0 against the root topic."""
        ...

    async def validate_urls(
        self,
        topic: str,
        query: str,
        items: list[tuple[str, str | None, str | None]],
    ) -> list[bool]:
        """
        Decide whether each (url, title, snippet) candidate is worth scraping
        for the given topic/query. Returns a list[bool] of the same length —
        True = keep, False = drop. Must be tolerant of malformed LLM output
        (fall back to keep-all rather than failing the stage).
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the LLM provider is reachable."""
        ...


@runtime_checkable
class SearchPort(Protocol):
    """URL discovery from a search backend."""

    @property
    def name(self) -> str:
        """Human-readable backend name (e.g. 'cc_cdx', 'searxng')."""
        ...

    @property
    def rate_limit(self) -> RateLimit:
        """Configured rate limit for this backend."""
        ...

    async def discover_urls(
        self,
        query: Query,
        limit: int = 20,
    ) -> list[DiscoveredURL]:
        """Return up to `limit` URLs for the given query."""
        ...

    async def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        ...


@runtime_checkable
class ScraperPort(Protocol):
    """Web page scraping."""

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        """Scrape a single URL. Returns ScrapedDoc on success, Failure on error."""
        ...

    async def scrape_batch(
        self,
        urls: list[DiscoveredURL],
        concurrency: int,
    ) -> AsyncIterator[ScrapedDoc | Failure]:
        """
        Scrape a batch of URLs with bounded concurrency.
        Yields results as they complete — do not wait for all to finish.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the scraper is ready."""
        ...


@runtime_checkable
class FilesystemPort(Protocol):
    """Raw-content + artefact persistence on the local filesystem."""

    async def write_doc(self, doc: "ScrapedDoc") -> object: ...
    async def write_manifest(self, manifest: "RunManifest") -> None: ...
    async def write_metrics(self, run_id: str, metrics: dict) -> None: ...
    async def read_doc(self, run_id: str, doc_id: str) -> "ScrapedDoc": ...
    def ensure_run_dirs(self, run_id: str) -> None: ...


@runtime_checkable
class StoragePort(Protocol):
    """Persistence layer for all pipeline data."""

    # --- writes ---
    async def save_run(self, config: RunConfig) -> None: ...
    async def save_query(self, query: Query) -> None: ...
    async def save_url(self, url: DiscoveredURL) -> None: ...
    async def save_doc(self, doc: ScrapedDoc) -> None: ...
    async def save_failure(self, failure: Failure) -> None: ...
    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        error_msg: str | None = None,
    ) -> None: ...
    async def save_stage_stats(self, run_id: str, stats: StageStats) -> None: ...

    # --- reads ---
    async def load_run(self, run_id: str) -> RunManifest: ...
    async def list_runs(self, limit: int = 50) -> list[RunSummary]: ...
    async def load_docs(self, run_id: str) -> AsyncIterator[ScrapedDoc]: ...
    async def load_failures(self, run_id: str) -> list[Failure]: ...
    async def run_exists(self, run_id: str) -> bool: ...

    # --- dedup helpers ---
    async def url_seen(self, run_id: str, url_hash: str) -> bool: ...
    async def content_hash_seen(self, run_id: str, content_hash: str) -> bool: ...
