"""Domain models — all Pydantic, zero I/O."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


def _utcnow() -> datetime:
    """Timezone-aware UTC now. Replaces deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)

from collection_system.core.constants import (
    RunStatus,
    SearchBackend,
    Stage,
    URLStatus,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    doc_count: int = 2000
    max_depth: int = 3
    max_queries: int = 600
    relevance_threshold: float = 0.5
    search_backends: list[SearchBackend] = Field(
        default_factory=lambda: [SearchBackend.CC_CDX, SearchBackend.SEARXNG]
    )
    scraper_concurrency: int = 40
    per_url_timeout_s: int = 30
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Query tree
# ---------------------------------------------------------------------------

class Query(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    text: str
    parent_id: str | None = None
    depth: int = 0
    relevance_score: float = 0.0
    source: str = "expansion"  # "root" | "expansion"


# ---------------------------------------------------------------------------
# URL discovery
# ---------------------------------------------------------------------------

class DiscoveredURL(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    query_id: str
    url: str
    domain: str
    source_backend: SearchBackend
    status: URLStatus = URLStatus.PENDING
    discovered_at: datetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[misc]
    @property
    def url_hash(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Scraped document
# ---------------------------------------------------------------------------

class ScrapedDoc(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    url_id: str
    url: str
    title: str | None = None
    markdown: str
    content_hash: str
    token_count: int
    extraction_confidence: float = 1.0
    scraped_at: datetime = Field(default_factory=_utcnow)
    scrape_duration_ms: int = 0
    path: Path = Field(default=Path("."))

    @classmethod
    def compute_content_hash(cls, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------

class Failure(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    stage: Stage
    target: str
    error_type: str
    error_msg: str
    retries: int = 0
    failed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

class StageStats(BaseModel):
    stage: Stage
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    input_count: int = 0
    output_count: int = 0
    failure_count: int = 0

    @property
    def duration_s(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class RunManifest(BaseModel):
    run_id: str
    config: RunConfig
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    stages: dict[Stage, StageStats] = Field(default_factory=dict)
    error_msg: str | None = None

    @property
    def total_docs(self) -> int:
        stats = self.stages.get(Stage.SCRAPE)
        return stats.output_count if stats else 0


class RunSummary(BaseModel):
    run_id: str
    topic: str
    status: RunStatus
    docs_collected: int = 0
    started_at: datetime
    duration_s: float | None = None


# ---------------------------------------------------------------------------
# Adapter bundle (passed into pipeline)
# ---------------------------------------------------------------------------

class AdapterBundle(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    llm: object
    search: object
    scraper: object
    storage: object
    filesystem: object
