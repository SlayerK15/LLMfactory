"""Integration test: real PostgreSQL CRUD. Requires DATABASE_URL env var."""
from __future__ import annotations

import os

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from collection_system.adapters.storage.orm import (
    DocRow,
    FailureRow,
    QueryRow,
    RunRow,
    StageStatsRow,
    URLRow,
)
from collection_system.adapters.storage.postgres import PostgresStorageAdapter
from collection_system.core.constants import RunStatus, SearchBackend, Stage
from collection_system.core.models import (
    DiscoveredURL,
    Failure,
    Query,
    RunConfig,
    ScrapedDoc,
    StageStats,
)
from collection_system.infra.db import Base, close_db, init_db

_DB_URL = os.getenv("DATABASE_URL", "")
_skip = pytest.mark.skipif(not _DB_URL, reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
async def _db_setup():
    """Create all tables before each test, drop after."""
    if not _DB_URL:
        yield
        return

    init_db(_DB_URL)
    engine = create_async_engine(_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    await close_db()


@pytest.fixture
def adapter() -> PostgresStorageAdapter:
    return PostgresStorageAdapter()


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(topic="DevOps", doc_count=10, max_queries=20, max_depth=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_and_load_run(adapter: PostgresStorageAdapter, run_config: RunConfig):
    """Save a run and load it back — round-trip integrity."""
    await adapter.save_run(run_config)

    manifest = await adapter.load_run(run_config.run_id)

    assert manifest.run_id == run_config.run_id
    assert manifest.config.topic == "DevOps"
    assert manifest.status == RunStatus.RUNNING


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_exists(adapter: PostgresStorageAdapter, run_config: RunConfig):
    await adapter.save_run(run_config)
    assert await adapter.run_exists(run_config.run_id) is True
    assert await adapter.run_exists("nonexistent-id") is False


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_and_list_runs(adapter: PostgresStorageAdapter, run_config: RunConfig):
    await adapter.save_run(run_config)
    summaries = await adapter.list_runs()
    assert len(summaries) == 1
    assert summaries[0].run_id == run_config.run_id
    assert summaries[0].topic == "DevOps"


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_run_status(adapter: PostgresStorageAdapter, run_config: RunConfig):
    await adapter.save_run(run_config)
    await adapter.update_run_status(run_config.run_id, RunStatus.COMPLETED)

    manifest = await adapter.load_run(run_config.run_id)
    assert manifest.status == RunStatus.COMPLETED
    assert manifest.completed_at is not None


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_query(adapter: PostgresStorageAdapter, run_config: RunConfig):
    await adapter.save_run(run_config)

    query = Query(run_id=run_config.run_id, text="CI/CD pipelines", depth=1)
    await adapter.save_query(query)

    manifest = await adapter.load_run(run_config.run_id)
    assert manifest.run_id == run_config.run_id  # run still intact


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_url_and_url_seen(adapter: PostgresStorageAdapter, run_config: RunConfig):
    await adapter.save_run(run_config)

    query = Query(run_id=run_config.run_id, text="test query", depth=1)
    await adapter.save_query(query)

    url = DiscoveredURL(
        run_id=run_config.run_id,
        query_id=query.id,
        url="https://example.com/devops",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
    )
    await adapter.save_url(url)

    assert await adapter.url_seen(run_config.run_id, url.url_hash) is True
    assert await adapter.url_seen(run_config.run_id, "nonexistent-hash") is False


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_url_dedup(adapter: PostgresStorageAdapter, run_config: RunConfig):
    """Saving the same URL twice should not raise — ON CONFLICT DO NOTHING."""
    await adapter.save_run(run_config)

    query = Query(run_id=run_config.run_id, text="test query", depth=1)
    await adapter.save_query(query)

    url = DiscoveredURL(
        run_id=run_config.run_id,
        query_id=query.id,
        url="https://example.com/devops",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
    )
    await adapter.save_url(url)
    await adapter.save_url(url)  # duplicate — must not raise


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_doc_and_content_hash_seen(
    adapter: PostgresStorageAdapter, run_config: RunConfig
):
    await adapter.save_run(run_config)

    query = Query(run_id=run_config.run_id, text="q", depth=1)
    await adapter.save_query(query)

    url = DiscoveredURL(
        run_id=run_config.run_id,
        query_id=query.id,
        url="https://example.com/doc",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
    )
    await adapter.save_url(url)

    text = "# DevOps guide\n\nContent here."
    doc = ScrapedDoc(
        run_id=run_config.run_id,
        url_id=url.id,
        url=url.url,
        markdown=text,
        content_hash=ScrapedDoc.compute_content_hash(text),
        token_count=20,
        path="/tmp/test.md",
    )
    await adapter.save_doc(doc)

    assert await adapter.content_hash_seen(run_config.run_id, doc.content_hash) is True
    assert await adapter.content_hash_seen(run_config.run_id, "bad-hash") is False


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_and_load_failures(
    adapter: PostgresStorageAdapter, run_config: RunConfig
):
    await adapter.save_run(run_config)

    failure = Failure(
        run_id=run_config.run_id,
        stage=Stage.SCRAPE,
        target="https://example.com/broken",
        error_type="TimeoutError",
        error_msg="30s exceeded",
        retries=3,
    )
    await adapter.save_failure(failure)

    failures = await adapter.load_failures(run_config.run_id)
    assert len(failures) == 1
    assert failures[0].error_type == "TimeoutError"
    assert failures[0].retries == 3


@_skip
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_stage_stats(adapter: PostgresStorageAdapter, run_config: RunConfig):
    from datetime import datetime, timezone

    await adapter.save_run(run_config)

    stats = StageStats(
        stage=Stage.SCRAPE,
        started_at=datetime.now(timezone.utc),
        input_count=100,
        output_count=95,
        failure_count=5,
    )
    await adapter.save_stage_stats(run_config.run_id, stats)

    manifest = await adapter.load_run(run_config.run_id)
    assert Stage.SCRAPE in manifest.stages
    assert manifest.stages[Stage.SCRAPE].input_count == 100
