"""
End-to-end test: collection-system → cleaning-system.

Real stack: Groq LLM · Common Crawl CDX · Crawl4AI · PostgreSQL · filesystem.
SearXNG is intentionally excluded — CC CDX alone is enough for a test run.

Run with:
    cd "D:/Collection System"
    uv run pytest tests/e2e/ -v -s -m e2e
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_REQUIRED_VARS = ("GROQ_API_KEY", "DATABASE_URL")
_missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
if _missing:
    pytest.skip(
        f"Missing required env vars: {', '.join(_missing)}. "
        "Set them in D:/Collection System/.env before running e2e tests.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared data directory for the whole module — collection writes here, cleaning reads."""
    return tmp_path_factory.mktemp("e2e_data")


@pytest.fixture(scope="module")
def collection_run_id() -> list[str]:
    """Mutable container so collect_phase can pass run_id to clean_phase."""
    return []


@pytest.mark.asyncio
async def test_collection_phase(data_dir: Path, collection_run_id: list[str]):
    """
    Phase 1 — collect 25 docs about 'CI/CD pipelines' using:
      - Groq (llama-3.3-70b) for query generation
      - Common Crawl CDX for URL discovery (no SearXNG)
      - Crawl4AI for scraping
      - PostgreSQL for metadata
      - Filesystem for raw docs
    """
    from collection_system.core.constants import SearchBackend
    from collection_system.core.events import DocScraped, RunCompleted, RunFailed, StageCompleted
    from collection_system.core.models import RunConfig
    from collection_system.infra.config import override_settings
    from collection_system.api import run_collection

    override_settings(
        groq_api_key=os.environ["GROQ_API_KEY"],
        database_url=os.environ["DATABASE_URL"],
        data_dir=data_dir,
        max_depth=2,
        max_queries=40,
        scraper_concurrency=5,
        per_url_timeout_s=20,
        global_timeout_s=300,
    )

    config = RunConfig(
        topic="CI/CD pipelines",
        doc_count=25,
        max_queries=40,
        max_depth=2,
        scraper_concurrency=5,
        per_url_timeout_s=20,
        search_backends=[SearchBackend.CC_CDX],  # no SearXNG needed
    )

    print(f"\n[e2e] Starting collection run {config.run_id}")
    print(f"[e2e] Data dir: {data_dir}")

    handle = await run_collection(config)

    docs_scraped = 0
    stages_done = []
    async for event in handle.events:
        name = type(event).__name__
        if isinstance(event, DocScraped):
            docs_scraped += 1
            if docs_scraped % 5 == 0:
                print(f"[e2e] ... {docs_scraped} docs scraped")
        elif isinstance(event, StageCompleted):
            stages_done.append(event.stage)
            print(f"[e2e] Stage {event.stage} completed")
        elif isinstance(event, RunFailed):
            pytest.fail(f"Collection run failed: {event.error}")

    manifest = await handle.wait()
    collection_run_id.append(config.run_id)

    print(f"[e2e] Collection done — status={manifest.status.value}, docs={docs_scraped}")

    # Assertions
    assert manifest.status.value == "COMPLETED", (
        f"Run ended with status {manifest.status.value}. "
        f"Check {data_dir}/runs/{config.run_id}/manifest.json for details."
    )

    docs_dir = data_dir / "runs" / config.run_id / "docs"
    md_files = list(docs_dir.glob("*.md"))
    assert len(md_files) > 0, "No markdown files written to docs directory"
    print(f"[e2e] {len(md_files)} doc files on disk")


@pytest.mark.asyncio
async def test_cleaning_phase(data_dir: Path, collection_run_id: list[str]):
    """
    Phase 2 — clean the docs produced by test_collection_phase.
    Runs the fast stages only (near-dedup, lang filter, Gopher) since
    we don't want to download BGE-M3 / Qwen during an e2e test run.
    Enable via CLEAN_ENABLE_RELEVANCE=1 / CLEAN_ENABLE_PERPLEXITY=1 locally.
    """
    assert collection_run_id, "test_collection_phase must run first (check ordering)"
    run_id = collection_run_id[0]

    from cleaning_system.api import run_cleaning

    enable_relevance = os.environ.get("CLEAN_ENABLE_RELEVANCE", "0") == "1"
    enable_perplexity = os.environ.get("CLEAN_ENABLE_PERPLEXITY", "0") == "1"

    print(f"\n[e2e] Cleaning run {run_id}")
    print(f"[e2e] relevance={enable_relevance}  perplexity={enable_perplexity}")

    report = await run_cleaning(
        run_id=run_id,
        topic="CI/CD pipelines",
        data_dir=data_dir,
        enable_perplexity=enable_perplexity,
        enable_relevance=enable_relevance,
        enable_trafilatura=False,  # skip re-fetching during tests
    )

    print(f"[e2e] Cleaning report:")
    print(f"[e2e]   input_docs      = {report.input_docs}")
    print(f"[e2e]   after_near_dedup= {report.after_near_dedup}")
    print(f"[e2e]   after_lang      = {report.after_lang_filter}")
    print(f"[e2e]   after_gopher    = {report.after_gopher}")
    print(f"[e2e]   after_perplexity= {report.after_perplexity}")
    print(f"[e2e]   after_relevance = {report.after_relevance}")
    print(f"[e2e]   total_tokens    = {report.total_tokens:,}")
    print(f"[e2e]   drop_rate       = {report.drop_rate:.1%}")
    print(f"[e2e]   duration        = {report.cleaning_duration_s:.1f}s")

    # Assertions
    assert report.input_docs > 0, "Cleaning saw 0 input docs — collection may have failed"
    assert report.after_gopher > 0, "All docs dropped by Gopher — check filter thresholds"
    assert report.total_tokens > 0

    cleaned_dir = data_dir / "runs" / run_id / "cleaned"
    assert cleaned_dir.exists(), "Cleaned docs directory was not created"
    cleaned_files = list(cleaned_dir.glob("*.md"))
    assert len(cleaned_files) == report.after_relevance, (
        f"File count {len(cleaned_files)} != report count {report.after_relevance}"
    )

    report_file = data_dir / "reports" / f"{run_id}.json"
    assert report_file.exists(), "Quality report JSON not written"
    print(f"[e2e] Report saved to {report_file}")
