"""
Main pipeline orchestration.
Stages: QUERY_GENERATION → URL_DISCOVERY → SCRAPE → FINALIZE.

Design contract:
  - Core never imports from adapters/. Adapters are passed via AdapterBundle.
  - Single-URL failures never abort the run (written to failures, pipeline continues).
  - Per-stage timeouts + a global wall-clock guard are enforced at the boundary.
  - Checkpoints are written between stages and every CHECKPOINT_INTERVAL_DOCS.
  - Events are emitted synchronously via an optional event_sink for streaming.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Callable

import structlog

from collection_system.core.constants import (
    CHECKPOINT_INTERVAL_DOCS,
    RunStatus,
    Stage,
)
from collection_system.core.errors import PipelineTimeoutError
from collection_system.core.events import (
    CollectionEvent,
    DocFailed,
    DocScraped,
    QueriesGenerated,
    RunCompleted,
    RunFailed,
    RunStarted,
    StageCompleted,
    StageStarted,
    URLsDiscovered,
)
from collection_system.core.models import (
    AdapterBundle,
    DiscoveredURL,
    Failure,
    Query,
    RunConfig,
    RunManifest,
    ScrapedDoc,
    StageStats,
)

log = structlog.get_logger()

# Per-stage budgets (soft caps, checkpointed)
STAGE_BUDGETS_S: dict[Stage, int] = {
    Stage.QUERY_GENERATION: 360,
    Stage.URL_DISCOVERY: 600,
    Stage.SCRAPE: 1800,
    Stage.FINALIZE: 60,
}

EventSink = Callable[[CollectionEvent], None]


def _noop_sink(_event: CollectionEvent) -> None:  # default no-op
    pass


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_collection(
    config: RunConfig,
    adapters: AdapterBundle,
    event_sink: EventSink | None = None,
) -> RunManifest:
    """
    Blocking pipeline run. Returns the final RunManifest on completion.
    Pass `event_sink` to receive live events (mostly used by streaming wrapper).
    """
    sink = event_sink or _noop_sink
    started_at = datetime.now(timezone.utc)
    manifest = RunManifest(run_id=config.run_id, config=config, started_at=started_at)

    sink(RunStarted(run_id=config.run_id, config=config, at=started_at))

    try:
        await adapters.storage.save_run(config)
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.save_run_failed", run_id=config.run_id)
        manifest.status = RunStatus.FAILED
        manifest.error_msg = f"save_run failed: {exc}"
        sink(RunFailed(run_id=config.run_id, error=str(exc)))
        return manifest

    try:
        async with asyncio.timeout(int(getattr(config, "global_timeout_s", 1800))):
            queries = await _stage_query_generation(config, adapters, manifest, sink)
            urls = await _stage_url_discovery(config, queries, adapters, manifest, sink)
            docs, failures = await _stage_scrape(
                config, urls, adapters, manifest, sink
            )
            await _stage_finalize(
                config, docs, failures, adapters, started_at, manifest, sink
            )
    except PipelineTimeoutError as exc:
        manifest.status = RunStatus.FAILED
        manifest.error_msg = str(exc)
        await adapters.storage.update_run_status(
            config.run_id, RunStatus.FAILED, error_msg=str(exc)
        )
        sink(RunFailed(run_id=config.run_id, error=str(exc)))
        return manifest
    except asyncio.TimeoutError:
        msg = "Global pipeline timeout exceeded"
        manifest.status = RunStatus.FAILED
        manifest.error_msg = msg
        await adapters.storage.update_run_status(
            config.run_id, RunStatus.FAILED, error_msg=msg
        )
        sink(RunFailed(run_id=config.run_id, error=msg))
        return manifest
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline.unexpected_failure", run_id=config.run_id)
        manifest.status = RunStatus.FAILED
        manifest.error_msg = str(exc)
        await adapters.storage.update_run_status(
            config.run_id, RunStatus.FAILED, error_msg=str(exc)
        )
        sink(RunFailed(run_id=config.run_id, error=str(exc)))
        return manifest

    manifest.status = RunStatus.COMPLETED
    manifest.completed_at = datetime.now(timezone.utc)
    await adapters.storage.update_run_status(config.run_id, RunStatus.COMPLETED)

    duration = (manifest.completed_at - started_at).total_seconds()
    sink(
        RunCompleted(
            run_id=config.run_id,
            status=RunStatus.COMPLETED,
            docs_collected=manifest.total_docs,
            duration_s=duration,
        )
    )
    return manifest


async def run_collection_streaming(
    config: RunConfig,
    adapters: AdapterBundle,
) -> AsyncIterator[CollectionEvent]:
    """
    Streaming pipeline run. Yields CollectionEvent objects as work progresses.
    The orchestrator and frontend consume this iterator.
    """
    queue: asyncio.Queue[CollectionEvent | None] = asyncio.Queue(maxsize=1024)

    def _sink(event: CollectionEvent) -> None:
        # All pipeline code runs in the event loop — put directly so events
        # reach the consumer before the sentinel does.
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("pipeline.event_queue_full", run_id=config.run_id)

    async def _runner() -> None:
        try:
            await run_collection(config, adapters, event_sink=_sink)
        finally:
            await queue.put(None)  # sentinel

    runner_task = asyncio.create_task(_runner())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        if not runner_task.done():
            runner_task.cancel()
            try:
                await runner_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Internal stage functions
# ---------------------------------------------------------------------------

async def _stage_query_generation(
    config: RunConfig,
    adapters: AdapterBundle,
    manifest: RunManifest,
    sink: EventSink,
) -> list[Query]:
    from collection_system.agents.graph import run_query_agent

    stage = Stage.QUERY_GENERATION
    started = datetime.now(timezone.utc)
    sink(StageStarted(run_id=config.run_id, stage=stage, at=started))

    try:
        async with asyncio.timeout(STAGE_BUDGETS_S[stage]):
            queries: list[Query] = await run_query_agent(
                topic=config.topic,
                run_id=config.run_id,
                llm=adapters.llm,
                max_depth=config.max_depth,
                max_queries=config.max_queries,
                relevance_threshold=config.relevance_threshold,
            )
    except asyncio.TimeoutError as exc:
        raise PipelineTimeoutError(stage.value, STAGE_BUDGETS_S[stage]) from exc

    # Persist queries
    for q in queries:
        try:
            await adapters.storage.save_query(q)
        except Exception as exc:  # noqa: BLE001
            log.warning("pipeline.save_query_failed", run_id=config.run_id, error=str(exc))

    completed = datetime.now(timezone.utc)
    stats = StageStats(
        stage=stage,
        started_at=started,
        completed_at=completed,
        input_count=1,  # one root topic
        output_count=len(queries),
        failure_count=0,
    )
    manifest.stages[stage] = stats
    await adapters.storage.save_stage_stats(config.run_id, stats)

    sink(QueriesGenerated(run_id=config.run_id, count=len(queries)))
    sink(StageCompleted(run_id=config.run_id, stage=stage, stats=stats))
    log.info(
        "stage.query_generation.done",
        run_id=config.run_id,
        queries=len(queries),
        duration_s=(completed - started).total_seconds(),
    )
    return queries


async def _stage_url_discovery(
    config: RunConfig,
    queries: list[Query],
    adapters: AdapterBundle,
    manifest: RunManifest,
    sink: EventSink,
) -> list[DiscoveredURL]:
    stage = Stage.URL_DISCOVERY
    started = datetime.now(timezone.utc)
    sink(StageStarted(run_id=config.run_id, stage=stage, at=started))

    # How many URLs do we need? Aim for 2× doc_count to survive scrape failures.
    target_urls = max(100, config.doc_count * 2)
    per_query = max(5, target_urls // max(1, len(queries)))

    unique_hashes: set[str] = set()
    discovered: list[DiscoveredURL] = []
    failure_count = 0

    try:
        async with asyncio.timeout(STAGE_BUDGETS_S[stage]):
            # Fan out across queries — composite adapter handles backend-level fan-out
            sem = asyncio.Semaphore(32)  # allow high concurrency — backends have their own rate limits

            async def _one(q: Query) -> list[DiscoveredURL]:
                async with sem:
                    try:
                        return await adapters.search.discover_urls(q, limit=per_query)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "url_discovery.query_failed",
                            run_id=config.run_id,
                            query=q.text,
                            error=str(exc),
                        )
                        return []

            batches = await asyncio.gather(*[_one(q) for q in queries])
    except asyncio.TimeoutError as exc:
        raise PipelineTimeoutError(stage.value, STAGE_BUDGETS_S[stage]) from exc

    # Dedup + persist
    for batch in batches:
        backend_counts: dict[str, int] = {}
        for du in batch:
            if du.url_hash in unique_hashes:
                continue
            unique_hashes.add(du.url_hash)
            discovered.append(du)
            backend_counts[du.source_backend.value] = (
                backend_counts.get(du.source_backend.value, 0) + 1
            )
            try:
                await adapters.storage.save_url(du)
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                log.warning(
                    "url_discovery.save_failed",
                    run_id=config.run_id,
                    error=str(exc),
                )
            if len(discovered) >= target_urls:
                break
        for backend, count in backend_counts.items():
            if count:
                sink(
                    URLsDiscovered(
                        run_id=config.run_id, count=count, backend=backend
                    )
                )
        if len(discovered) >= target_urls:
            break

    completed = datetime.now(timezone.utc)
    stats = StageStats(
        stage=stage,
        started_at=started,
        completed_at=completed,
        input_count=len(queries),
        output_count=len(discovered),
        failure_count=failure_count,
    )
    manifest.stages[stage] = stats
    await adapters.storage.save_stage_stats(config.run_id, stats)

    sink(StageCompleted(run_id=config.run_id, stage=stage, stats=stats))
    log.info(
        "stage.url_discovery.done",
        run_id=config.run_id,
        urls=len(discovered),
        duration_s=(completed - started).total_seconds(),
    )
    return discovered


async def _stage_scrape(
    config: RunConfig,
    urls: list[DiscoveredURL],
    adapters: AdapterBundle,
    manifest: RunManifest,
    sink: EventSink,
) -> tuple[list[ScrapedDoc], list[Failure]]:
    stage = Stage.SCRAPE
    started = datetime.now(timezone.utc)
    sink(StageStarted(run_id=config.run_id, stage=stage, at=started))

    docs: list[ScrapedDoc] = []
    failures: list[Failure] = []
    target = config.doc_count
    seen_content: set[str] = set()

    async def _run_scrape() -> None:
        # Use scrape_batch if available for streaming results
        batcher = adapters.scraper
        async for result in batcher.scrape_batch(urls, config.scraper_concurrency):
            if isinstance(result, Failure):
                failures.append(result)
                try:
                    await adapters.storage.save_failure(result)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "scrape.save_failure_failed",
                        run_id=config.run_id,
                        error=str(exc),
                    )
                sink(
                    DocFailed(
                        run_id=config.run_id,
                        url=result.target,
                        error_type=result.error_type,
                    )
                )
                continue

            # Exact-content dedup
            if result.content_hash in seen_content:
                continue
            if await adapters.storage.content_hash_seen(
                config.run_id, result.content_hash
            ):
                seen_content.add(result.content_hash)
                continue
            seen_content.add(result.content_hash)

            # Write to filesystem (updates path field)
            try:
                path = await adapters.filesystem.write_doc(result)
                result = result.model_copy(update={"path": path})
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "scrape.fs_write_failed",
                    run_id=config.run_id,
                    error=str(exc),
                )
                continue

            # Persist DB row
            try:
                await adapters.storage.save_doc(result)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "scrape.save_doc_failed",
                    run_id=config.run_id,
                    error=str(exc),
                )
                continue

            docs.append(result)
            sink(
                DocScraped(
                    run_id=config.run_id,
                    url=result.url,
                    token_count=result.token_count,
                )
            )

            # Checkpoint every N docs
            if len(docs) % CHECKPOINT_INTERVAL_DOCS == 0:
                log.debug("scrape.checkpoint", run_id=config.run_id, docs=len(docs))

            if len(docs) >= target:
                break

    try:
        async with asyncio.timeout(STAGE_BUDGETS_S[stage]):
            async with adapters.scraper:
                await _run_scrape()
    except asyncio.TimeoutError:
        log.warning(
            "scrape.stage_timeout",
            run_id=config.run_id,
            docs_collected=len(docs),
            budget_s=STAGE_BUDGETS_S[stage],
        )
        # Not fatal — we keep what we got

    completed = datetime.now(timezone.utc)
    stats = StageStats(
        stage=stage,
        started_at=started,
        completed_at=completed,
        input_count=len(urls),
        output_count=len(docs),
        failure_count=len(failures),
    )
    manifest.stages[stage] = stats
    await adapters.storage.save_stage_stats(config.run_id, stats)

    sink(StageCompleted(run_id=config.run_id, stage=stage, stats=stats))
    log.info(
        "stage.scrape.done",
        run_id=config.run_id,
        docs=len(docs),
        failures=len(failures),
        duration_s=(completed - started).total_seconds(),
    )
    return docs, failures


async def _stage_finalize(
    config: RunConfig,
    docs: list[ScrapedDoc],
    failures: list[Failure],
    adapters: AdapterBundle,
    started_at: datetime,
    manifest: RunManifest,
    sink: EventSink,
) -> None:
    stage = Stage.FINALIZE
    s_started = datetime.now(timezone.utc)
    sink(StageStarted(run_id=config.run_id, stage=stage, at=s_started))

    # Total so far
    manifest.stages[stage] = StageStats(
        stage=stage,
        started_at=s_started,
        completed_at=None,
        input_count=len(docs),
        output_count=len(docs),
        failure_count=len(failures),
    )

    # Write artefacts on disk
    try:
        await adapters.filesystem.write_manifest(manifest)
    except Exception as exc:  # noqa: BLE001
        log.warning("finalize.manifest_failed", run_id=config.run_id, error=str(exc))

    metrics = {
        "run_id": config.run_id,
        "topic": config.topic,
        "stages": {
            s.value: {
                "duration_s": st.duration_s,
                "input_count": st.input_count,
                "output_count": st.output_count,
                "failure_count": st.failure_count,
            }
            for s, st in manifest.stages.items()
        },
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_docs": len(docs),
        "total_failures": len(failures),
    }
    try:
        await adapters.filesystem.write_metrics(config.run_id, metrics)
    except Exception as exc:  # noqa: BLE001
        log.warning("finalize.metrics_failed", run_id=config.run_id, error=str(exc))

    completed = datetime.now(timezone.utc)
    manifest.stages[stage] = StageStats(
        stage=stage,
        started_at=s_started,
        completed_at=completed,
        input_count=len(docs),
        output_count=len(docs),
        failure_count=len(failures),
    )
    await adapters.storage.save_stage_stats(config.run_id, manifest.stages[stage])

    sink(StageCompleted(run_id=config.run_id, stage=stage, stats=manifest.stages[stage]))
    log.info(
        "stage.finalize.done",
        run_id=config.run_id,
        docs=len(docs),
        duration_s=(completed - s_started).total_seconds(),
    )
