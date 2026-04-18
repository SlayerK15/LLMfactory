"""
Public programmatic API and FastAPI application.
This is the frozen contract consumed by the future orchestrator and frontend.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from collection_system.bootstrap import build_adapters
from collection_system.core.events import CollectionEvent, RunFailed
from collection_system.core.models import RunConfig, RunManifest
from collection_system.core.pipeline import run_collection as _pipeline_run
from collection_system.core.pipeline import (
    run_collection_streaming as _pipeline_stream,
)
from collection_system.core.errors import RunNotFoundError

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Event serialisation (dataclass → JSON dict)
# ---------------------------------------------------------------------------

def event_to_dict(event: CollectionEvent) -> dict:
    """Serialise a CollectionEvent to a JSON-safe dict."""
    if is_dataclass(event):
        data = asdict(event)
    else:
        data = dict(getattr(event, "__dict__", {}))

    def _coerce(value: object) -> object:
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "value"):  # StrEnum / IntEnum
            try:
                return value.value
            except Exception:  # noqa: BLE001
                return str(value)
        if isinstance(value, dict):
            return {k: _coerce(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_coerce(v) for v in value]
        return value

    return {k: _coerce(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Programmatic API (used directly in Python — orchestrator will consume this)
# ---------------------------------------------------------------------------

@dataclass
class RunHandle:
    """
    Live handle to an in-flight run. Consume `events` to stream progress,
    or `await handle.wait()` to block until completion.
    """
    run_id: str
    _events: AsyncIterator[CollectionEvent]
    _final: asyncio.Future[RunManifest]

    @property
    def events(self) -> AsyncIterator[CollectionEvent]:
        return self._events

    async def wait(self) -> RunManifest:
        return await self._final


async def run_collection(config: RunConfig) -> RunHandle:
    """
    Start a collection run and return a handle. The caller may consume
    handle.events for live progress or await handle.wait() for the final
    manifest. Events and manifest are produced concurrently.
    """
    adapters = await build_adapters(config)

    # Shared queue — the consumer (handle.events) and the runner share it.
    queue: asyncio.Queue[CollectionEvent | None] = asyncio.Queue(maxsize=1024)
    final: asyncio.Future[RunManifest] = asyncio.get_running_loop().create_future()

    def _sink(event: CollectionEvent) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("api.event_queue_full", run_id=config.run_id)

    async def _runner() -> None:
        try:
            scraper = getattr(adapters, "scraper", None)
            if hasattr(scraper, "__aenter__"):
                async with scraper:
                    manifest = await _pipeline_run(config, adapters, event_sink=_sink)
            else:
                manifest = await _pipeline_run(config, adapters, event_sink=_sink)
            if not final.done():
                final.set_result(manifest)
        except Exception as exc:  # noqa: BLE001
            log.exception("api.run_failed", run_id=config.run_id)
            if not final.done():
                final.set_exception(exc)
            _sink(RunFailed(run_id=config.run_id, error=str(exc)))
        finally:
            await queue.put(None)

    asyncio.create_task(_runner())

    async def _event_iter() -> AsyncIterator[CollectionEvent]:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    return RunHandle(run_id=config.run_id, _events=_event_iter(), _final=final)


async def run_collection_streaming(config: RunConfig) -> AsyncIterator[CollectionEvent]:
    """Yield CollectionEvent objects as work progresses."""
    adapters = await build_adapters(config)
    scraper = getattr(adapters, "scraper", None)

    if hasattr(scraper, "__aenter__"):
        async with scraper:
            async for event in _pipeline_stream(config, adapters):
                yield event
    else:
        async for event in _pipeline_stream(config, adapters):
            yield event


# ---------------------------------------------------------------------------
# FastAPI application (public HTTP surface)
# ---------------------------------------------------------------------------

app = FastAPI(title="Collection System", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/runs")
async def create_run(config: RunConfig) -> dict:
    """
    Start a new collection run. Returns immediately with the run_id;
    consume /runs/{run_id}/stream for live events or /runs/{run_id} for status.
    """
    handle = await run_collection(config)

    async def _drain() -> None:
        # Keep the run alive even if no one streams — drain events to the void.
        async for _ in handle.events:
            pass

    asyncio.create_task(_drain())
    return {"run_id": handle.run_id, "status": "RUNNING"}


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """
    SSE endpoint — streams events for a run that the server just started.
    NOTE: this is a minimal design. A production orchestrator should store
    events in Redis / Postgres and let the frontend reconnect on drop.
    """
    # For v0 we only stream runs started in the same process.
    # Multi-process streaming is a Phase 3 concern (pub/sub).
    raise HTTPException(
        status_code=501,
        detail=(
            "Multi-process event streaming is deferred to Phase 3. "
            "For in-process streaming, call run_collection() directly."
        ),
    )


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Fetch the current RunManifest for a run."""
    config = RunConfig(run_id=run_id, topic="")  # topic is unused for reads
    adapters = await build_adapters(config)
    try:
        manifest = await adapters.storage.load_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return manifest.model_dump(mode="json")


@app.get("/runs")
async def list_runs(limit: int = 20) -> dict:
    """List recent runs."""
    config = RunConfig(run_id="list-probe", topic="")
    adapters = await build_adapters(config)
    summaries = await adapters.storage.list_runs(limit=limit)
    return {"runs": [s.model_dump(mode="json") for s in summaries]}


# ---------------------------------------------------------------------------
# SSE helper — for future orchestrator/frontend use
# ---------------------------------------------------------------------------

async def _sse_format(events: AsyncIterator[CollectionEvent]) -> AsyncIterator[str]:
    async for event in events:
        payload = json.dumps(event_to_dict(event))
        kind = getattr(event, "kind", "event")
        yield f"event: {kind}\ndata: {payload}\n\n"
