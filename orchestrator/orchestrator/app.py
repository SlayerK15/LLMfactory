"""FastAPI app with SSE streaming + static frontend."""
from __future__ import annotations

import sys

# Force UTF-8 stdio on Windows so Crawl4AI's unicode progress chars (e.g. ↓)
# don't crash scrapes with `charmap codec can't encode` (cp1252 default).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

# Load .env from the repo root (walks up from CWD) BEFORE any subsystem imports,
# so pydantic-settings in collection_system / cleaning_system / etc. can pick up
# DATABASE_URL, GROQ_API_KEY, etc.
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from orchestrator.log_bridge import configure_logging, sse_log_queue
from orchestrator.runner import run_full_pipeline

configure_logging()

STATIC_DIR = Path(__file__).parent / "static"


class PipelineRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    doc_count: int = Field(default=2000, ge=10, le=50_000)
    output_mode: str = Field(default="pretrain", pattern="^(pretrain|instruct)$")
    data_dir: str = "./data"
    mode: Literal["dataset_only", "train"] = "dataset_only"
    # Training knobs — only used when mode == "train".
    base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    epochs: int = Field(default=1, ge=1, le=20)
    lora_rank: int = Field(default=16, ge=4, le=128)
    output_format: str = Field(
        default="gguf_q4_k_m", pattern="^(gguf_q4_k_m|gguf_q8_0|safetensors)$"
    )
    s3_bucket: str | None = None

    def train_knobs(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "epochs": self.epochs,
            "lora_rank": self.lora_rank,
            "output_format": self.output_format,
            "s3_bucket": self.s3_bucket,
        }


def create_app() -> FastAPI:
    app = FastAPI(title="Collection System Orchestrator", version="0.1.0")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        if not (STATIC_DIR / "index.html").exists():
            raise HTTPException(500, "frontend not found")
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/pipeline")
    async def run(req: PipelineRequest) -> EventSourceResponse:
        async def event_stream() -> AsyncIterator[dict[str, str]]:
            async for item in _pump_with_logs(
                run_full_pipeline(
                    topic=req.topic,
                    doc_count=req.doc_count,
                    output_mode=req.output_mode,
                    data_dir=Path(req.data_dir),
                    mode=req.mode,
                    train_config=req.train_knobs(),
                )
            ):
                yield item

        return EventSourceResponse(event_stream())

    @app.post("/api/pipeline/dry-run")
    async def dry_run(req: PipelineRequest) -> EventSourceResponse:
        """Runs the pipeline with in-memory fakes — used by the UI smoke test."""

        async def fake_collect(*, topic: str, run_id: str, data_dir: Path, doc_count: int) -> dict:
            return {"topic": topic, "doc_count": doc_count, "dry_run": True}

        async def fake_clean(*, topic: str, run_id: str, data_dir: Path) -> dict:
            return {"cleaned": True, "dry_run": True}

        async def fake_forge(*, topic: str, run_id: str, data_dir: Path, output_mode: str) -> dict:
            return {"output_mode": output_mode, "dry_run": True}

        async def fake_train(
            *,
            topic: str,
            run_id: str,
            data_dir: Path,
            output_mode: str,
            train_config: dict[str, Any],
        ) -> dict:
            return {
                "trained": True,
                "dry_run": True,
                "base_model": train_config.get("base_model"),
                "epochs": train_config.get("epochs"),
                "artifact": {
                    "local_path": str(data_dir / "trainer" / f"{run_id}.gguf"),
                    "s3_uri": None,
                },
            }

        async def event_stream() -> AsyncIterator[dict[str, str]]:
            async for item in _pump_with_logs(
                run_full_pipeline(
                    topic=req.topic,
                    doc_count=req.doc_count,
                    output_mode=req.output_mode,
                    data_dir=Path(req.data_dir),
                    mode=req.mode,
                    train_config=req.train_knobs(),
                    collect=fake_collect,
                    clean=fake_clean,
                    forge=fake_forge,
                    train=fake_train,
                )
            ):
                yield item

        return EventSourceResponse(event_stream())

    return app


async def _pump_with_logs(pipeline_gen):
    """Merge pipeline OrchestratorEvent yields with structlog lines on one SSE stream."""
    queue: asyncio.Queue = asyncio.Queue()
    token = sse_log_queue.set(queue)

    async def drive() -> None:
        try:
            async for evt in pipeline_gen:
                await queue.put(("event", evt))
        except Exception as exc:  # noqa: BLE001
            await queue.put(("error", str(exc)))
        finally:
            await queue.put(None)

    task = asyncio.create_task(drive())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "event":
                evt = item[1]
                yield {"event": evt.kind, "data": evt.model_dump_json()}
            elif isinstance(item, tuple) and item[0] == "error":
                yield {"event": "error", "data": json.dumps({"kind": "error", "message": item[1]})}
            else:
                yield {"event": "log", "data": json.dumps({"kind": "log", "message": item})}
    finally:
        sse_log_queue.reset(token)
        if not task.done():
            task.cancel()


app = create_app()
