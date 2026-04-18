"""Chains collection → cleaning → dataset-forge (→ optional training) and emits events."""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import structlog

from orchestrator.events import OrchestratorEvent, Stage

log = structlog.get_logger()


class CollectFn(Protocol):
    async def __call__(self, *, topic: str, run_id: str, data_dir: Path, doc_count: int) -> dict: ...


class CleanFn(Protocol):
    async def __call__(self, *, topic: str, run_id: str, data_dir: Path) -> dict: ...


class ForgeFn(Protocol):
    async def __call__(self, *, topic: str, run_id: str, data_dir: Path, output_mode: str) -> dict: ...


class TrainFn(Protocol):
    async def __call__(
        self,
        *,
        topic: str,
        run_id: str,
        data_dir: Path,
        output_mode: str,
        train_config: dict[str, Any],
    ) -> dict: ...


async def _default_collect(*, topic: str, run_id: str, data_dir: Path, doc_count: int) -> dict:
    from collection_system import RunConfig, run_collection  # type: ignore[import-not-found]
    from collection_system.core.constants import RunStatus  # type: ignore[import-not-found]

    handle = await run_collection(RunConfig(topic=topic, doc_count=doc_count, run_id=run_id))
    # Draining events keeps the runner task unblocked; the queue has a small maxsize.
    # All structured logs are already surfacing through the orchestrator's SSE bridge.
    async for _ in handle.events:
        pass
    manifest = await handle.wait()
    if manifest.status != RunStatus.COMPLETED:
        raise RuntimeError(f"collection failed: {manifest.error_msg or 'no error message'}")
    return {"run_id": run_id, "total_docs": manifest.total_docs}


async def _default_clean(*, topic: str, run_id: str, data_dir: Path) -> dict:
    from cleaning_system.core.models import CleaningConfig
    from cleaning_system.core.pipeline import run_cleaning

    config = CleaningConfig(run_id=run_id, topic=topic, data_dir=data_dir)
    report = await run_cleaning(config)
    return report.model_dump(mode="json")


async def _default_forge(*, topic: str, run_id: str, data_dir: Path, output_mode: str) -> dict:
    from dataset_forge import ForgeConfig, OutputMode, run_forge

    config = ForgeConfig(
        run_id=run_id, topic=topic, data_dir=data_dir, output_mode=OutputMode(output_mode)
    )
    report = await run_forge(config)
    return report.model_dump(mode="json")


async def _default_train(
    *,
    topic: str,
    run_id: str,
    data_dir: Path,
    output_mode: str,
    train_config: dict[str, Any],
) -> dict:
    from trainer_service.core.models import OutputFormat, TrainConfig, TrainingStyle
    from trainer_service.api import run_training

    train_jsonl = data_dir / "datasets" / f"{run_id}.train.jsonl"
    eval_jsonl = data_dir / "datasets" / f"{run_id}.eval.jsonl"

    config = TrainConfig(
        run_id=run_id,
        topic=topic,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        training_style=TrainingStyle(output_mode),
        base_model=train_config.get("base_model", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"),
        params_b=float(train_config.get("params_b", 3.0)),
        epochs=int(train_config.get("epochs", 1)),
        lora_rank=int(train_config.get("lora_rank", 16)),
        output_format=OutputFormat(train_config.get("output_format", "gguf_q4_k_m")),
        s3_bucket=train_config.get("s3_bucket"),
        workspace=data_dir / "trainer",
    )
    report = await run_training(config)
    return report.model_dump(mode="json")


async def run_full_pipeline(
    topic: str,
    *,
    doc_count: int = 2000,
    output_mode: str = "pretrain",
    data_dir: Path = Path("./data"),
    mode: str = "dataset_only",
    train_config: dict[str, Any] | None = None,
    run_id: str | None = None,
    collect: CollectFn = _default_collect,
    clean: CleanFn = _default_clean,
    forge: ForgeFn = _default_forge,
    train: TrainFn = _default_train,
) -> AsyncIterator[OrchestratorEvent]:
    """Async generator. Yields progress events as each stage starts/finishes.

    mode="dataset_only" runs collect → clean → forge.
    mode="train" appends a training stage after forge.
    """
    rid = run_id or str(uuid.uuid4())
    yield OrchestratorEvent(
        kind="stage_started",
        stage=Stage.INIT,
        run_id=rid,
        message=f"topic={topic} mode={mode}",
    )

    stages: list[tuple[Stage, str, Any]] = [
        (Stage.COLLECT, "collecting docs", lambda: collect(topic=topic, run_id=rid, data_dir=data_dir, doc_count=doc_count)),
        (Stage.CLEAN, "cleaning corpus", lambda: clean(topic=topic, run_id=rid, data_dir=data_dir)),
        (Stage.FORGE, "building dataset", lambda: forge(topic=topic, run_id=rid, data_dir=data_dir, output_mode=output_mode)),
    ]
    if mode == "train":
        tcfg = train_config or {}
        stages.append(
            (
                Stage.TRAIN,
                "training model",
                lambda: train(
                    topic=topic,
                    run_id=rid,
                    data_dir=data_dir,
                    output_mode=output_mode,
                    train_config=tcfg,
                ),
            )
        )

    results: dict[str, dict] = {}
    for stage, msg, call in stages:
        yield OrchestratorEvent(kind="stage_started", stage=stage, run_id=rid, message=msg)
        try:
            payload = await call()
        except Exception as exc:
            log.exception("orchestrator.stage_failed", stage=stage.value)
            yield OrchestratorEvent(kind="error", stage=Stage.FAILED, run_id=rid, message=str(exc))
            return
        results[stage.value] = payload
        yield OrchestratorEvent(
            kind="stage_done", stage=stage, run_id=rid, message=f"{stage.value} complete", data=payload
        )

    yield OrchestratorEvent(
        kind="pipeline_done", stage=Stage.DONE, run_id=rid, message="pipeline complete", data=results
    )
