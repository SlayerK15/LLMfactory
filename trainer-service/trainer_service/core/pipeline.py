"""Orchestrator: dispatch training, write artifact locally, upload to S3, build report."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import structlog

from trainer_service.core.errors import DatasetNotFoundError
from trainer_service.core.models import (
    OutputFormat,
    TrainConfig,
    TrainingArtifact,
    TrainingStyle,
    TrainReport,
)

log = structlog.get_logger()


class ComputeAdapter(Protocol):
    async def dispatch(self, config: TrainConfig) -> dict[str, Any]: ...


class StorageAdapter(Protocol):
    def upload(self, local_path: Path, key: str, content_type: str = "application/octet-stream") -> str: ...


def _ensure_dataset(config: TrainConfig) -> None:
    for p in (config.train_jsonl, config.eval_jsonl):
        if not p.exists():
            raise DatasetNotFoundError(str(p))


def _write_artifact(bytes_: bytes, config: TrainConfig) -> Path:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = "gguf" if config.output_format.value.startswith("gguf") else "safetensors.zip"
    out = config.artifact_dir / f"model.{suffix}"
    out.write_bytes(bytes_)
    return out


async def run_training(
    config: TrainConfig,
    compute: ComputeAdapter,
    storage: StorageAdapter | None = None,
) -> TrainReport:
    t0 = time.monotonic()
    log.info(
        "trainer.start",
        run_id=config.run_id,
        base_model=config.base_model,
        style=config.training_style.value,
        output_format=config.output_format.value,
    )
    _ensure_dataset(config)

    result = await compute.dispatch(config)
    artifact_bytes: bytes = result["artifact_bytes"]
    sha: str | None = result.get("artifact_sha256")
    metrics: dict[str, Any] = result.get("metrics", {})

    local_path = _write_artifact(artifact_bytes, config)
    s3_uri: str | None = None
    if storage is not None and config.s3_bucket:
        content_type = "application/octet-stream"
        s3_uri = storage.upload(local_path, config.s3_key, content_type=content_type)

    artifact = TrainingArtifact(
        run_id=config.run_id,
        local_path=local_path,
        size_bytes=local_path.stat().st_size,
        output_format=config.output_format,
        s3_uri=s3_uri,
        merged_sha256=sha,
    )

    total_duration = time.monotonic() - t0
    modal_duration = float(metrics.get("modal_duration_s") or metrics.get("train_duration_s") or 0.0)

    report = TrainReport(
        run_id=config.run_id,
        topic=config.topic,
        base_model=config.base_model,
        params_b=config.params_b,
        training_style=config.training_style,
        epochs=config.epochs,
        train_tokens=int(metrics.get("train_tokens") or 0),
        eval_tokens=int(metrics.get("eval_tokens") or 0),
        train_loss_final=metrics.get("train_loss_final"),
        eval_loss_final=metrics.get("eval_loss_final"),
        modal_duration_s=round(modal_duration, 1),
        total_duration_s=round(total_duration, 1),
        artifact=artifact,
    )
    log.info(
        "trainer.done",
        run_id=config.run_id,
        artifact_mb=round(artifact.size_bytes / 1e6, 1),
        s3_uri=s3_uri,
        total_s=report.total_duration_s,
    )
    return report
