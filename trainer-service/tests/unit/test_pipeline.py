"""Unit tests for the trainer pipeline using fake compute + fake storage adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trainer_service.core.errors import DatasetNotFoundError
from trainer_service.core.models import OutputFormat, TrainConfig, TrainingStyle
from trainer_service.core.pipeline import run_training


class FakeCompute:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[TrainConfig] = []

    async def dispatch(self, config: TrainConfig) -> dict[str, Any]:
        self.calls.append(config)
        return self._payload


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[Path, str]] = []

    def upload(self, local_path: Path, key: str, content_type: str = "application/octet-stream") -> str:
        self.uploads.append((local_path, key))
        return f"s3://fake/{key}"


def _fake_payload(artifact: bytes = b"abc123") -> dict[str, Any]:
    return {
        "artifact_bytes": artifact,
        "artifact_sha256": "deadbeef",
        "metrics": {
            "train_loss_final": 0.42,
            "eval_loss_final": 0.51,
            "train_tokens": 12_345,
            "eval_tokens": 678,
            "modal_duration_s": 55.0,
        },
    }


async def test_pipeline_writes_artifact_locally(train_config: TrainConfig):
    compute = FakeCompute(_fake_payload())
    report = await run_training(train_config, compute=compute, storage=None)
    assert report.artifact.local_path.exists()
    assert report.artifact.size_bytes == len(b"abc123")
    assert report.train_loss_final == pytest.approx(0.42)
    assert report.artifact.s3_uri is None


async def test_pipeline_uploads_when_storage_configured(train_config: TrainConfig):
    train_config.s3_bucket = "my-bucket"
    compute = FakeCompute(_fake_payload(artifact=b"x" * 1024))
    storage = FakeStorage()
    report = await run_training(train_config, compute=compute, storage=storage)
    assert report.artifact.s3_uri == f"s3://fake/{train_config.s3_key}"
    assert len(storage.uploads) == 1
    assert storage.uploads[0][1] == train_config.s3_key


async def test_pipeline_raises_when_dataset_missing(tmp_path: Path):
    config = TrainConfig(
        run_id="missing",
        topic="x",
        train_jsonl=tmp_path / "nope_train.jsonl",
        eval_jsonl=tmp_path / "nope_eval.jsonl",
        training_style=TrainingStyle.INSTRUCT,
        output_format=OutputFormat.GGUF_Q4_K_M,
        workspace=tmp_path,
    )
    with pytest.raises(DatasetNotFoundError):
        await run_training(config, compute=FakeCompute(_fake_payload()), storage=None)


async def test_pipeline_passes_config_to_compute(train_config: TrainConfig):
    compute = FakeCompute(_fake_payload())
    await run_training(train_config, compute=compute, storage=None)
    assert len(compute.calls) == 1
    assert compute.calls[0].run_id == train_config.run_id
