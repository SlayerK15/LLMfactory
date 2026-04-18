"""Fixtures for trainer-service tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer_service.core.models import OutputFormat, TrainConfig, TrainingStyle


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")


@pytest.fixture
def instruct_dataset(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    _write_jsonl(
        train,
        [
            {
                "instruction": "What is CI/CD?",
                "input": "",
                "response": "Continuous integration / continuous delivery pipelines.",
                "doc_id": "d1",
            }
        ]
        * 20,
    )
    _write_jsonl(
        eval_,
        [
            {
                "instruction": "What is Docker?",
                "input": "",
                "response": "A container runtime.",
                "doc_id": "d2",
            }
        ]
        * 2,
    )
    return train, eval_


@pytest.fixture
def train_config(tmp_path: Path, instruct_dataset) -> TrainConfig:
    train_jsonl, eval_jsonl = instruct_dataset
    return TrainConfig(
        run_id="run-abc",
        topic="DevOps",
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        base_model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        params_b=3.0,
        training_style=TrainingStyle.INSTRUCT,
        output_format=OutputFormat.GGUF_Q4_K_M,
        workspace=tmp_path / "workspace",
    )
