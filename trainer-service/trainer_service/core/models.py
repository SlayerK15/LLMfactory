"""Domain models for trainer-service."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class TrainingStyle(str, Enum):
    PRETRAIN = "pretrain"
    INSTRUCT = "instruct"


class OutputFormat(str, Enum):
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q8_0 = "gguf_q8_0"
    SAFETENSORS = "safetensors"


class ModelSpec(BaseModel):
    """Describes a base model candidate — used by the budget planner."""

    hf_id: str
    params_b: float
    default_lora_rank: int = 16
    context_window: int = 4096


class TrainConfig(BaseModel):
    """All knobs for one training run."""

    run_id: str
    topic: str

    # dataset
    train_jsonl: Path
    eval_jsonl: Path
    training_style: TrainingStyle = TrainingStyle.INSTRUCT

    # model
    base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    params_b: float = 3.0
    context_window: int = 4096

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )

    # training hyperparams
    epochs: int = 1
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.03
    seed: int = 42

    # output
    output_format: OutputFormat = OutputFormat.GGUF_Q4_K_M
    s3_bucket: str | None = None
    s3_key_prefix: str = "trainer"

    # compute
    modal_app_name: str = "collection-system-trainer"
    modal_gpu: str = "A100-40GB"
    modal_timeout_s: int = 60 * 60 * 2
    time_budget_s: int = 60 * 30  # 30-minute default soft budget

    # local workspace
    workspace: Path = Path("./data/trainer")

    @property
    def artifact_dir(self) -> Path:
        return self.workspace / self.run_id

    @property
    def s3_key(self) -> str:
        suffix = "gguf" if self.output_format.value.startswith("gguf") else "safetensors"
        return f"{self.s3_key_prefix}/{self.run_id}/model.{suffix}"


class TrainingArtifact(BaseModel):
    run_id: str
    local_path: Path
    size_bytes: int
    output_format: OutputFormat
    s3_uri: str | None = None
    merged_sha256: str | None = None


class TrainReport(BaseModel):
    run_id: str
    topic: str
    base_model: str
    params_b: float
    training_style: TrainingStyle
    epochs: int
    train_tokens: int
    eval_tokens: int
    train_loss_final: float | None = None
    eval_loss_final: float | None = None
    modal_duration_s: float
    total_duration_s: float
    artifact: TrainingArtifact
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        if self.modal_duration_s == 0:
            return 0.0
        return round((self.train_tokens * self.epochs) / self.modal_duration_s, 1)
