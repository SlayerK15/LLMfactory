"""Public API for the trainer. Same shape as other projects' api.py modules."""
from __future__ import annotations

from trainer_service.adapters.compute.modal_adapter import ModalCompute
from trainer_service.adapters.storage.s3_adapter import S3Storage
from trainer_service.core.models import TrainConfig, TrainReport
from trainer_service.core.pipeline import run_training as _run_training


async def run_training(config: TrainConfig) -> TrainReport:
    """Run a training job using Modal for compute and S3 for artifact upload."""
    compute = ModalCompute(app_name=config.modal_app_name)
    storage = S3Storage(bucket=config.s3_bucket) if config.s3_bucket else None
    return await _run_training(config, compute=compute, storage=storage)
