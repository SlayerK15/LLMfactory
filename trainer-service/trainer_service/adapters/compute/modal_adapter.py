"""Modal-backed compute adapter — dispatches a training job, waits, returns artifact bytes + metrics."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

from trainer_service.core.errors import ComputeDispatchError
from trainer_service.core.models import TrainConfig

log = structlog.get_logger()


class ModalCompute:
    """Dispatches training to the Modal app defined in ``trainer_service.training.modal_app``.

    The Modal app exposes a function ``train(config_dict, train_bytes, eval_bytes) -> dict``
    returning: {artifact_bytes, metrics: {train_loss, eval_loss, duration_s, train_tokens, eval_tokens}}.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    async def dispatch(self, config: TrainConfig) -> dict[str, Any]:
        return await asyncio.to_thread(self._dispatch_sync, config)

    def _dispatch_sync(self, config: TrainConfig) -> dict[str, Any]:
        try:
            import modal  # noqa: F401
        except ImportError as exc:
            raise ComputeDispatchError("modal SDK not installed") from exc
        try:
            from trainer_service.training.modal_app import train as modal_train
        except Exception as exc:
            raise ComputeDispatchError(f"cannot import modal_app: {exc}") from exc

        train_bytes = _read(config.train_jsonl)
        eval_bytes = _read(config.eval_jsonl)

        log.info(
            "modal.dispatch",
            run_id=config.run_id,
            base_model=config.base_model,
            gpu=config.modal_gpu,
            train_bytes=len(train_bytes),
        )
        t0 = time.monotonic()
        payload = config.model_dump(mode="json", exclude={"train_jsonl", "eval_jsonl", "workspace"})
        result = modal_train.remote(payload, train_bytes, eval_bytes)
        duration = time.monotonic() - t0
        result.setdefault("metrics", {})["modal_duration_s"] = duration
        log.info("modal.done", run_id=config.run_id, duration_s=round(duration, 1))
        return result


def _read(path: Path) -> bytes:
    return path.read_bytes()
