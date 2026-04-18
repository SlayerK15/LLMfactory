"""Run checkpoint save/load for crash recovery."""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from collection_system.core.errors import CheckpointError
from collection_system.core.constants import Stage

log = structlog.get_logger()


class CheckpointManager:
    """
    Saves pipeline stage state to JSON files so a run can be resumed
    after a crash. One file per run: data/checkpoints/{run_id}.json
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "checkpoints"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    async def save(self, run_id: str, stage: Stage, state: dict) -> None:
        """Persist checkpoint atomically (write-then-rename)."""
        path = self._path(run_id)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"run_id": run_id, "stage": stage, "state": state}))
            tmp.replace(path)
            log.debug("checkpoint_saved", run_id=run_id, stage=stage)
        except OSError as e:
            raise CheckpointError(f"Failed to save checkpoint for {run_id}") from e

    async def load(self, run_id: str) -> dict | None:
        """Load last checkpoint. Returns None if no checkpoint exists."""
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise CheckpointError(f"Failed to load checkpoint for {run_id}") from e

    async def delete(self, run_id: str) -> None:
        path = self._path(run_id)
        if path.exists():
            path.unlink()
