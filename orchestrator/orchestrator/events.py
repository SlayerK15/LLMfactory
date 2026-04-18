"""Event model for pipeline progress. Serialised to JSON over SSE."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Stage(str, Enum):
    INIT = "init"
    COLLECT = "collect"
    CLEAN = "clean"
    FORGE = "forge"
    TRAIN = "train"
    DONE = "done"
    FAILED = "failed"


class OrchestratorEvent(BaseModel):
    kind: Literal["stage_started", "stage_done", "log", "error", "pipeline_done"]
    stage: Stage
    run_id: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
