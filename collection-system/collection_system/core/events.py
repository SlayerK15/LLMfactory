"""
Pipeline event types — the frozen wire format for streaming consumers
(orchestrator, CLI, future FastAPI SSE frontend).

These are intentionally simple dataclasses (not Pydantic) for cheap
creation in hot paths. Serialisation to JSON is handled in api.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from collection_system.core.constants import RunStatus, Stage
from collection_system.core.models import RunConfig, StageStats


@dataclass(frozen=True, slots=True)
class RunStarted:
    kind: Literal["RunStarted"] = "RunStarted"
    run_id: str = ""
    config: RunConfig | None = None
    at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class StageStarted:
    stage: Stage = Stage.QUERY_GENERATION
    kind: Literal["StageStarted"] = "StageStarted"
    run_id: str = ""
    at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class QueriesGenerated:
    count: int = 0
    kind: Literal["QueriesGenerated"] = "QueriesGenerated"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class URLsDiscovered:
    count: int = 0
    backend: str = ""
    kind: Literal["URLsDiscovered"] = "URLsDiscovered"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class DocScraped:
    url: str = ""
    token_count: int = 0
    kind: Literal["DocScraped"] = "DocScraped"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class DocFailed:
    url: str = ""
    error_type: str = ""
    kind: Literal["DocFailed"] = "DocFailed"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class StageCompleted:
    stage: Stage = Stage.QUERY_GENERATION
    stats: StageStats | None = None
    kind: Literal["StageCompleted"] = "StageCompleted"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class RunCompleted:
    kind: Literal["RunCompleted"] = "RunCompleted"
    run_id: str = ""
    status: RunStatus = RunStatus.COMPLETED
    docs_collected: int = 0
    duration_s: float = 0.0


@dataclass(frozen=True, slots=True)
class RunFailed:
    error: str = ""
    kind: Literal["RunFailed"] = "RunFailed"
    run_id: str = ""


CollectionEvent = (
    RunStarted
    | StageStarted
    | QueriesGenerated
    | URLsDiscovered
    | DocScraped
    | DocFailed
    | StageCompleted
    | RunCompleted
    | RunFailed
)
