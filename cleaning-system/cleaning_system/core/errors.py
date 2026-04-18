"""Typed error hierarchy for cleaning-system."""
from __future__ import annotations


class CleaningError(Exception):
    """Base for all cleaning-system errors."""


class RunNotFoundError(CleaningError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"No collection run found at expected path for run_id={run_id!r}")
        self.run_id = run_id


class StageError(CleaningError):
    """A cleaning stage failed unrecoverably."""

    def __init__(self, stage: str, cause: str) -> None:
        super().__init__(f"Stage '{stage}' failed: {cause}")
        self.stage = stage


class ModelLoadError(CleaningError):
    """A required ML model could not be loaded."""

    def __init__(self, model_id: str, cause: str) -> None:
        super().__init__(f"Failed to load model '{model_id}': {cause}")
        self.model_id = model_id
