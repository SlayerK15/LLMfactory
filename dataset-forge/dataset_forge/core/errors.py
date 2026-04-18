"""Typed error hierarchy for dataset-forge."""
from __future__ import annotations


class ForgeError(Exception):
    """Base class for all forge errors."""


class CleanedCorpusNotFoundError(ForgeError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"No cleaned corpus found for run_id={run_id}")
        self.run_id = run_id


class EmptyCorpusError(ForgeError):
    """Raised when loaded corpus yields zero chunks (all docs filtered out)."""


class QASynthError(ForgeError):
    """Raised when Q/A synthesis fails in a non-retryable way."""


class EmbedError(ForgeError):
    """Raised when embedding into the vector store fails."""
