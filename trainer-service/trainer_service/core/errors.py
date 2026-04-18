"""Typed errors for trainer-service."""
from __future__ import annotations


class TrainerError(Exception):
    pass


class DatasetNotFoundError(TrainerError):
    def __init__(self, path: str) -> None:
        super().__init__(f"dataset not found: {path}")
        self.path = path


class BudgetExceededError(TrainerError):
    """Selected model + epochs are estimated to exceed the time budget."""


class ComputeDispatchError(TrainerError):
    """Modal (or other compute) dispatch failed."""


class ArtifactUploadError(TrainerError):
    """S3 (or other storage) upload failed."""
