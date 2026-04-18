"""Typed error hierarchy for the collection system."""


class CollectionError(Exception):
    """Base for all collection-system errors."""


class ConfigurationError(CollectionError):
    """Invalid or missing configuration."""


class LLMError(CollectionError):
    """LLM provider call failed."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit hit."""


class SearchError(CollectionError):
    """Search backend call failed."""


class ScraperError(CollectionError):
    """Scraper failed for a URL."""


class StorageError(CollectionError):
    """Database or filesystem write failed."""


class RateLimitError(CollectionError):
    """Internal rate limiter blocked the call."""


class CircuitOpenError(CollectionError):
    """Circuit breaker is open for the target backend."""

    def __init__(self, backend: str) -> None:
        super().__init__(f"Circuit breaker open for backend: {backend}")
        self.backend = backend


class PipelineTimeoutError(CollectionError):
    """A pipeline stage exceeded its time budget."""

    def __init__(self, stage: str, timeout_s: int) -> None:
        super().__init__(f"Stage {stage} exceeded timeout of {timeout_s}s")
        self.stage = stage
        self.timeout_s = timeout_s


class CheckpointError(CollectionError):
    """Failed to save or load a checkpoint."""


class RunNotFoundError(CollectionError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run not found: {run_id}")
        self.run_id = run_id
