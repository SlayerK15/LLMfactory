"""Public API for cleaning-system — consumed by dataset-forge and the orchestrator."""
from __future__ import annotations

from pathlib import Path

from cleaning_system.core.models import CleaningConfig, CleaningReport
from cleaning_system.core.pipeline import run_cleaning as _run_cleaning
from cleaning_system.infra.config import get_settings
from cleaning_system.infra.logging import configure_logging


async def run_cleaning(
    run_id: str,
    topic: str,
    *,
    data_dir: Path | str | None = None,
    enable_perplexity: bool | None = None,
    enable_relevance: bool | None = None,
    enable_trafilatura: bool | None = None,
    relevance_threshold: float | None = None,
    near_dup_threshold: float | None = None,
    target_language: str | None = None,
) -> CleaningReport:
    """
    Run the full cleaning pipeline for a collection run.

    Args:
        run_id:   The run ID produced by collection-system.
        topic:    The original topic string (used for relevance scoring).
        data_dir: Root data directory. Defaults to settings.data_dir (./data).

    Returns:
        CleaningReport with per-stage doc counts and quality metrics.

    Example::

        from cleaning_system import run_cleaning

        report = await run_cleaning("abc123", topic="DevOps")
        print(report.model_dump_json(indent=2))
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    config = CleaningConfig(
        run_id=run_id,
        topic=topic,
        data_dir=Path(data_dir) if data_dir else settings.data_dir,
        enable_perplexity=enable_perplexity
        if enable_perplexity is not None
        else settings.enable_perplexity,
        enable_relevance=enable_relevance
        if enable_relevance is not None
        else settings.enable_relevance,
        enable_trafilatura=enable_trafilatura
        if enable_trafilatura is not None
        else settings.enable_trafilatura,
        relevance_threshold=relevance_threshold
        if relevance_threshold is not None
        else settings.relevance_threshold,
        near_dup_threshold=near_dup_threshold
        if near_dup_threshold is not None
        else settings.near_dup_threshold,
        target_language=target_language
        if target_language is not None
        else settings.target_language,
        perplexity_model_id=settings.perplexity_model_id,
        relevance_model_id=settings.relevance_model_id,
    )

    return await _run_cleaning(config)
