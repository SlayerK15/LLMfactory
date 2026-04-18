"""
Integration test: full pipeline with optional ML stages.
Heavy stages (perplexity, relevance) require model downloads — marked slow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.core.pipeline import run_cleaning


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_relevance(populated_data_dir: Path):
    """BGE-M3 relevance filter should keep most DevOps docs above threshold 0.20."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

    config = CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_data_dir,
        enable_perplexity=False,
        enable_relevance=True,
        enable_trafilatura=False,
        relevance_threshold=0.20,
    )

    report = await run_cleaning(config)

    # All 10 sample docs are clearly about DevOps — most should survive
    assert report.after_relevance >= 7, (
        f"Expected >= 7 docs to pass relevance filter, got {report.after_relevance}"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_with_perplexity(populated_data_dir: Path):
    """Perplexity filter should keep most of 10 high-quality docs."""
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")

    config = CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_data_dir,
        enable_perplexity=True,
        enable_relevance=False,
        enable_trafilatura=False,
        perplexity_low_pct=0.05,
        perplexity_high_pct=0.10,
    )

    report = await run_cleaning(config)

    # With only 10 docs and 5%+10% trimming, we expect to lose at most 2
    assert report.after_perplexity >= 8
