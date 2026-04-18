"""Unit tests for the cleaning pipeline — uses fast stages only (no ML models)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.core.pipeline import run_cleaning


@pytest.mark.asyncio
async def test_pipeline_end_to_end(populated_data_dir: Path):
    """Full pipeline run with perplexity/relevance disabled — fast, no model downloads."""
    config = CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_data_dir,
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=False,
    )

    report = await run_cleaning(config)

    assert report.run_id == "test-run"
    assert report.topic == "DevOps"
    assert report.input_docs == 10
    assert report.after_near_dedup <= report.input_docs
    assert report.after_lang_filter <= report.after_near_dedup
    assert report.after_gopher <= report.after_lang_filter
    assert report.after_perplexity == report.after_gopher  # disabled
    assert report.after_relevance == report.after_perplexity  # disabled
    assert report.total_tokens > 0
    assert report.median_doc_tokens > 0
    assert report.cleaning_duration_s > 0
    assert 0.0 <= report.drop_rate <= 1.0


@pytest.mark.asyncio
async def test_pipeline_writes_cleaned_docs(populated_data_dir: Path):
    config = CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_data_dir,
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=False,
    )

    report = await run_cleaning(config)

    cleaned_dir = populated_data_dir / "runs" / "test-run" / "cleaned"
    assert cleaned_dir.exists()

    md_files = list(cleaned_dir.glob("*.md"))
    assert len(md_files) == report.after_relevance

    meta_files = list(cleaned_dir.glob("*.meta.json"))
    assert len(meta_files) == len(md_files)


@pytest.mark.asyncio
async def test_pipeline_writes_report(populated_data_dir: Path):
    config = CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_data_dir,
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=False,
    )

    report = await run_cleaning(config)

    report_path = populated_data_dir / "reports" / "test-run.json"
    assert report_path.exists()

    saved = json.loads(report_path.read_text())
    assert saved["run_id"] == "test-run"
    assert saved["input_docs"] == 10
    assert saved["after_relevance"] == report.after_relevance


@pytest.mark.asyncio
async def test_pipeline_missing_run_raises(tmp_path: Path):
    from cleaning_system.core.errors import RunNotFoundError

    config = CleaningConfig(
        run_id="nonexistent-run",
        topic="DevOps",
        data_dir=tmp_path,
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=False,
    )

    with pytest.raises(RunNotFoundError):
        await run_cleaning(config)


@pytest.mark.asyncio
async def test_pipeline_keeps_all_unique_english_docs(populated_data_dir: Path):
    """10 unique English DevOps docs should all survive near-dedup, lang, and gopher stages."""
    config = CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_data_dir,
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=False,
    )

    report = await run_cleaning(config)

    # All 10 sample docs are unique English text that should survive all fast stages
    assert report.after_gopher == 10
