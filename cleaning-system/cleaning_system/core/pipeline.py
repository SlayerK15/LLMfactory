"""
Core cleaning pipeline — orchestrates the 7 stages in order.
Pure orchestration only: no I/O beyond reading/writing the data directory.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

from cleaning_system.core.errors import RunNotFoundError
from cleaning_system.core.models import CleaningConfig, CleaningReport, DocRecord
from cleaning_system.stages import gopher, lang_filter, near_dedup, perplexity, relevance, trafilatura_fix

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_docs(config: CleaningConfig) -> list[DocRecord]:
    docs_dir = config.input_docs_dir
    if not docs_dir.exists():
        raise RunNotFoundError(config.run_id)

    docs: list[DocRecord] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        meta_path = md_path.with_suffix(".meta.json")
        if not meta_path.exists():
            log.warning("pipeline.missing_meta", path=str(md_path))
            continue
        try:
            docs.append(DocRecord.from_files(md_path, meta_path))
        except Exception as exc:
            log.warning("pipeline.load_error", path=str(md_path), error=str(exc))

    log.info("pipeline.loaded", run_id=config.run_id, n_docs=len(docs))
    return docs


def _save_cleaned_docs(docs: list[DocRecord], config: CleaningConfig) -> None:
    out_dir = config.cleaned_docs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        md_out = out_dir / f"{doc.id}.md"
        meta_out = out_dir / f"{doc.id}.meta.json"
        md_out.write_text(doc.text, encoding="utf-8")
        meta_out.write_text(
            json.dumps(
                {
                    "id": doc.id,
                    "run_id": doc.run_id,
                    "url": doc.url,
                    "title": doc.title,
                    "content_hash": doc.content_hash,
                    "token_count": doc.token_count,
                    "extraction_confidence": doc.extraction_confidence,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    log.info("pipeline.saved_cleaned", n_docs=len(docs), dir=str(out_dir))


def _save_report(report: CleaningReport, config: CleaningConfig) -> None:
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{config.run_id}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info("pipeline.report_saved", path=str(report_path))


def _enforce_min_keep_ratio(
    stage: str,
    before: list[DocRecord],
    after: list[DocRecord],
    input_docs: int,
    config: CleaningConfig,
) -> list[DocRecord]:
    """
    Avoid catastrophic stage drops by enforcing a minimum end-to-end keep ratio.
    If a stage would take the run below `min_cleaning_keep_ratio`, keep the
    pre-stage set instead and continue.
    """
    if input_docs <= 0:
        return after
    min_docs = max(1, int(input_docs * config.min_cleaning_keep_ratio))
    if len(after) >= min_docs:
        return after
    log.warning(
        "pipeline.stage_overdrop_guardrail",
        stage=stage,
        before=len(before),
        after=len(after),
        min_docs=min_docs,
        min_keep_ratio=config.min_cleaning_keep_ratio,
    )
    return before


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_cleaning(config: CleaningConfig) -> CleaningReport:
    """
    Execute all 7 cleaning stages in order and return the quality report.

    Stages:
      1. Load docs from filesystem
      2. Near-duplicate removal (MinHash LSH)
      3. Language filter (langdetect)
      4. Gopher quality heuristics
      5. Perplexity filter (Qwen2.5-0.5B — optional)
      6. Relevance filter (BGE-M3 embeddings — optional)
      7. Trafilatura re-extraction for low-confidence docs (optional)
    """
    log.info(
        "pipeline.start",
        run_id=config.run_id,
        topic=config.topic,
        enable_perplexity=config.enable_perplexity,
        enable_relevance=config.enable_relevance,
        enable_trafilatura=config.enable_trafilatura,
    )
    t0 = time.monotonic()

    # 1. Load
    docs = await asyncio.to_thread(_load_docs, config)
    n_input = len(docs)

    # 2. Near-dedup
    docs_before = docs
    docs = await asyncio.to_thread(near_dedup.run, docs, config)
    docs = _enforce_min_keep_ratio("near_dedup", docs_before, docs, n_input, config)
    n_after_near_dedup = len(docs)

    # 3. Language filter
    docs_before = docs
    docs = await asyncio.to_thread(lang_filter.run, docs, config)
    docs = _enforce_min_keep_ratio("lang_filter", docs_before, docs, n_input, config)
    n_after_lang = len(docs)

    # 4. Gopher heuristics
    docs_before = docs
    docs = await asyncio.to_thread(gopher.run, docs, config)
    docs = _enforce_min_keep_ratio("gopher", docs_before, docs, n_input, config)
    n_after_gopher = len(docs)

    # 5. Perplexity filter (model loaded in thread to avoid blocking event loop)
    if config.enable_perplexity and docs:
        docs_before = docs
        docs = await asyncio.to_thread(perplexity.run, docs, config)
        docs = _enforce_min_keep_ratio("perplexity", docs_before, docs, n_input, config)
    n_after_perplexity = len(docs)

    # 6. Relevance filter
    if config.enable_relevance and docs:
        docs_before = docs
        docs = await asyncio.to_thread(relevance.run, docs, config)
        docs = _enforce_min_keep_ratio("relevance", docs_before, docs, n_input, config)
    n_after_relevance = len(docs)

    # 7. Trafilatura re-extraction (async — makes HTTP calls)
    if config.enable_trafilatura and docs:
        docs = await trafilatura_fix.run(docs, config)

    # Save outputs
    await asyncio.to_thread(_save_cleaned_docs, docs, config)

    duration = time.monotonic() - t0
    token_counts = [doc.token_count for doc in docs]

    report = CleaningReport(
        run_id=config.run_id,
        topic=config.topic,
        input_docs=n_input,
        after_near_dedup=n_after_near_dedup,
        after_lang_filter=n_after_lang,
        after_gopher=n_after_gopher,
        after_perplexity=n_after_perplexity,
        after_relevance=n_after_relevance,
        total_tokens=sum(token_counts),
        median_doc_tokens=int(statistics.median(token_counts)) if token_counts else 0,
        cleaning_duration_s=round(duration, 1),
        cleaned_at=datetime.now(timezone.utc),
    )

    await asyncio.to_thread(_save_report, report, config)

    log.info(
        "pipeline.done",
        run_id=config.run_id,
        input=n_input,
        output=n_after_relevance,
        duration_s=report.cleaning_duration_s,
        drop_rate=report.drop_rate,
    )
    return report
