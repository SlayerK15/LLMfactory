"""
Trafilatura re-extraction for low-confidence documents.
Documents flagged with extraction_confidence < threshold are re-fetched
and re-extracted with Trafilatura as a quality improvement pass.
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from cleaning_system.core.models import CleaningConfig, DocRecord

log = structlog.get_logger()


async def _reextract_one(
    doc: DocRecord,
    client: httpx.AsyncClient,
    timeout_s: int,
) -> DocRecord:
    """Re-fetch URL and re-extract text. Returns original doc on any failure."""
    try:
        import trafilatura

        response = await asyncio.wait_for(
            client.get(doc.url),
            timeout=timeout_s,
        )
        if response.status_code >= 400:
            return doc

        extracted = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
        )
        if not extracted or len(extracted.split()) < 50:
            return doc

        improved = doc.with_text(extracted)
        log.debug(
            "trafilatura.reextracted",
            doc_id=doc.id,
            old_tokens=doc.token_count,
            new_tokens=improved.token_count,
        )
        return improved

    except Exception as exc:
        log.debug("trafilatura.skip", doc_id=doc.id, reason=str(exc)[:100])
        return doc


async def run(docs: list[DocRecord], config: CleaningConfig) -> list[DocRecord]:
    """
    Re-extract low-confidence documents with Trafilatura.
    Runs concurrently; falls back to the original doc on any failure.
    Skipped (returns docs unchanged) if trafilatura is not installed.
    """
    if not docs:
        return docs

    try:
        import trafilatura  # noqa: F401 — presence check
    except ImportError:
        log.warning("trafilatura.skip", reason="trafilatura not installed")
        return docs

    low_conf = [
        (i, doc)
        for i, doc in enumerate(docs)
        if doc.extraction_confidence < config.confidence_threshold_for_reextract
    ]

    if not low_conf:
        log.info("trafilatura.skip", reason="no low-confidence docs")
        return docs

    log.info(
        "trafilatura.start",
        total_docs=len(docs),
        to_reextract=len(low_conf),
        threshold=config.confidence_threshold_for_reextract,
    )

    result = list(docs)

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; cleaning-system/0.1)",
            "Accept": "text/html,application/xhtml+xml",
        },
        timeout=config.trafilatura_timeout_s,
    ) as client:
        tasks = [
            _reextract_one(doc, client, config.trafilatura_timeout_s)
            for _, doc in low_conf
        ]
        improved = await asyncio.gather(*tasks, return_exceptions=False)

    for (orig_idx, _), new_doc in zip(low_conf, improved):
        if isinstance(new_doc, DocRecord):
            result[orig_idx] = new_doc

    improved_count = sum(
        1
        for (_, old), new in zip(low_conf, improved)
        if isinstance(new, DocRecord) and new.token_count > old.token_count
    )
    log.info(
        "trafilatura.done",
        reextracted=len(low_conf),
        improved=improved_count,
    )
    return result
