"""Language detection filter — keep only documents in the target language."""
from __future__ import annotations

import structlog
from langdetect import DetectorFactory, LangDetectException, detect

from cleaning_system.core.models import CleaningConfig, DocRecord

log = structlog.get_logger()

# Make langdetect deterministic across runs
DetectorFactory.seed = 42


def run(docs: list[DocRecord], config: CleaningConfig) -> list[DocRecord]:
    """
    Keep documents whose detected language matches config.target_language.
    Documents that are too short for reliable detection are dropped.
    """
    if not docs:
        return docs

    kept: list[DocRecord] = []
    dropped_lang = 0
    dropped_short = 0
    dropped_error = 0

    for doc in docs:
        sample = doc.text[: config.lang_min_text_len * 5]

        if len(doc.text.split()) < 20:
            dropped_short += 1
            continue

        try:
            lang = detect(sample)
        except LangDetectException:
            dropped_error += 1
            continue

        if lang != config.target_language:
            dropped_lang += 1
            log.debug("lang_filter.drop", doc_id=doc.id, detected=lang)
            continue

        kept.append(doc)

    log.info(
        "lang_filter.done",
        input=len(docs),
        kept=len(kept),
        dropped_lang=dropped_lang,
        dropped_short=dropped_short,
        dropped_error=dropped_error,
    )
    return kept
