"""
Relevance filter using BGE-M3 embeddings.
Embeds all documents and the topic string, then drops documents whose
cosine similarity to the topic centroid falls below the threshold.
"""
from __future__ import annotations

import structlog

from cleaning_system.core.errors import ModelLoadError
from cleaning_system.core.models import CleaningConfig, DocRecord

log = structlog.get_logger()

_cached_model = None
_cached_model_id: str | None = None


def _get_model(model_id: str):
    global _cached_model, _cached_model_id
    if _cached_model is None or _cached_model_id != model_id:
        try:
            from sentence_transformers import SentenceTransformer

            log.info("relevance.load_model", model_id=model_id)
            _cached_model = SentenceTransformer(model_id)
            _cached_model_id = model_id
            log.info("relevance.model_ready", model_id=model_id)
        except Exception as exc:
            raise ModelLoadError(model_id, str(exc)) from exc
    return _cached_model


def run(docs: list[DocRecord], config: CleaningConfig) -> list[DocRecord]:
    """
    Drop documents with cosine similarity to the topic below relevance_threshold.
    Skipped (returns docs unchanged) if sentence-transformers is not installed.
    """
    if not docs:
        return docs

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        log.warning("relevance.skip", reason="sentence-transformers not installed")
        return docs

    model = _get_model(config.relevance_model_id)

    texts = [doc.text[:2000] for doc in docs]

    log.info("relevance.encoding", n_docs=len(docs), model=config.relevance_model_id)
    doc_embeddings = model.encode(
        texts,
        batch_size=config.relevance_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    topic_embedding = model.encode(
        [config.topic],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Since embeddings are L2-normalized, dot product == cosine similarity
    similarities = doc_embeddings @ topic_embedding

    kept = [
        doc
        for doc, sim in zip(docs, similarities)
        if float(sim) >= config.relevance_threshold
    ]

    dropped = len(docs) - len(kept)
    log.info(
        "relevance.done",
        input=len(docs),
        kept=len(kept),
        dropped=dropped,
        threshold=config.relevance_threshold,
    )
    return kept
