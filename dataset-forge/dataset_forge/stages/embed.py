"""Embedding stage — BGE-M3 → LanceDB. Pluggable via Encoder protocol for testability."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import structlog

from dataset_forge.core.models import Chunk, ForgeConfig

if TYPE_CHECKING:
    import numpy as np

log = structlog.get_logger()


class Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int) -> "np.ndarray": ...


class VectorSink(Protocol):
    def upsert(self, records: list[dict[str, object]]) -> int: ...


@lru_cache(maxsize=2)
def _load_sbert(model_id: str) -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


class SBertEncoder:
    def __init__(self, model_id: str) -> None:
        self._model = _load_sbert(model_id)

    def encode(self, texts: list[str], batch_size: int) -> "np.ndarray":
        return self._model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def run(
    chunks: list[Chunk],
    config: ForgeConfig,
    encoder: Encoder | None = None,
    sink: VectorSink | None = None,
) -> int:
    if not chunks:
        return 0
    enc = encoder if encoder is not None else SBertEncoder(config.embed_model_id)
    if sink is None:
        from dataset_forge.adapters.vector.lancedb_adapter import LanceDBSink

        sink = LanceDBSink(config.lancedb_path, config.lancedb_table)
    texts = [c.text for c in chunks]
    vectors = enc.encode(texts, batch_size=config.embed_batch_size)
    records = [
        {
            "id": c.id,
            "doc_id": c.doc_id,
            "run_id": c.run_id,
            "url": c.url,
            "text": c.text,
            "token_count": c.token_count,
            "vector": vec.tolist() if hasattr(vec, "tolist") else list(vec),
        }
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    n = sink.upsert(records)
    log.info("stage.embed.done", chunks=len(chunks), written=n)
    return n
