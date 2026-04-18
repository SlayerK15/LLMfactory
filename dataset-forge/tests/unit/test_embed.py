"""Unit tests for the embedding stage using a fake encoder + sink — no models or LanceDB."""
from __future__ import annotations

from typing import Any

import numpy as np

from dataset_forge.core.models import Chunk, ForgeConfig
from dataset_forge.stages import embed as embed_stage


class FakeEncoder:
    """Returns a deterministic unit vector per text; records call args."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[tuple[list[str], int]] = []

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls.append((list(texts), batch_size))
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs[i, hash(t) % self.dim] = 1.0
        return vecs


class FakeSink:
    """In-memory VectorSink that captures the records it was asked to upsert."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def upsert(self, records: list[dict[str, Any]]) -> int:
        self.records.extend(records)
        return len(records)


def _chunk(doc_id: str, idx: int, text: str = "some passage about CI/CD") -> Chunk:
    return Chunk(
        id=Chunk.make_id(doc_id, idx),
        doc_id=doc_id,
        run_id="r",
        url="https://example.com",
        text=text,
        token_count=len(text.split()),
        chunk_index=idx,
    )


def _config(tmp_path) -> ForgeConfig:
    return ForgeConfig(
        run_id="r",
        topic="t",
        data_dir=tmp_path,
        enable_embeddings=True,
        embed_batch_size=4,
    )


def test_embed_returns_zero_for_empty_input(tmp_path):
    enc = FakeEncoder()
    sink = FakeSink()
    n = embed_stage.run([], _config(tmp_path), encoder=enc, sink=sink)
    assert n == 0
    assert enc.calls == []
    assert sink.records == []


def test_embed_upserts_one_record_per_chunk(tmp_path):
    chunks = [_chunk("d1", 0), _chunk("d1", 1), _chunk("d2", 0)]
    enc = FakeEncoder(dim=16)
    sink = FakeSink()

    n = embed_stage.run(chunks, _config(tmp_path), encoder=enc, sink=sink)

    assert n == 3
    assert len(sink.records) == 3
    assert {r["id"] for r in sink.records} == {c.id for c in chunks}


def test_embed_record_fields_match_chunk_and_include_vector(tmp_path):
    chunks = [_chunk("d1", 0, text="Docker and Kubernetes rule the roost")]
    enc = FakeEncoder(dim=8)
    sink = FakeSink()

    embed_stage.run(chunks, _config(tmp_path), encoder=enc, sink=sink)

    rec = sink.records[0]
    assert rec["doc_id"] == "d1"
    assert rec["run_id"] == "r"
    assert rec["url"] == "https://example.com"
    assert rec["text"].startswith("Docker")
    assert rec["token_count"] > 0
    assert isinstance(rec["vector"], list)
    assert len(rec["vector"]) == 8
    # exactly one hot dimension from FakeEncoder
    assert sum(rec["vector"]) == 1.0


def test_embed_passes_configured_batch_size(tmp_path):
    chunks = [_chunk("d1", i) for i in range(10)]
    enc = FakeEncoder()
    sink = FakeSink()
    config = _config(tmp_path)

    embed_stage.run(chunks, config, encoder=enc, sink=sink)

    assert len(enc.calls) == 1
    _, batch_size = enc.calls[0]
    assert batch_size == config.embed_batch_size
