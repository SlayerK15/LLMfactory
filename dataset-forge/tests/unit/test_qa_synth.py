"""Unit tests for the QA-synth orchestration stage with a fake synthesiser."""
from __future__ import annotations

import pytest

from dataset_forge.core.models import Chunk, ForgeConfig, OutputMode, QAPair
from dataset_forge.stages import qa_synth


class FakeSynth:
    """Emits N QAPairs per chunk; records concurrency arg."""

    def __init__(self, pairs_per_chunk: int = 2, raise_on: str | None = None) -> None:
        self.pairs_per_chunk = pairs_per_chunk
        self.raise_on = raise_on
        self.received_concurrency: int | None = None

    async def synth(self, chunks: list[Chunk], max_concurrency: int) -> list[QAPair]:
        self.received_concurrency = max_concurrency
        out: list[QAPair] = []
        for c in chunks:
            if self.raise_on and self.raise_on in c.text:
                continue  # synth may drop a chunk but shouldn't blow the run
            for i in range(self.pairs_per_chunk):
                out.append(
                    QAPair(
                        id=QAPair.make_id(c.id, i),
                        chunk_id=c.id,
                        doc_id=c.doc_id,
                        run_id=c.run_id,
                        instruction=f"Q{i} on {c.id}",
                        response=f"A{i} grounded in {c.text[:20]}",
                    )
                )
        return out


def _chunk(doc_id: str, idx: int, text: str = "CI/CD is the backbone of DevOps") -> Chunk:
    return Chunk(
        id=Chunk.make_id(doc_id, idx),
        doc_id=doc_id,
        run_id="r",
        url="u",
        text=text,
        token_count=len(text.split()),
        chunk_index=idx,
    )


def _config(**kwargs) -> ForgeConfig:
    defaults = dict(
        run_id="r",
        topic="DevOps",
        output_mode=OutputMode.INSTRUCT,
        qa_max_concurrency=3,
        enable_embeddings=False,
    )
    defaults.update(kwargs)
    return ForgeConfig(**defaults)


@pytest.mark.asyncio
async def test_qa_synth_produces_expected_pair_count():
    chunks = [_chunk("d1", 0), _chunk("d1", 1), _chunk("d2", 0)]
    synth = FakeSynth(pairs_per_chunk=3)

    pairs = await qa_synth.run(chunks, _config(), synth)

    assert len(pairs) == 9  # 3 chunks × 3 pairs


@pytest.mark.asyncio
async def test_qa_synth_passes_concurrency_from_config():
    chunks = [_chunk("d1", 0)]
    synth = FakeSynth()

    await qa_synth.run(chunks, _config(qa_max_concurrency=7), synth)

    assert synth.received_concurrency == 7


@pytest.mark.asyncio
async def test_qa_synth_preserves_chunk_linkage():
    chunks = [_chunk("docA", 0), _chunk("docB", 0)]
    synth = FakeSynth(pairs_per_chunk=1)

    pairs = await qa_synth.run(chunks, _config(), synth)

    pair_by_chunk = {p.chunk_id: p for p in pairs}
    assert set(pair_by_chunk.keys()) == {c.id for c in chunks}
    for c in chunks:
        assert pair_by_chunk[c.id].doc_id == c.doc_id
        assert pair_by_chunk[c.id].run_id == c.run_id


@pytest.mark.asyncio
async def test_qa_synth_empty_chunks_returns_empty():
    synth = FakeSynth()
    pairs = await qa_synth.run([], _config(), synth)
    assert pairs == []


@pytest.mark.asyncio
async def test_qa_synth_tolerates_synth_dropping_some_chunks():
    chunks = [_chunk("d1", 0, text="good"), _chunk("d2", 0, text="bad-drop-me")]
    synth = FakeSynth(pairs_per_chunk=2, raise_on="bad-drop-me")

    pairs = await qa_synth.run(chunks, _config(), synth)

    assert len(pairs) == 2  # only the good chunk contributed
    assert all(p.doc_id == "d1" for p in pairs)
