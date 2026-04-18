"""Unit tests for chunking stage using a fake tokenizer (no HF downloads)."""
from __future__ import annotations

from dataset_forge.core.models import Chunk
from dataset_forge.stages import chunk as chunk_stage


def test_chunk_doc_respects_window_and_overlap(sample_cleaned_docs, pretrain_config, fake_tokenizer):
    doc = sample_cleaned_docs[0]
    chunks = chunk_stage.chunk_doc(doc, pretrain_config, fake_tokenizer)

    assert len(chunks) > 1, "long doc should produce multiple chunks"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.token_count <= pretrain_config.chunk_tokens for c in chunks)
    assert all(c.text.strip() for c in chunks), "no empty chunk texts"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_doc_short_text_produces_one_chunk(pretrain_config, fake_tokenizer):
    from dataset_forge.core.models import CleanedDoc

    short = CleanedDoc(
        id="short",
        run_id="r",
        url="u",
        text="just a few words",
        content_hash="h",
    )
    chunks = chunk_stage.chunk_doc(short, pretrain_config, fake_tokenizer)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunk_ids_are_deterministic(sample_cleaned_docs, pretrain_config, fake_tokenizer):
    doc = sample_cleaned_docs[0]
    a = chunk_stage.chunk_doc(doc, pretrain_config, fake_tokenizer)
    b = chunk_stage.chunk_doc(doc, pretrain_config, fake_tokenizer)
    assert [c.id for c in a] == [c.id for c in b]


def test_chunk_run_aggregates_across_docs(sample_cleaned_docs, pretrain_config, fake_tokenizer):
    chunks = chunk_stage.run(sample_cleaned_docs, pretrain_config, tokenizer=fake_tokenizer)
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {d.id for d in sample_cleaned_docs}
