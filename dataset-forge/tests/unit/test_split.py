"""Unit tests for train/eval split."""
from __future__ import annotations

from dataset_forge.core.models import Chunk, ForgeConfig, OutputMode
from dataset_forge.stages import split as split_stage


def _chunk(doc_id: str, idx: int) -> Chunk:
    return Chunk(
        id=f"{doc_id}:{idx}",
        doc_id=doc_id,
        run_id="r",
        url="u",
        text="t",
        token_count=1,
        chunk_index=idx,
    )


def test_split_is_deterministic():
    config = ForgeConfig(run_id="r", topic="t", output_mode=OutputMode.PRETRAIN, eval_fraction=0.2, split_seed=7)
    chunks = [_chunk(f"d{i}", 0) for i in range(100)]
    a_train, a_eval = split_stage.split_by_doc(chunks, config)
    b_train, b_eval = split_stage.split_by_doc(chunks, config)
    assert [c.id for c in a_train] == [c.id for c in b_train]
    assert [c.id for c in a_eval] == [c.id for c in b_eval]


def test_split_respects_eval_fraction_roughly():
    config = ForgeConfig(run_id="r", topic="t", output_mode=OutputMode.PRETRAIN, eval_fraction=0.20, split_seed=1)
    chunks = [_chunk(f"d{i}", 0) for i in range(500)]
    train, eval_ = split_stage.split_by_doc(chunks, config)
    assert len(train) + len(eval_) == 500
    frac = len(eval_) / 500
    assert 0.15 < frac < 0.25, f"expected ~20% eval, got {frac:.2%}"


def test_split_keeps_all_chunks_of_doc_together():
    config = ForgeConfig(run_id="r", topic="t", output_mode=OutputMode.PRETRAIN, eval_fraction=0.5, split_seed=42)
    chunks = [_chunk(f"d{i}", j) for i in range(20) for j in range(5)]
    train, eval_ = split_stage.split_by_doc(chunks, config)
    train_docs = {c.doc_id for c in train}
    eval_docs = {c.doc_id for c in eval_}
    assert train_docs.isdisjoint(eval_docs)
