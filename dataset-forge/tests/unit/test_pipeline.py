"""Unit tests for the pipeline with a fake tokenizer + fake QA synth — no network or models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_forge.core.errors import CleanedCorpusNotFoundError, EmptyCorpusError
from dataset_forge.core.models import Chunk, ForgeConfig, OutputMode, QAPair
from dataset_forge.core.pipeline import run_forge
from dataset_forge.stages import chunk as chunk_stage


class FakeSynth:
    async def synth(self, chunks: list[Chunk], max_concurrency: int) -> list[QAPair]:
        return [
            QAPair(
                id=QAPair.make_id(c.id, 0),
                chunk_id=c.id,
                doc_id=c.doc_id,
                run_id=c.run_id,
                instruction=f"What is chunk {c.chunk_index} about?",
                response="It is about DevOps practices and CI/CD pipelines.",
            )
            for c in chunks
        ]


@pytest.fixture(autouse=True)
def _patch_tokenizer(monkeypatch, fake_tokenizer):
    monkeypatch.setattr(chunk_stage, "_load_hf_tokenizer", lambda _model: fake_tokenizer)


async def test_pretrain_mode_writes_jsonl_and_card(populated_cleaned_dir: Path, pretrain_config: ForgeConfig):
    report = await run_forge(pretrain_config)

    assert report.total_chunks > 0
    assert report.train_records + report.eval_records == report.total_chunks
    assert pretrain_config.train_path.exists()
    assert pretrain_config.card_path.exists()

    line = pretrain_config.train_path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert set(rec.keys()) == {"text", "doc_id", "url", "token_count"}

    card = json.loads(pretrain_config.card_path.read_text(encoding="utf-8"))
    assert card["output_mode"] == "pretrain"
    assert card["source_docs"] == 5


async def test_instruct_mode_writes_qa_jsonl(tmp_path: Path, populated_cleaned_dir: Path):
    config = ForgeConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=populated_cleaned_dir,
        output_mode=OutputMode.INSTRUCT,
        chunk_tokens=64,
        chunk_overlap_tokens=8,
        enable_embeddings=False,
    )
    report = await run_forge(config, synth=FakeSynth())

    assert report.qa_pairs > 0
    assert report.train_records + report.eval_records == report.qa_pairs
    line = config.train_path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert set(rec.keys()) == {"instruction", "input", "response", "doc_id"}


async def test_missing_run_raises(tmp_path: Path):
    config = ForgeConfig(
        run_id="missing", topic="t", data_dir=tmp_path, enable_embeddings=False
    )
    with pytest.raises(CleanedCorpusNotFoundError):
        await run_forge(config)


async def test_empty_corpus_raises(tmp_path: Path):
    cleaned = tmp_path / "runs" / "empty-run" / "cleaned"
    cleaned.mkdir(parents=True)
    config = ForgeConfig(
        run_id="empty-run", topic="t", data_dir=tmp_path, enable_embeddings=False
    )
    with pytest.raises(EmptyCorpusError):
        await run_forge(config)
