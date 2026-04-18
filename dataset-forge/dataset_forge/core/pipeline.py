"""Core orchestration: cleaned docs → chunks → (optional Q/A) → split + write JSONL + card."""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Protocol

import structlog

from dataset_forge.core.errors import CleanedCorpusNotFoundError, EmptyCorpusError
from dataset_forge.core.models import (
    Chunk,
    CleanedDoc,
    DatasetCard,
    ForgeConfig,
    ForgeReport,
    OutputMode,
    QAPair,
)
from dataset_forge.stages import chunk as chunk_stage
from dataset_forge.stages import embed as embed_stage
from dataset_forge.stages import qa_synth as qa_stage
from dataset_forge.stages import split as split_stage

log = structlog.get_logger()


class QASynthesiser(Protocol):
    async def synth(self, chunks: list[Chunk], max_concurrency: int) -> list[QAPair]: ...


def _load_cleaned(config: ForgeConfig) -> list[CleanedDoc]:
    cleaned_dir = config.cleaned_dir
    if not cleaned_dir.exists():
        raise CleanedCorpusNotFoundError(config.run_id)
    docs: list[CleanedDoc] = []
    for md_path in sorted(cleaned_dir.glob("*.md")):
        meta_path = md_path.with_suffix(".meta.json")
        if not meta_path.exists():
            log.warning("forge.missing_meta", path=str(md_path))
            continue
        try:
            docs.append(CleanedDoc.from_files(md_path, meta_path))
        except Exception as exc:
            log.warning("forge.load_error", path=str(md_path), error=str(exc))
    log.info("forge.loaded", run_id=config.run_id, n_docs=len(docs))
    return docs


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def _chunk_record(c: Chunk) -> dict[str, object]:
    return {"text": c.text, "doc_id": c.doc_id, "url": c.url, "token_count": c.token_count}


def _qa_record(qa: QAPair) -> dict[str, object]:
    return {"instruction": qa.instruction, "input": qa.input, "response": qa.response, "doc_id": qa.doc_id}


async def run_forge(
    config: ForgeConfig,
    synth: QASynthesiser | None = None,
) -> ForgeReport:
    t0 = time.monotonic()
    log.info("forge.start", run_id=config.run_id, mode=config.output_mode.value)

    # 1. Load cleaned docs
    docs = await asyncio.to_thread(_load_cleaned, config)
    if not docs:
        raise EmptyCorpusError(f"no cleaned docs for run_id={config.run_id}")

    # 2. Chunk
    chunks = await asyncio.to_thread(chunk_stage.run, docs, config)
    if not chunks:
        raise EmptyCorpusError("chunking produced zero chunks")

    qa_pairs: list[QAPair] = []
    train_count = 0
    eval_count = 0

    # 3. Branch on output mode
    if config.output_mode is OutputMode.PRETRAIN:
        train_chunks, eval_chunks = split_stage.split_by_doc(chunks, config)
        await asyncio.to_thread(
            _write_jsonl, config.train_path, [_chunk_record(c) for c in train_chunks]
        )
        await asyncio.to_thread(
            _write_jsonl, config.eval_path, [_chunk_record(c) for c in eval_chunks]
        )
        train_count = len(train_chunks)
        eval_count = len(eval_chunks)
    else:
        if synth is None:
            from dataset_forge.adapters.llm.groq_adapter import GroqQASynth

            synth = GroqQASynth(config)
        qa_pairs = await qa_stage.run(chunks, config, synth)
        if not qa_pairs:
            raise EmptyCorpusError("QA synthesis produced zero pairs")
        train_qa, eval_qa = split_stage.split_by_doc(qa_pairs, config)
        await asyncio.to_thread(
            _write_jsonl, config.train_path, [_qa_record(q) for q in train_qa]
        )
        await asyncio.to_thread(
            _write_jsonl, config.eval_path, [_qa_record(q) for q in eval_qa]
        )
        train_count = len(train_qa)
        eval_count = len(eval_qa)

    # 4. Embeddings (optional) — always over chunks, regardless of output mode
    if config.enable_embeddings:
        try:
            await asyncio.to_thread(embed_stage.run, chunks, config)
        except Exception as exc:
            log.warning("forge.embed_failed", error=str(exc))

    # 5. Card
    token_counts = [c.token_count for c in chunks]
    card = DatasetCard(
        run_id=config.run_id,
        topic=config.topic,
        output_mode=config.output_mode,
        source_docs=len(docs),
        total_chunks=len(chunks),
        qa_pairs=len(qa_pairs),
        train_records=train_count,
        eval_records=eval_count,
        total_tokens=sum(token_counts),
        median_chunk_tokens=int(statistics.median(token_counts)) if token_counts else 0,
        tokenizer=config.tokenizer_model_id,
        embed_model=config.embed_model_id if config.enable_embeddings else None,
        lancedb_table=config.lancedb_table if config.enable_embeddings else None,
    )
    config.datasets_dir.mkdir(parents=True, exist_ok=True)
    config.card_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

    duration = time.monotonic() - t0
    report = ForgeReport(
        run_id=config.run_id,
        topic=config.topic,
        output_mode=config.output_mode,
        source_docs=len(docs),
        total_chunks=len(chunks),
        qa_pairs=len(qa_pairs),
        train_records=train_count,
        eval_records=eval_count,
        total_tokens=sum(token_counts),
        duration_s=round(duration, 2),
    )
    log.info("forge.done", **report.model_dump(exclude={"topic"}))
    return report
