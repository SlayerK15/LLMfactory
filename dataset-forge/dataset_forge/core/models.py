"""Domain models for dataset-forge."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class OutputMode(str, Enum):
    PRETRAIN = "pretrain"
    INSTRUCT = "instruct"


class CleanedDoc(BaseModel):
    """A doc loaded from cleaning-system's `cleaned/` output directory."""

    id: str
    run_id: str
    url: str
    title: str | None = None
    text: str
    content_hash: str
    token_count: int = 0
    extraction_confidence: float = 1.0

    @classmethod
    def from_files(cls, md_path: Path, meta_path: Path) -> "CleanedDoc":
        text = md_path.read_text(encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            id=meta["id"],
            run_id=meta["run_id"],
            url=meta["url"],
            title=meta.get("title"),
            text=text,
            content_hash=meta["content_hash"],
            token_count=meta.get("token_count", 0),
            extraction_confidence=meta.get("extraction_confidence", 1.0),
        )


class Chunk(BaseModel):
    """A fixed-size chunk of a cleaned doc — the atomic unit of training data."""

    id: str
    doc_id: str
    run_id: str
    url: str
    text: str
    token_count: int
    chunk_index: int

    @staticmethod
    def make_id(doc_id: str, chunk_index: int) -> str:
        return hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()[:16]


class QAPair(BaseModel):
    """One instruction-tuning record synthesised from a chunk."""

    id: str
    chunk_id: str
    doc_id: str
    run_id: str
    instruction: str
    input: str = ""
    response: str

    @staticmethod
    def make_id(chunk_id: str, idx: int) -> str:
        return hashlib.sha256(f"{chunk_id}:qa:{idx}".encode()).hexdigest()[:16]


class ForgeConfig(BaseModel):
    """Configuration for a single dataset-forge run."""

    run_id: str
    topic: str
    data_dir: Path = Path("./data")
    output_mode: OutputMode = OutputMode.PRETRAIN

    # chunking
    chunk_tokens: int = 2048
    chunk_overlap_tokens: int = 200
    tokenizer_model_id: str = "BAAI/bge-m3"

    # Q/A synth
    qa_pairs_per_chunk: int = 3
    qa_batch_chunks: int = 20
    qa_model: str = "llama-3.3-70b-versatile"
    qa_max_concurrency: int = 4
    qa_timeout_s: int = 60
    qa_min_instruction_len: int = 10
    qa_min_response_len: int = 20

    # embeddings / vector store
    enable_embeddings: bool = True
    embed_model_id: str = "BAAI/bge-m3"
    embed_batch_size: int = 64
    lancedb_table: str = "chunks"

    # split
    eval_fraction: float = Field(default=0.10, ge=0.0, le=0.5)
    split_seed: int = 42

    @property
    def cleaned_dir(self) -> Path:
        return self.data_dir / "runs" / self.run_id / "cleaned"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def train_path(self) -> Path:
        return self.datasets_dir / f"{self.run_id}.train.jsonl"

    @property
    def eval_path(self) -> Path:
        return self.datasets_dir / f"{self.run_id}.eval.jsonl"

    @property
    def card_path(self) -> Path:
        return self.datasets_dir / f"{self.run_id}.card.json"

    @property
    def lancedb_path(self) -> Path:
        return self.data_dir / "lancedb"


class DatasetCard(BaseModel):
    run_id: str
    topic: str
    output_mode: OutputMode
    source_docs: int
    total_chunks: int
    qa_pairs: int = 0
    train_records: int
    eval_records: int
    total_tokens: int
    median_chunk_tokens: int
    tokenizer: str
    embed_model: str | None
    lancedb_table: str | None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForgeReport(BaseModel):
    run_id: str
    topic: str
    output_mode: OutputMode
    source_docs: int
    total_chunks: int
    qa_pairs: int
    train_records: int
    eval_records: int
    total_tokens: int
    duration_s: float

    @computed_field
    @property
    def records_per_second(self) -> float:
        if self.duration_s == 0:
            return 0.0
        return round((self.train_records + self.eval_records) / self.duration_s, 2)
