"""Domain models for cleaning-system."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class DocRecord(BaseModel):
    """In-memory representation of a scraped document during cleaning."""

    id: str
    run_id: str
    url: str
    title: str | None = None
    text: str
    content_hash: str
    token_count: int = 0
    extraction_confidence: float = 1.0
    path: Path

    @classmethod
    def from_files(cls, md_path: Path, meta_path: Path) -> "DocRecord":
        """Load a DocRecord from the filesystem paths produced by collection-system."""
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
            path=md_path,
        )

    def with_text(self, new_text: str) -> "DocRecord":
        """Return a copy with updated text (and recalculated hash + token_count)."""
        new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
        return self.model_copy(
            update={
                "text": new_text,
                "content_hash": new_hash,
                "token_count": int(len(new_text.split()) * 1.3),
            }
        )


class CleaningConfig(BaseModel):
    """Configuration for a single cleaning run."""

    run_id: str
    topic: str
    data_dir: Path = Path("./data")

    # stage toggles
    enable_perplexity: bool = True
    enable_relevance: bool = True
    enable_trafilatura: bool = True

    # near-dup
    near_dup_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    near_dup_num_perm: int = 128
    near_dup_shingle_size: int = 5

    # language filter
    target_language: str = "en"
    lang_min_text_len: int = 200

    # Gopher heuristics
    min_words: int = 50
    max_words: int = 100_000
    mean_word_len_min: float = 3.0
    mean_word_len_max: float = 10.0
    max_symbol_word_ratio: float = 0.1
    max_ellipsis_line_ratio: float = 0.3
    max_duplicate_line_ratio: float = 0.2
    min_stopword_density: float = 0.02

    # perplexity filter
    perplexity_model_id: str = "Qwen/Qwen2.5-0.5B"
    perplexity_low_pct: float = Field(default=0.05, ge=0.0, le=0.5)
    perplexity_high_pct: float = Field(default=0.10, ge=0.0, le=0.5)
    perplexity_max_tokens: int = 512

    # relevance filter
    relevance_model_id: str = "BAAI/bge-m3"
    relevance_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    relevance_batch_size: int = 64

    # trafilatura re-extraction
    confidence_threshold_for_reextract: float = 0.80
    trafilatura_timeout_s: int = 20

    @property
    def input_docs_dir(self) -> Path:
        return self.data_dir / "runs" / self.run_id / "docs"

    @property
    def cleaned_docs_dir(self) -> Path:
        return self.data_dir / "runs" / self.run_id / "cleaned"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


class CleaningReport(BaseModel):
    """Quality report produced after a cleaning run — the core feedback loop."""

    run_id: str
    topic: str
    input_docs: int
    after_near_dedup: int
    after_lang_filter: int
    after_gopher: int
    after_perplexity: int
    after_relevance: int
    total_tokens: int
    median_doc_tokens: int
    cleaning_duration_s: float
    cleaned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def total_docs_cleaned(self) -> int:
        return self.after_relevance

    @computed_field
    @property
    def drop_rate(self) -> float:
        if self.input_docs == 0:
            return 0.0
        return round(1.0 - self.after_relevance / self.input_docs, 4)
