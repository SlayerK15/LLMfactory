"""Fixtures for dataset-forge tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dataset_forge.core.models import CleanedDoc, ForgeConfig, OutputMode


def _make_doc(id: str, text: str, url: str = "https://example.com", run_id: str = "test-run") -> CleanedDoc:
    return CleanedDoc(
        id=id,
        run_id=run_id,
        url=url,
        title="t",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        token_count=int(len(text.split()) * 1.3),
    )


@pytest.fixture
def sample_cleaned_docs() -> list[CleanedDoc]:
    base = (
        "CI/CD pipelines automate testing and deployment so that code reaches production "
        "quickly and safely. Jenkins, GitHub Actions and GitLab CI are common choices. "
        "Each commit runs unit tests, integration tests and security scans before merging. "
    )
    return [_make_doc(f"doc{i}", base * (10 + i), url=f"https://example.com/{i}") for i in range(5)]


@pytest.fixture
def populated_cleaned_dir(tmp_path: Path, sample_cleaned_docs: list[CleanedDoc]) -> Path:
    cleaned = tmp_path / "runs" / "test-run" / "cleaned"
    cleaned.mkdir(parents=True)
    for doc in sample_cleaned_docs:
        (cleaned / f"{doc.id}.md").write_text(doc.text, encoding="utf-8")
        (cleaned / f"{doc.id}.meta.json").write_text(
            json.dumps(
                {
                    "id": doc.id,
                    "run_id": doc.run_id,
                    "url": doc.url,
                    "title": doc.title,
                    "content_hash": doc.content_hash,
                    "token_count": doc.token_count,
                    "extraction_confidence": 1.0,
                }
            ),
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def pretrain_config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=tmp_path,
        output_mode=OutputMode.PRETRAIN,
        chunk_tokens=64,
        chunk_overlap_tokens=8,
        enable_embeddings=False,
    )


class FakeTokenizer:
    """Whitespace tokeniser so unit tests don't need HF downloads."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [hash(tok) & 0xFFFFFF for tok in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(["tok" for _ in ids])


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
