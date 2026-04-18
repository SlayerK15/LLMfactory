"""Unit tests for MinHash LSH near-dedup stage."""
from __future__ import annotations

import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.stages import near_dedup
from tests.conftest import _make_doc


@pytest.fixture
def config() -> CleaningConfig:
    return CleaningConfig(
        run_id="r", topic="t",
        enable_perplexity=False, enable_relevance=False, enable_trafilatura=False,
        near_dup_threshold=0.8,
    )


def test_keeps_all_unique_docs(config: CleaningConfig, sample_docs):
    kept = near_dedup.run(sample_docs, config)
    assert len(kept) == len(sample_docs)


def test_drops_exact_duplicate(config: CleaningConfig):
    text = (
        "Continuous integration automates testing. Kubernetes orchestrates containers. "
        "Terraform manages infrastructure declaratively. Monitoring ensures reliability. "
        "Git enables version control collaboration." * 3
    )
    docs = [_make_doc("d1", text), _make_doc("d2", text)]
    kept = near_dedup.run(docs, config)
    assert len(kept) == 1
    assert kept[0].id == "d1"


def test_drops_near_duplicate(config: CleaningConfig):
    # Long enough that changing one word makes Jaccard >> 0.8
    sentence = (
        "Continuous integration automates the testing and deployment of software applications. "
        "Teams use CI pipelines to verify code quality on every commit to the repository. "
        "This ensures the main branch remains stable and deployable at all times. "
        "Kubernetes orchestrates containers at scale across cloud environments. "
        "Terraform manages infrastructure declaratively as version-controlled code. "
    )
    base = sentence * 6  # ~300 words → ~295 5-grams
    # Replace one word once — changes ≤5 shingles out of ~295; Jaccard ≈ 0.98
    near = base.replace("applications.", "programs.", 1)
    docs = [_make_doc("d1", base), _make_doc("d2", near)]
    kept = near_dedup.run(docs, config)
    assert len(kept) == 1


def test_keeps_different_docs(config: CleaningConfig):
    docs = [
        _make_doc("d1", "Docker containers isolate application dependencies and runtimes. " * 10),
        _make_doc(
            "d2",
            "Terraform is an infrastructure as code tool by HashiCorp for cloud provisioning. " * 10,
        ),
    ]
    kept = near_dedup.run(docs, config)
    assert len(kept) == 2


def test_empty_input(config: CleaningConfig):
    assert near_dedup.run([], config) == []


def test_single_doc(config: CleaningConfig):
    docs = [_make_doc("d1", "Only document. " * 20)]
    assert len(near_dedup.run(docs, config)) == 1
