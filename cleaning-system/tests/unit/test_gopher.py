"""Unit tests for Gopher quality heuristics."""
from __future__ import annotations

import pytest

from cleaning_system.core.models import CleaningConfig, DocRecord
from cleaning_system.stages import gopher
from tests.conftest import _make_doc


@pytest.fixture
def config() -> CleaningConfig:
    return CleaningConfig(
        run_id="r", topic="t",
        enable_perplexity=False, enable_relevance=False, enable_trafilatura=False,
    )


def test_passes_good_doc(config: CleaningConfig):
    docs = [
        _make_doc(
            "d1",
            "The Continuous Integration pipeline automates testing and deployment of software. "
            "Teams use tools like Jenkins and GitHub Actions to run tests on every commit. "
            "This practice ensures that bugs are caught early and the main branch remains stable. "
            "DevOps culture encourages collaboration between development and operations teams. "
            "Monitoring with Prometheus and Grafana gives visibility into system performance.",
        )
    ]
    assert len(gopher.run(docs, config)) == 1


def test_drops_too_short(config: CleaningConfig):
    docs = [_make_doc("d1", "Too short text.")]
    assert gopher.run(docs, config) == []


def test_drops_too_long(config: CleaningConfig):
    long_text = "word " * 200_000
    docs = [_make_doc("d1", long_text)]
    assert gopher.run(docs, config) == []


def test_drops_high_symbol_ratio(config: CleaningConfig):
    # > 10% symbols (#) relative to word count
    base = "The infrastructure pipeline enables automated deployments and monitoring. " * 5
    symbols = "# " * 200
    docs = [_make_doc("d1", symbols + base)]
    assert gopher.run(docs, config) == []


def test_drops_high_ellipsis_ratio(config: CleaningConfig):
    # > 30% of lines end with "..."
    lines = ["This line ends normally.\n"] * 5 + ["Trailing ellipsis...\n"] * 10
    docs = [_make_doc("d1", "".join(lines) * 5)]
    assert gopher.run(docs, config) == []


def test_drops_repetitive_lines(config: CleaningConfig):
    # > 20% duplicate lines
    repetitive = ("Same line repeated\n" * 50) + ("Unique content here\n" * 5)
    docs = [_make_doc("d1", repetitive)]
    assert gopher.run(docs, config) == []


def test_drops_no_stop_words(config: CleaningConfig):
    # Text with no English stop words — just code-like tokens
    code_text = "func main() { } var x int := 0; return nil } " * 50
    docs = [_make_doc("d1", code_text)]
    assert gopher.run(docs, config) == []


def test_empty_input(config: CleaningConfig):
    assert gopher.run([], config) == []


def test_keeps_all_good_docs(config: CleaningConfig, sample_docs):
    kept = gopher.run(sample_docs, config)
    assert len(kept) == len(sample_docs)
