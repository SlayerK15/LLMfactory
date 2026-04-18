"""Unit tests for the perplexity stage — monkeypatches the scoring model."""
from __future__ import annotations

import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.stages import perplexity
from tests.conftest import _make_doc


class _FakeModel:
    """Returns a pre-canned score based on doc id → simulates model output."""

    _model_id = "fake-perplexity"

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores = scores_by_id
        self._current_doc_id: str | None = None

    def set_current(self, doc_id: str) -> None:
        self._current_doc_id = doc_id

    def compute(self, text: str) -> float:
        return self._scores.get(self._current_doc_id or "", 100.0)


def _config(**kwargs) -> CleaningConfig:
    defaults = dict(
        run_id="r",
        topic="t",
        enable_perplexity=True,
        enable_relevance=False,
        enable_trafilatura=False,
        perplexity_low_pct=0.10,
        perplexity_high_pct=0.10,
    )
    defaults.update(kwargs)
    return CleaningConfig(**defaults)


def _install_fake_model(monkeypatch, scores_by_id: dict[str, float]) -> _FakeModel:
    """
    Swap the module-level _get_model with one that returns a fake that tracks
    which doc is being scored (so we can emit per-doc synthetic scores).
    """
    fake = _FakeModel(scores_by_id)

    # Patch _get_model to return our fake; patch `compute` to read the current doc.
    def _fake_get_model(config):
        return fake

    # Wrap compute so the caller sets the current doc id before each call.
    original_compute = fake.compute
    doc_iter: list[str] = []

    def _wrapped_compute(text: str) -> float:
        # pop the next id from the queue (test sets queue order explicitly)
        if doc_iter:
            fake.set_current(doc_iter.pop(0))
        return original_compute(text)

    fake.compute = _wrapped_compute  # type: ignore[method-assign]
    fake._id_queue = doc_iter  # type: ignore[attr-defined]

    monkeypatch.setattr(perplexity, "_get_model", _fake_get_model)
    return fake


def test_perplexity_drops_both_tails(monkeypatch):
    """Very-low and very-high scores fall in the bottom/top trim percentiles."""
    scores = {
        "lo": 1.0,       # bottom tail — drop
        "hi": 10_000.0,  # top tail — drop
        "m1": 10.0,
        "m2": 12.0,
        "m3": 15.0,
        "m4": 18.0,
        "m5": 20.0,
        "m6": 22.0,
        "m7": 25.0,
        "m8": 30.0,
    }
    docs = [_make_doc(k, f"{k} doc body " * 20) for k in scores.keys()]

    fake = _install_fake_model(monkeypatch, scores)
    fake._id_queue.extend(scores.keys())  # type: ignore[attr-defined]

    kept = perplexity.run(docs, _config(perplexity_low_pct=0.10, perplexity_high_pct=0.10))
    kept_ids = {d.id for d in kept}

    assert "lo" not in kept_ids
    assert "hi" not in kept_ids
    # All middle docs survive
    assert kept_ids == {f"m{i}" for i in range(1, 9)}


def test_perplexity_empty_input():
    assert perplexity.run([], _config()) == []


def test_perplexity_too_few_valid_returns_unchanged(monkeypatch):
    """With <4 valid scores the stage bails out and returns docs unchanged."""
    scores = {"d1": 10.0, "d2": 12.0}
    docs = [_make_doc(k, "text " * 20) for k in scores.keys()]

    fake = _install_fake_model(monkeypatch, scores)
    fake._id_queue.extend(scores.keys())  # type: ignore[attr-defined]

    kept = perplexity.run(docs, _config())
    assert len(kept) == 2


def test_perplexity_score_error_is_logged_and_skipped(monkeypatch):
    """A compute exception for one doc produces a NaN score and should not crash the stage."""
    # 10 docs total so after NaN-ing one we still have 9 valid > 4.
    scores = {f"d{i}": 10.0 + i for i in range(10)}
    docs = [_make_doc(k, "text " * 20) for k in scores.keys()]

    fake = _install_fake_model(monkeypatch, scores)

    # Make d5 raise mid-pipeline
    original = fake.compute

    def _flaky(text: str) -> float:
        if fake._id_queue and fake._id_queue[0] == "d5":  # type: ignore[attr-defined]
            fake._id_queue.pop(0)  # consume
            raise RuntimeError("boom")
        return original(text)

    fake.compute = _flaky  # type: ignore[method-assign]
    fake._id_queue.extend(scores.keys())  # type: ignore[attr-defined]

    kept = perplexity.run(docs, _config(perplexity_low_pct=0.0, perplexity_high_pct=0.0))
    # d5 gets NaN which is excluded from percentile bounds but still indexed;
    # with 0% trim on both ends, lo=min, hi=max, and d5's NaN fails the `lo <= s <= hi` check.
    kept_ids = {d.id for d in kept}
    assert "d5" not in kept_ids
    assert len(kept_ids) == 9
