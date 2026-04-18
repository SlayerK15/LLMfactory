"""Unit tests for the relevance stage — injects a fake SentenceTransformer so no model is loaded."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.stages import relevance
from tests.conftest import _make_doc


class _FakeEncoder:
    """
    Deterministic encoder: documents about 'DevOps' get similarity ~0.9 to the topic;
    unrelated docs get ~0.1. Good enough to drive the threshold branch.
    """

    def __init__(self, relevant_kw: str = "devops") -> None:
        self._kw = relevant_kw.lower()

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    ) -> np.ndarray:
        # topic vector points purely along axis 0; relevant docs lean on axis 0, others on axis 1
        vecs = []
        for t in texts:
            s = t.lower()
            if self._kw in s:
                vecs.append([0.95, 0.312])  # cos ≈ 0.95 to [1,0]
            elif t.lower() == self._kw:
                vecs.append([1.0, 0.0])  # the topic itself
            else:
                vecs.append([0.1, 0.995])
        arr = np.asarray(vecs, dtype=np.float32)
        # Normalize
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


@pytest.fixture(autouse=True)
def _stub_sentence_transformers(monkeypatch):
    """
    Install a fake `sentence_transformers` module so relevance.run thinks the
    optional dep is available and does not attempt a real download.
    """
    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = lambda *_a, **_kw: _FakeEncoder()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    # Reset the module-level cache so each test gets a fresh model
    monkeypatch.setattr(relevance, "_cached_model", None, raising=False)
    monkeypatch.setattr(relevance, "_cached_model_id", None, raising=False)
    yield


def _config(**kwargs) -> CleaningConfig:
    defaults = dict(
        run_id="r",
        topic="DevOps",
        enable_perplexity=False,
        enable_relevance=True,
        enable_trafilatura=False,
        relevance_threshold=0.5,
    )
    defaults.update(kwargs)
    return CleaningConfig(**defaults)


def test_relevance_keeps_on_topic_docs():
    docs = [
        _make_doc("d1", "DevOps is the practice of combining dev and ops teams for faster delivery. " * 10),
        _make_doc("d2", "DevOps engineers build and maintain CI/CD pipelines with automation tools. " * 10),
    ]
    kept = relevance.run(docs, _config())
    assert len(kept) == 2


def test_relevance_drops_off_topic_docs():
    docs = [
        _make_doc(
            "d1",
            "Baking sourdough bread at home is an art that blends microbiology and patience. " * 10,
        )
    ]
    kept = relevance.run(docs, _config(relevance_threshold=0.5))
    assert kept == []


def test_relevance_mixed_corpus():
    docs = [
        _make_doc("on1", "DevOps teams shift-left on security and automate the delivery pipeline. " * 10),
        _make_doc("on2", "DevOps culture depends on blameless postmortems and shared ownership. " * 10),
        _make_doc("off1", "The history of medieval bread recipes spans multiple continents. " * 10),
    ]
    kept = relevance.run(docs, _config(relevance_threshold=0.5))
    kept_ids = {d.id for d in kept}
    assert kept_ids == {"on1", "on2"}


def test_relevance_empty_input():
    assert relevance.run([], _config()) == []


def test_relevance_skips_gracefully_when_lib_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    # With the module set to None, `from sentence_transformers import ...` will raise ImportError
    # and the stage should short-circuit and return docs unchanged.
    docs = [_make_doc("d1", "Completely off-topic bread history. " * 10)]
    # Re-import so that the guarded import inside run() takes the ImportError branch
    import importlib
    import cleaning_system.stages.relevance as rel_mod

    importlib.reload(rel_mod)
    kept = rel_mod.run(docs, _config())
    assert kept == docs
