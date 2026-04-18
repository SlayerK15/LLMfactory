"""Unit tests for language detection filter."""
from __future__ import annotations

import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.stages import lang_filter
from tests.conftest import _make_doc


@pytest.fixture
def config() -> CleaningConfig:
    return CleaningConfig(
        run_id="r", topic="t",
        enable_perplexity=False, enable_relevance=False, enable_trafilatura=False,
        target_language="en",
    )


def test_keeps_english_docs(config: CleaningConfig, sample_docs):
    kept = lang_filter.run(sample_docs, config)
    assert len(kept) == len(sample_docs)


def test_drops_non_english(config: CleaningConfig):
    french = _make_doc(
        "d_fr",
        "Le déploiement continu est une pratique essentielle dans le développement logiciel "
        "moderne. Les équipes utilisent des pipelines CI/CD pour automatiser les tests et "
        "les déploiements. Cette approche réduit les erreurs humaines et accélère les livraisons. "
        "L'intégration continue permet de détecter les bugs plus tôt dans le cycle de développement.",
    )
    kept = lang_filter.run([french], config)
    assert kept == []


def test_drops_too_short(config: CleaningConfig):
    short = _make_doc("d_short", "CI CD")
    kept = lang_filter.run([short], config)
    assert kept == []


def test_empty_input(config: CleaningConfig):
    assert lang_filter.run([], config) == []


def test_mixed_batch(config: CleaningConfig):
    english = _make_doc(
        "en1",
        "Kubernetes is a container orchestration system for automating deployment and "
        "scaling of applications. It manages clusters of machines running containers. "
        "Pods, services, and deployments are core Kubernetes primitives used daily.",
    )
    german = _make_doc(
        "de1",
        "Kubernetes ist ein Container-Orchestrierungssystem zur Automatisierung von "
        "Bereitstellung und Skalierung von Anwendungen. Es verwaltet Cluster von Maschinen. "
        "Pods, Dienste und Deployments sind grundlegende Kubernetes-Primitiven.",
    )
    kept = lang_filter.run([english, german], config)
    assert len(kept) == 1
    assert kept[0].id == "en1"
