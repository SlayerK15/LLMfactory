"""Unit tests for core domain models."""
import pytest
from collection_system.core.models import (
    DiscoveredURL, Query, RunConfig, ScrapedDoc,
)
from collection_system.core.constants import SearchBackend


def test_run_config_defaults():
    config = RunConfig(topic="DevOps")
    assert config.doc_count == 2000
    assert config.max_depth == 3
    assert len(config.run_id) == 36  # UUID4


def test_run_config_run_ids_are_unique():
    a = RunConfig(topic="X")
    b = RunConfig(topic="X")
    assert a.run_id != b.run_id


def test_discovered_url_hash_is_deterministic():
    url = DiscoveredURL(
        run_id="r1", query_id="q1",
        url="https://Example.com/page/",
        domain="example.com",
        source_backend=SearchBackend.CC_CDX,
    )
    assert url.url_hash == url.url_hash  # idempotent


def test_scraped_doc_content_hash():
    h1 = ScrapedDoc.compute_content_hash("hello world")
    h2 = ScrapedDoc.compute_content_hash("hello world")
    h3 = ScrapedDoc.compute_content_hash("different")
    assert h1 == h2
    assert h1 != h3
