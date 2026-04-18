"""Unit tests for trafilatura_fix stage — no real HTTP or real trafilatura."""
from __future__ import annotations

import sys
import types

import httpx
import pytest

from cleaning_system.core.models import CleaningConfig
from cleaning_system.stages import trafilatura_fix
from tests.conftest import _make_doc


def _config(**kwargs) -> CleaningConfig:
    defaults = dict(
        run_id="r",
        topic="t",
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=True,
        confidence_threshold_for_reextract=0.80,
        trafilatura_timeout_s=2,
    )
    defaults.update(kwargs)
    return CleaningConfig(**defaults)


@pytest.fixture
def fake_trafilatura(monkeypatch):
    """Install a fake `trafilatura` module whose .extract() is controllable."""
    extract_return = {"value": None}  # mutable closure

    fake = types.ModuleType("trafilatura")

    def _extract(html: str, include_comments=False, include_tables=True):
        return extract_return["value"]

    fake.extract = _extract  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trafilatura", fake)
    return extract_return


def _mock_transport(html_by_url: dict[str, str], status_by_url: dict[str, int] | None = None):
    status_by_url = status_by_url or {}

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        status = status_by_url.get(url, 200)
        return httpx.Response(status, content=html_by_url.get(url, "").encode("utf-8"))

    return httpx.MockTransport(_handler)


@pytest.mark.asyncio
async def test_trafilatura_skips_when_all_docs_high_confidence(fake_trafilatura):
    docs = [_make_doc("d1", "body " * 100, extraction_confidence=0.99)]
    result = await trafilatura_fix.run(docs, _config())
    assert result == docs


@pytest.mark.asyncio
async def test_trafilatura_reextracts_low_confidence_docs(monkeypatch, fake_trafilatura):
    """A low-confidence doc should be re-fetched and its text replaced."""
    fake_trafilatura["value"] = " ".join(["improved"] * 80)  # passes min-50-word gate

    low = _make_doc(
        "low",
        "original body " * 30,
        url="https://example.com/low",
        extraction_confidence=0.5,
    )
    high = _make_doc(
        "high", "untouched body " * 30, extraction_confidence=0.95
    )
    docs = [low, high]

    # Replace the AsyncClient constructor so every new client uses a mock transport.
    transport = _mock_transport({"https://example.com/low": "<html>ignored</html>"})
    orig_ctor = httpx.AsyncClient

    def _ctor(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_ctor(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _ctor)

    result = await trafilatura_fix.run(docs, _config())

    # Same length, same order
    assert len(result) == 2
    # Low-confidence doc was replaced with trafilatura's output
    assert result[0].id == "low"
    assert "improved" in result[0].text
    assert result[0].content_hash != low.content_hash  # text changed → hash recomputed
    # High-confidence doc is untouched object-equality
    assert result[1] is high


@pytest.mark.asyncio
async def test_trafilatura_keeps_original_on_http_error(monkeypatch, fake_trafilatura):
    """When the re-fetch returns 500, the original doc is preserved."""
    fake_trafilatura["value"] = "should not be used"

    low = _make_doc(
        "low",
        "original " * 30,
        url="https://example.com/broken",
        extraction_confidence=0.4,
    )
    transport = _mock_transport(
        html_by_url={"https://example.com/broken": "<html>err</html>"},
        status_by_url={"https://example.com/broken": 500},
    )
    orig_ctor = httpx.AsyncClient

    def _ctor(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_ctor(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _ctor)

    result = await trafilatura_fix.run([low], _config())

    assert len(result) == 1
    # Original text preserved because HTTP 500 short-circuited re-extract
    assert result[0].text == low.text


@pytest.mark.asyncio
async def test_trafilatura_keeps_original_on_empty_extraction(monkeypatch, fake_trafilatura):
    """When trafilatura.extract returns < 50 words, the original is kept."""
    fake_trafilatura["value"] = "too short"  # 2 words

    low = _make_doc(
        "low",
        "original " * 100,
        url="https://example.com/low",
        extraction_confidence=0.4,
    )
    transport = _mock_transport({"https://example.com/low": "<html>ok</html>"})
    orig_ctor = httpx.AsyncClient

    def _ctor(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_ctor(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _ctor)

    result = await trafilatura_fix.run([low], _config())
    assert result[0].text == low.text


@pytest.mark.asyncio
async def test_trafilatura_empty_input(fake_trafilatura):
    assert await trafilatura_fix.run([], _config()) == []


@pytest.mark.asyncio
async def test_trafilatura_skips_when_lib_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    import importlib

    import cleaning_system.stages.trafilatura_fix as tf

    importlib.reload(tf)
    docs = [_make_doc("d1", "body " * 20, extraction_confidence=0.1)]
    result = await tf.run(docs, _config())
    assert result == docs
