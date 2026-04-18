"""Unit tests for the shared search helpers (normalize_url, url_hash, extract_domain)."""
from __future__ import annotations

from collection_system.adapters.search.base import (
    extract_domain,
    normalize_url,
    url_hash,
)


class TestNormalizeUrl:
    def test_lowercases_scheme_and_host(self):
        assert (
            normalize_url("HTTPS://Example.COM/Path")
            == "https://example.com/Path"
        )

    def test_strips_fragment(self):
        assert (
            normalize_url("https://example.com/page#section-2")
            == "https://example.com/page"
        )

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/foo/") == "https://example.com/foo"

    def test_preserves_query_string(self):
        assert (
            normalize_url("https://example.com/q?a=1&b=2")
            == "https://example.com/q?a=1&b=2"
        )


class TestUrlHash:
    def test_deterministic(self):
        assert url_hash("https://example.com/") == url_hash("https://example.com/")

    def test_normalises_before_hashing(self):
        # These three URLs are equivalent after normalization
        assert (
            url_hash("https://Example.com/page")
            == url_hash("HTTPS://example.com/page")
            == url_hash("https://example.com/page#fragment")
        )

    def test_is_sha256_hex(self):
        h = url_hash("https://example.com/")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestExtractDomain:
    def test_basic(self):
        assert extract_domain("https://example.com/page") == "example.com"

    def test_subdomain(self):
        assert extract_domain("https://blog.example.com/x") == "blog.example.com"

    def test_lowercases(self):
        assert extract_domain("https://EXAMPLE.COM/x") == "example.com"
