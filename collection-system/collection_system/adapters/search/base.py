"""Shared helpers for all search adapters."""
from __future__ import annotations

import hashlib
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    """Lowercase scheme+host, strip trailing slash and fragments."""
    parsed = urlparse(url.strip())
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
    )
    return urlunparse(normalized).rstrip("/")


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")
