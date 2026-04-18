"""
DuckDuckGo html/lite fallback adapter.

Disabled by default — DDG aggressively rate-limits HTML scraping.
Sustainable rate is ≤0.3 req/s per IP. Use only as last-resort fallback
when CC CDX and SearXNG both fail or are cold on a topic.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import structlog

from collection_system.adapters.search.base import extract_domain, normalize_url
from collection_system.adapters.search.cc_cdx import BLOCKED_DOMAINS
from collection_system.core.constants import SearchBackend, URLStatus
from collection_system.core.errors import SearchError
from collection_system.core.models import DiscoveredURL, Query
from collection_system.core.ports import RateLimit

log = structlog.get_logger()

DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"

# DDG lite wraps result links as either direct or via /l/?uddg=<encoded-url>
_LINK_RE = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>', re.IGNORECASE)


def _extract_target_url(href: str) -> str | None:
    """Unwrap DDG's /l/?uddg=... redirector if present, else return href."""
    if href.startswith("/l/?") or "uddg=" in href:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [None])[0]
        if target:
            return unquote(target)
    if href.startswith(("http://", "https://")):
        return href
    return None


class DDGLiteAdapter:
    """
    Scrapes DuckDuckGo's lite endpoint for results.
    Max sustainable rate: ~0.3 req/s per IP to avoid blocks.
    """

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ddg_lite"

    @property
    def rate_limit(self) -> RateLimit:
        # ~0.33 req/s per IP — DDG blocks above this
        return RateLimit(requests=1, per_seconds=3.0)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def discover_urls(self, query: Query, limit: int = 10) -> list[DiscoveredURL]:
        client = await self._get_client()
        try:
            resp = await client.post(DDG_LITE_URL, data={"q": query.text})
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 429):
                raise SearchError(
                    f"ddg_lite blocked ({exc.response.status_code}) — IP flagged"
                ) from exc
            raise SearchError(f"ddg_lite HTTP error: {exc}") from exc
        except httpx.HTTPError as exc:
            raise SearchError(f"ddg_lite request failed: {exc}") from exc

        seen: set[str] = set()
        results: list[DiscoveredURL] = []
        for match in _LINK_RE.finditer(html):
            href = match.group(1)
            target = _extract_target_url(href)
            if target is None:
                continue
            normed = normalize_url(target)
            if normed in seen:
                continue
            domain = extract_domain(normed)
            if domain in BLOCKED_DOMAINS or domain.endswith("duckduckgo.com"):
                continue
            seen.add(normed)
            results.append(
                DiscoveredURL(
                    run_id=query.run_id,
                    query_id=query.id,
                    url=normed,
                    domain=domain,
                    source_backend=SearchBackend.DDG_LITE,
                    status=URLStatus.PENDING,
                )
            )
            if len(results) >= limit:
                break

        log.info(
            "ddg_lite.discover_urls",
            query=query.text,
            returned=len(results),
        )
        return results

    async def health_check(self) -> bool:
        client = await self._get_client()
        try:
            resp = await client.get(DDG_LITE_URL, timeout=5.0)
            return resp.status_code < 500
        except httpx.HTTPError:
            return False
