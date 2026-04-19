"""SearXNG self-hosted meta-search adapter."""
from __future__ import annotations

import asyncio
import itertools

import httpx
import structlog

from collection_system.adapters.search.base import extract_domain, normalize_url
from collection_system.adapters.search.blocklist import is_blocked
from collection_system.core.constants import SearchBackend, URLStatus
from collection_system.core.errors import SearchError
from collection_system.core.models import DiscoveredURL, Query
from collection_system.core.ports import RateLimit

log = structlog.get_logger()


class SearXNGAdapter:
    """
    Queries a self-hosted SearXNG pool via its JSON API.

    Accepts either a single base URL or a comma-separated list, e.g.
    ``http://host:8081,http://host:8082``. Requests are round-robined across
    the pool, and if one instance fails (timeout, 5xx, CAPTCHA-shaped error)
    we retry on the next instance up to ``max_instance_retries`` times. This
    smooths over per-instance throttling from upstream search engines when the
    pool has multiple workers.

    Rate-limited per config to avoid triggering upstream CAPTCHA on Google/Bing.
    The configured rate is the aggregate across the pool — so a 10-instance
    pool at ``rate_per_second=20`` gives ~2 rps per instance.
    """

    def __init__(
        self,
        base_url: str,
        rate_per_second: float = 2.0,
        categories: str = "general",
        engines: str | None = None,
        timeout_s: float = 15.0,
        max_instance_retries: int = 3,
        language: str = "en",
        safesearch: str = "1",
    ) -> None:
        urls = [u.strip().rstrip("/") for u in base_url.split(",") if u.strip()]
        if not urls:
            raise ValueError("SearXNGAdapter needs at least one base URL")
        self._base_urls: list[str] = urls
        # Primary base_url kept for compatibility with callers/logs.
        self.base_url = urls[0]
        self._rr = itertools.cycle(urls)
        self._rr_lock = asyncio.Lock()
        self._rate_per_second = rate_per_second
        self._categories = categories
        self._engines = engines  # e.g. "duckduckgo,brave" — None uses server default
        self._timeout_s = timeout_s
        self._max_instance_retries = max(1, min(max_instance_retries, len(urls)))
        self._language = language
        self._safesearch = safesearch
        self._clients: dict[str, httpx.AsyncClient] = {}

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def pool_size(self) -> int:
        return len(self._base_urls)

    @property
    def rate_limit(self) -> RateLimit:
        return RateLimit(requests=max(1, int(self._rate_per_second)), per_seconds=1.0)

    async def _next_url(self) -> str:
        # itertools.cycle isn't thread-safe under heavy async concurrency —
        # take a lock to get a clean round-robin.
        async with self._rr_lock:
            return next(self._rr)

    async def _client_for(self, url: str) -> httpx.AsyncClient:
        client = self._clients.get(url)
        if client is None:
            client = httpx.AsyncClient(
                base_url=url,
                timeout=self._timeout_s,
                headers={"User-Agent": "collection-system/0.1 (self-hosted)"},
            )
            self._clients[url] = client
        return client

    async def _get_client(self) -> httpx.AsyncClient:
        """Kept for backward compatibility / single-URL call sites."""
        return await self._client_for(self.base_url)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def discover_urls(self, query: Query, limit: int = 20) -> list[DiscoveredURL]:
        params: dict[str, str] = {
            "q": query.text,
            "format": "json",
            "categories": self._categories,
            "safesearch": self._safesearch,
            "language": self._language,
            "pageno": "1",
        }
        if self._engines:
            params["engines"] = self._engines

        payload: dict | None = None
        last_exc: Exception | None = None
        tried: list[str] = []
        for _ in range(self._max_instance_retries):
            url = await self._next_url()
            if url in tried:
                # Pool exhausted before retries — stop spinning on the same host.
                continue
            tried.append(url)
            client = await self._client_for(url)
            try:
                resp = await client.get("/search", params=params)
                resp.raise_for_status()
                payload = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                last_exc = SearchError(
                    f"searxng {exc.response.status_code} for {query.text!r} via {url}"
                )
                log.warning(
                    "searxng.instance_http_error",
                    url=url, status=exc.response.status_code, query=query.text,
                )
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = SearchError(f"searxng request failed via {url}: {exc}")
                log.warning(
                    "searxng.instance_failed",
                    url=url, error=str(exc), query=query.text,
                )
        if payload is None:
            assert last_exc is not None
            raise last_exc

        items = payload.get("results") or []
        seen: set[str] = set()
        results: list[DiscoveredURL] = []
        for item in items:
            raw = item.get("url")
            if not raw:
                continue
            normed = normalize_url(raw)
            if normed in seen:
                continue
            domain = extract_domain(normed)
            if is_blocked(domain, normed):
                continue
            seen.add(normed)
            title = (item.get("title") or "").strip() or None
            snippet = (item.get("content") or "").strip() or None
            results.append(
                DiscoveredURL(
                    run_id=query.run_id,
                    query_id=query.id,
                    url=normed,
                    domain=domain,
                    source_backend=SearchBackend.SEARXNG,
                    status=URLStatus.PENDING,
                    title=title,
                    snippet=snippet,
                )
            )
            if len(results) >= limit:
                break

        log.info(
            "searxng.discover_urls",
            query=query.text,
            raw=len(items),
            returned=len(results),
        )
        return results

    async def health_check(self) -> bool:
        """Healthy if *any* pool member responds."""
        for url in self._base_urls:
            client = await self._client_for(url)
            try:
                resp = await client.get("/healthz", timeout=5.0)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                # /healthz may not exist on all versions — fall back to /
                try:
                    resp = await client.get("/", timeout=5.0)
                    if resp.status_code < 500:
                        return True
                except httpx.HTTPError as exc:
                    log.warning(
                        "searxng.health_check_failed", url=url, error=str(exc)
                    )
        return False
