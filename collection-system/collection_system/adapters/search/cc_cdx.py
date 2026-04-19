"""
Common Crawl CDX index search adapter.

CC's CDX index is a URL-prefix index, not a full-text one — so topic-based
discovery works by querying a curated list of high-signal content domains
and filtering results where the query slug appears in the URL path.
Pairs with SearXNG for live, text-based discovery; CC is ban-proof historical.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

import structlog

from collection_system.adapters.search.base import extract_domain, normalize_url
from collection_system.core.constants import SearchBackend, URLStatus
from collection_system.core.errors import SearchError
from collection_system.core.models import DiscoveredURL, Query
from collection_system.core.ports import RateLimit

log = structlog.get_logger()

CC_INDEX = "CC-MAIN-2026-12"

# Curated high-signal content domains. These produce dense, topical content
# that's useful for LLM training. Extend via constructor arg.
DEFAULT_SEED_DOMAINS: tuple[str, ...] = (
    "en.wikipedia.org",
    "medium.com",
    "dev.to",
    "github.com",
    "stackoverflow.com",
    "news.ycombinator.com",
    "arxiv.org",
    "docs.python.org",
    "real-python.com",
    "www.ibm.com",
    "www.redhat.com",
    "martinfowler.com",
    "towardsdatascience.com",
    "blog.cloudflare.com",
    "blog.google",
    "engineering.fb.com",
    "netflixtechblog.com",
    "aws.amazon.com",
)

# Permanent noise domains — moved to blocklist.py.  Re-exported here for
# backwards-compatibility with callers that imported BLOCKED_DOMAINS from
# this module before the shared blocklist existed.
from collection_system.adapters.search.blocklist import (  # noqa: E402
    BLOCKED_DOMAINS,
    is_blocked,
)


def _slugify(text: str) -> str:
    """Turn 'Python Async Await' → 'python-async-await'."""
    cleaned = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    return re.sub(r"\s+", "-", cleaned).strip("-")


def _keywords(text: str) -> list[str]:
    """Extract lowercase keywords longer than 3 chars."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3]


class CCDXAdapter:
    """
    Queries the Common Crawl CDX API via cdx-toolkit.
    Free, ban-proof, offline index — primary historical source.
    """

    def __init__(
        self,
        index: str = CC_INDEX,
        seed_domains: tuple[str, ...] | list[str] = DEFAULT_SEED_DOMAINS,
        max_per_query: int = 20,
        seeds_per_query: int = 3,
        seed_timeout_s: float = 10.0,
        short_circuit_after_failures: int = 8,
    ) -> None:
        self.index = index
        self.seed_domains = tuple(seed_domains)
        self.max_per_query = max_per_query
        self.seeds_per_query = seeds_per_query
        self._seed_timeout_s = seed_timeout_s
        self._short_circuit_after = short_circuit_after_failures
        self._consecutive_failures = 0
        self._cdx: Any = None  # lazy init — cdx_toolkit import is expensive

    @property
    def name(self) -> str:
        return "cc_cdx"

    @property
    def rate_limit(self) -> RateLimit:
        # CC CDX is generous; keep this sane so we don't hammer the index server.
        return RateLimit(requests=5, per_seconds=1.0)

    def _get_client(self) -> Any:
        if self._cdx is None:
            import cdx_toolkit

            self._cdx = cdx_toolkit.CDXFetcher(source="cc")
        return self._cdx

    def _pick_seeds(self, query_id: str) -> list[str]:
        """Deterministically rotate seeds based on query id."""
        h = int(hashlib.sha256(query_id.encode()).hexdigest(), 16)
        n = min(self.seeds_per_query, len(self.seed_domains))
        idx = h % len(self.seed_domains)
        return [
            self.seed_domains[(idx + i) % len(self.seed_domains)] for i in range(n)
        ]

    def _iter_seed_blocking(
        self,
        seed: str,
        keywords: list[str],
        slug: str,
        limit: int,
    ) -> list[str]:
        """Synchronous CDX iter — run in asyncio.to_thread."""
        client = self._get_client()
        found: list[str] = []
        try:
            for obj in client.iter(f"{seed}/*", limit=limit):
                url = obj.get("url", "") if hasattr(obj, "get") else str(obj)
                if not url:
                    continue
                url_lower = url.lower()
                # Relevance filter: either slug appears or ≥2 keywords appear
                kw_hits = sum(1 for k in keywords if k in url_lower)
                if slug and slug in url_lower:
                    found.append(url)
                elif kw_hits >= 2:
                    found.append(url)
                if len(found) >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 — CDX client raises varied types
            log.warning("cc_cdx.iter_failed", seed=seed, error=str(exc))
        return found

    async def discover_urls(self, query: Query, limit: int = 20) -> list[DiscoveredURL]:
        # Short-circuit: if the index has been failing repeatedly this run,
        # stop spawning threads that will only block the executor pool.
        if self._consecutive_failures >= self._short_circuit_after:
            log.debug(
                "cc_cdx.short_circuited",
                query=query.text,
                failures=self._consecutive_failures,
            )
            return []

        slug = _slugify(query.text)
        keywords = _keywords(query.text)
        seeds = self._pick_seeds(query.id)
        per_seed = max(5, limit * 2)

        async def _run_seed(seed: str) -> list[str]:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self._iter_seed_blocking, seed, keywords, slug, per_seed
                    ),
                    timeout=self._seed_timeout_s,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "cc_cdx.seed_timeout",
                    seed=seed,
                    timeout_s=self._seed_timeout_s,
                )
                return []

        tasks = [_run_seed(seed) for seed in seeds]
        try:
            batches = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001
            raise SearchError(f"cc_cdx discover_urls failed: {exc}") from exc

        seen: set[str] = set()
        results: list[DiscoveredURL] = []
        all_empty = True
        for batch in batches:
            if isinstance(batch, BaseException):
                log.warning("cc_cdx.seed_failed", error=str(batch))
                continue
            if batch:
                all_empty = False
            for raw_url in batch:
                normed = normalize_url(raw_url)
                if normed in seen:
                    continue
                domain = extract_domain(normed)
                if is_blocked(domain, normed):
                    continue
                seen.add(normed)
                results.append(
                    DiscoveredURL(
                        run_id=query.run_id,
                        query_id=query.id,
                        url=normed,
                        domain=domain,
                        source_backend=SearchBackend.CC_CDX,
                        status=URLStatus.PENDING,
                    )
                )
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        if all_empty:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        log.info(
            "cc_cdx.discover_urls",
            query=query.text,
            seeds=seeds,
            returned=len(results),
        )
        return results

    async def health_check(self) -> bool:
        def _ping() -> bool:
            try:
                client = self._get_client()
                # A tiny probe — iter with limit=1 over a known domain
                it = client.iter("en.wikipedia.org/*", limit=1)
                next(iter(it), None)
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("cc_cdx.health_check_failed", error=str(exc))
                return False

        return await asyncio.to_thread(_ping)
