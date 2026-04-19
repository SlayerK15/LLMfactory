"""
curl_cffi fallback scraper — impersonates a real Chrome TLS fingerprint so
Cloudflare / other TLS-based antibot gates that block plain httpx often
let the request through. Last-ditch static-HTML fetch for URLs that both
Crawl4AI and HttpxFallbackAdapter couldn't recover.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import structlog

from collection_system.adapters.scraper.httpx_fallback import (
    _approx_token_count,
    _extract_title,
    _html_to_text,
)
from collection_system.core.constants import Stage
from collection_system.core.models import DiscoveredURL, Failure, ScrapedDoc

log = structlog.get_logger()

DEFAULT_IMPERSONATE = "chrome124"


class CurlCffiFallbackAdapter:
    """
    TLS-fingerprint-impersonating static fetcher. Use only when the primary
    scraper (and the httpx fallback) have both failed — it's slower than
    httpx and the point is to look like a real browser, not to be fast.
    """

    def __init__(
        self,
        timeout_s: int = 30,
        impersonate: str = DEFAULT_IMPERSONATE,
    ) -> None:
        self.timeout_s = timeout_s
        self.impersonate = impersonate
        self._session: object | None = None  # AsyncSession — imported lazily

    @property
    def name(self) -> str:
        return "curl_cffi_fallback"

    async def _get_session(self):
        if self._session is None:
            # Imported lazily so the adapter module can be imported in
            # environments where curl_cffi isn't installed (tests, CI).
            from curl_cffi.requests import AsyncSession

            self._session = AsyncSession(
                timeout=self.timeout_s,
                impersonate=self.impersonate,
            )
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                log.warning("curl_cffi_fallback.close_failed", error=str(exc))
            self._session = None

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        session = await self._get_session()
        t0 = time.monotonic()
        try:
            resp = await session.get(url.url, allow_redirects=True)  # type: ignore[attr-defined]
        except asyncio.TimeoutError:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="TimeoutError",
                error_msg=f"curl_cffi exceeded {self.timeout_s}s",
            )
        except Exception as exc:  # noqa: BLE001 — curl_cffi raises many types
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:500],
            )

        duration_ms = int((time.monotonic() - t0) * 1000)
        status = getattr(resp, "status_code", 0)
        if status >= 400:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="HTTPError",
                error_msg=f"status {status}",
            )

        headers = getattr(resp, "headers", {}) or {}
        ctype = ""
        try:
            ctype = headers.get("content-type", "") if hasattr(headers, "get") else ""
        except Exception:  # noqa: BLE001
            ctype = ""
        if ctype and "html" not in ctype.lower() and "text" not in ctype.lower():
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="NonHTMLContent",
                error_msg=f"content-type {ctype}",
            )

        html = getattr(resp, "text", "") or ""
        title = _extract_title(html)
        markdown = _html_to_text(html)

        if len(markdown.strip()) < 100:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="EmptyContent",
                error_msg=f"Only {len(markdown)} chars extracted",
            )

        return ScrapedDoc(
            run_id=url.run_id,
            url_id=url.id,
            url=url.url,
            title=title,
            markdown=markdown,
            content_hash=ScrapedDoc.compute_content_hash(markdown),
            token_count=_approx_token_count(markdown),
            # Even lower than httpx: we only hit this path after two other
            # scrapers have failed, so these docs carry the most risk.
            extraction_confidence=0.5,
            scrape_duration_ms=duration_ms,
        )

    async def scrape_batch(
        self,
        urls: list[DiscoveredURL],
        concurrency: int,
    ) -> AsyncIterator[ScrapedDoc | Failure]:
        sem = asyncio.Semaphore(max(1, concurrency))
        queue: asyncio.Queue[ScrapedDoc | Failure | None] = asyncio.Queue()

        async def _one(u: DiscoveredURL) -> None:
            async with sem:
                result = await self.scrape(u)
            await queue.put(result)

        async def _producer() -> None:
            try:
                await asyncio.gather(*[_one(u) for u in urls])
            finally:
                await queue.put(None)

        producer = asyncio.create_task(_producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            resp = await session.get("https://example.com", timeout=5.0)  # type: ignore[attr-defined]
            return getattr(resp, "status_code", 0) < 500
        except Exception:  # noqa: BLE001
            return False
