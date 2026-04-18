"""Plain httpx fallback scraper for simple static pages."""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

import httpx
import structlog

from collection_system.core.constants import Stage
from collection_system.core.models import DiscoveredURL, Failure, ScrapedDoc

log = structlog.get_logger()

# Very rough HTML → text for static pages. Crawl4AI is the primary path;
# this exists so the pipeline keeps running if Playwright is broken.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _html_to_text(html: str) -> str:
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    return _WS_RE.sub(" ", text).strip()


def _extract_title(html: str) -> str | None:
    m = _TITLE_RE.search(html)
    if m:
        return _WS_RE.sub(" ", m.group(1)).strip()
    return None


def _approx_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


class HttpxFallbackAdapter:
    """
    Lightweight fallback scraper using httpx for simple static HTML pages.
    No JS rendering. Uses basic HTML-to-text extraction.
    """

    def __init__(self, timeout_s: int = 15) -> None:
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "httpx_fallback"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; collection-system/0.1; "
                        "+https://github.com/local/collection-system)"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        client = await self._get_client()
        t0 = time.monotonic()
        try:
            resp = await client.get(url.url)
        except httpx.TimeoutException:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="TimeoutError",
                error_msg=f"Exceeded {self.timeout_s}s",
            )
        except httpx.HTTPError as exc:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:500],
            )

        duration_ms = int((time.monotonic() - t0) * 1000)

        if resp.status_code >= 400:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="HTTPError",
                error_msg=f"status {resp.status_code}",
            )

        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype.lower() and "text" not in ctype.lower():
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="NonHTMLContent",
                error_msg=f"content-type {ctype}",
            )

        html = resp.text
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
            extraction_confidence=0.6,  # lower than Crawl4AI — no JS, crude extraction
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
            client = await self._get_client()
            resp = await client.get("https://example.com", timeout=5.0)
            return resp.status_code < 500
        except httpx.HTTPError:
            return False
