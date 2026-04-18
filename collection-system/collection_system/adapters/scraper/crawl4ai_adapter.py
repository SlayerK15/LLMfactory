"""Crawl4AI scraper adapter — primary scraping engine."""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import structlog

from collection_system.core.constants import Stage, URLStatus
from collection_system.core.errors import ScraperError
from collection_system.core.models import DiscoveredURL, Failure, ScrapedDoc

log = structlog.get_logger()


def _approx_token_count(text: str) -> int:
    """Cheap token estimate: words × 1.3 (close enough without tiktoken)."""
    return int(len(text.split()) * 1.3)


class Crawl4AIAdapter:
    """
    Primary scraper using Crawl4AI with Playwright.
    Handles JS-rendered pages, extracts clean markdown, returns token counts.
    Must be used as an async context manager so the browser lifecycle is managed.
    """

    def __init__(
        self,
        concurrency: int = 40,
        per_url_timeout_s: int = 30,
        headless: bool = True,
        browser_type: str = "chromium",
    ) -> None:
        self.concurrency = concurrency
        self.per_url_timeout_s = per_url_timeout_s
        self.headless = headless
        self.browser_type = browser_type
        self._crawler: Any = None
        self._browser_cfg: Any = None
        self._run_cfg: Any = None

    @property
    def name(self) -> str:
        return "crawl4ai"

    async def __aenter__(self) -> "Crawl4AIAdapter":
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        self._browser_cfg = BrowserConfig(
            headless=self.headless,
            browser_type=self.browser_type,
            verbose=False,
        )
        self._run_cfg = CrawlerRunConfig(
            page_timeout=self.per_url_timeout_s * 1000,
            wait_until="domcontentloaded",
            word_count_threshold=50,
            remove_overlay_elements=True,
        )
        self._crawler = AsyncWebCrawler(config=self._browser_cfg)
        await self._crawler.__aenter__()
        log.info(
            "crawl4ai.started",
            concurrency=self.concurrency,
            timeout_s=self.per_url_timeout_s,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._crawler is not None:
            try:
                await self._crawler.__aexit__(*args)
            except Exception as exc:  # noqa: BLE001
                log.warning("crawl4ai.shutdown_failed", error=str(exc))
            self._crawler = None
        log.info("crawl4ai.stopped")

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        if self._crawler is None:
            raise ScraperError("Crawl4AIAdapter used outside async context")

        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._crawler.arun(url=url.url, config=self._run_cfg),
                timeout=self.per_url_timeout_s + 5,
            )
        except asyncio.TimeoutError:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="TimeoutError",
                error_msg=f"Exceeded {self.per_url_timeout_s}s",
            )
        except Exception as exc:  # noqa: BLE001 — Crawl4AI raises many types
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type=type(exc).__name__,
                error_msg=str(exc)[:500],
            )

        duration_ms = int((time.monotonic() - t0) * 1000)

        if not getattr(result, "success", False):
            err_msg = getattr(result, "error_message", "unknown") or "unknown"
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="ScrapeFailed",
                error_msg=str(err_msg)[:500],
            )

        # Crawl4AI returns markdown as either str or an object with .raw_markdown
        md_attr = getattr(result, "markdown", None)
        if md_attr is None:
            markdown = ""
        elif isinstance(md_attr, str):
            markdown = md_attr
        else:
            markdown = getattr(md_attr, "raw_markdown", None) or getattr(
                md_attr, "fit_markdown", ""
            )

        if not markdown or len(markdown.strip()) < 100:
            return Failure(
                run_id=url.run_id,
                stage=Stage.SCRAPE,
                target=url.url,
                error_type="EmptyContent",
                error_msg=f"Only {len(markdown)} chars extracted",
            )

        title = getattr(result, "metadata", {}).get("title") if getattr(
            result, "metadata", None
        ) else None

        return ScrapedDoc(
            run_id=url.run_id,
            url_id=url.id,
            url=url.url,
            title=title,
            markdown=markdown,
            content_hash=ScrapedDoc.compute_content_hash(markdown),
            token_count=_approx_token_count(markdown),
            extraction_confidence=1.0,
            scrape_duration_ms=duration_ms,
        )

    async def scrape_batch(
        self,
        urls: list[DiscoveredURL],
        concurrency: int,
    ) -> AsyncIterator[ScrapedDoc | Failure]:
        """Yield results as they complete — unordered, bounded by `concurrency`."""
        if self._crawler is None:
            raise ScraperError("Crawl4AIAdapter used outside async context")

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
                await queue.put(None)  # sentinel

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
        """Spin up a quick throwaway probe if not already running."""
        if self._crawler is not None:
            return True
        try:
            async with self.__class__(
                concurrency=1, per_url_timeout_s=10, headless=self.headless
            ) as probe:
                result = await probe.scrape(
                    DiscoveredURL(
                        run_id="health-check",
                        query_id="health-check",
                        url="https://example.com",
                        domain="example.com",
                        source_backend=__import__(
                            "collection_system.core.constants",
                            fromlist=["SearchBackend"],
                        ).SearchBackend.CC_CDX,
                        status=URLStatus.PENDING,
                    )
                )
                return isinstance(result, ScrapedDoc)
        except Exception as exc:  # noqa: BLE001
            log.warning("crawl4ai.health_check_failed", error=str(exc))
            return False
