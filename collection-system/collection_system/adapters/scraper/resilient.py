"""Resilient scraper adapter chaining a primary engine with a fallback engine."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

import structlog

from collection_system.core.models import DiscoveredURL, Failure, ScrapedDoc

log = structlog.get_logger()


class ResilientScraperAdapter:
    """Try the primary scraper first, then fallback scraper when primary fails."""

    def __init__(self, primary: object, fallback: object) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def name(self) -> str:
        primary_name = getattr(self.primary, "name", "primary")
        fallback_name = getattr(self.fallback, "name", "fallback")
        return f"resilient({primary_name}->{fallback_name})"

    async def __aenter__(self) -> ResilientScraperAdapter:
        if hasattr(self.primary, "__aenter__"):
            await self.primary.__aenter__()
        if hasattr(self.fallback, "__aenter__"):
            await self.fallback.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if hasattr(self.fallback, "__aexit__"):
            await self.fallback.__aexit__(exc_type, exc, tb)
        if hasattr(self.primary, "__aexit__"):
            await self.primary.__aexit__(exc_type, exc, tb)
        return False

    async def scrape(self, url: DiscoveredURL) -> ScrapedDoc | Failure:
        primary_result = await self.primary.scrape(url)
        if isinstance(primary_result, ScrapedDoc):
            return primary_result

        fallback_result = await self.fallback.scrape(url)
        if isinstance(fallback_result, ScrapedDoc):
            log.info(
                "scraper.fallback_recovered",
                run_id=url.run_id,
                url=url.url,
                primary_error=primary_result.error_type,
            )
            return fallback_result

        return primary_result

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
                with suppress(asyncio.CancelledError, Exception):
                    await producer

    async def health_check(self) -> bool:
        primary_ok = await self.primary.health_check()
        fallback_ok = await self.fallback.health_check()
        return primary_ok or fallback_ok
