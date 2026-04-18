"""Async token bucket rate limiter — per-backend."""
from __future__ import annotations

import asyncio
import time

import structlog

log = structlog.get_logger()


class AsyncTokenBucket:
    """
    Async token bucket rate limiter.
    Thread-safe via asyncio lock — safe for use with asyncio.gather().
    """

    def __init__(self, rate: float, per: float = 1.0, burst: int | None = None) -> None:
        self._rate = rate
        self._per = per
        self._capacity = float(burst or max(1, int(rate * per)))
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, backend: str = "") -> None:
        """Block until a token is available."""
        async with self._lock:
            self._refill()
            while self._tokens < 1.0:
                wait = (1.0 - self._tokens) / (self._rate / self._per)
                if backend:
                    log.debug("rate_limit_wait", backend=backend, wait_s=round(wait, 3))
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * (self._rate / self._per))
        self._last_refill = now


class RateLimiterRegistry:
    """Registry of per-backend rate limiters."""

    def __init__(self) -> None:
        self._limiters: dict[str, AsyncTokenBucket] = {}

    def register(self, backend: str, rate: float, per: float = 1.0) -> None:
        self._limiters[backend] = AsyncTokenBucket(rate=rate, per=per)

    async def acquire(self, backend: str) -> None:
        limiter = self._limiters.get(backend)
        if limiter:
            await limiter.acquire(backend=backend)

    def get(self, backend: str) -> AsyncTokenBucket | None:
        return self._limiters.get(backend)
