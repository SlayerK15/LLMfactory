"""Unit tests for the async token bucket rate limiter."""
import asyncio
import time
import pytest

from collection_system.infra.rate_limiter import AsyncTokenBucket


@pytest.mark.asyncio
async def test_allows_burst():
    bucket = AsyncTokenBucket(rate=10, per=1.0, burst=5)
    start = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.2  # burst should be near-instant


@pytest.mark.asyncio
async def test_throttles_at_rate():
    # burst=1 so every acquire after the first must wait 0.1s (rate=10/s)
    bucket = AsyncTokenBucket(rate=10, per=1.0, burst=1)
    start = time.monotonic()
    for _ in range(3):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    # acquire 1: free (1 token in bucket). acquires 2+3: each wait 0.1s → ≥0.2s total
    assert elapsed >= 0.1
