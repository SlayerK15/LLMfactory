"""Tenacity retry decorators for network calls."""
from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from collection_system.core.errors import LLMRateLimitError, ScraperError, SearchError

# Standard retry for search backends
retry_search = retry(
    retry=retry_if_exception_type((SearchError, ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    reraise=True,
)

# Retry for LLM calls with longer backoff for rate limits
retry_llm = retry(
    retry=retry_if_exception_type((LLMRateLimitError, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=30),
    reraise=True,
)

# Retry for scraper — shorter timeout, fewer attempts
retry_scrape = retry(
    retry=retry_if_exception_type((ScraperError, ConnectionError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential_jitter(initial=0.5, max=5),
    reraise=True,
)
