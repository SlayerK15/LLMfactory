"""
Composition root — builds a fully-wired AdapterBundle from Settings.
Lives outside `core/` so the rule "core never imports adapters" holds.
Entry points (api.py, cli.py) call build_adapters() to get a runnable bundle.
"""
from __future__ import annotations

from collection_system.adapters.llm.groq_adapter import GroqAdapter
from collection_system.adapters.llm.ollama_adapter import OllamaAdapter
from collection_system.adapters.scraper.crawl4ai_adapter import Crawl4AIAdapter
from collection_system.adapters.search.cc_cdx import CCDXAdapter
from collection_system.adapters.search.composite import CompositeSearchAdapter
from collection_system.adapters.search.ddg_lite import DDGLiteAdapter
from collection_system.adapters.search.searxng import SearXNGAdapter
from collection_system.adapters.storage.filesystem import FilesystemAdapter
from collection_system.adapters.storage.postgres import PostgresStorageAdapter
from collection_system.core.constants import SearchBackend
from collection_system.core.models import AdapterBundle, RunConfig
from collection_system.infra.config import Settings, get_settings
from collection_system.infra.db import init_db


def build_llm(settings: Settings):
    """Select the configured LLM provider."""
    if settings.llm_provider == "groq" and settings.groq_api_key:
        return GroqAdapter(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            max_depth=settings.max_depth,
        )
    return OllamaAdapter(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        max_depth=settings.max_depth,
    )


def build_search(settings: Settings, enabled: list[SearchBackend]):
    """Build a CompositeSearchAdapter containing only enabled backends."""
    backends = []
    for backend in enabled:
        if backend == SearchBackend.CC_CDX:
            backends.append(CCDXAdapter(index=settings.cc_cdx_index))
        elif backend == SearchBackend.SEARXNG:
            backends.append(
                SearXNGAdapter(
                    base_url=settings.searxng_base_url,
                    rate_per_second=settings.searxng_rate_per_second,
                )
            )
        elif backend == SearchBackend.DDG_LITE:
            backends.append(DDGLiteAdapter())
    return CompositeSearchAdapter(backends=backends)


def build_scraper(settings: Settings) -> Crawl4AIAdapter:
    return Crawl4AIAdapter(
        concurrency=settings.scraper_concurrency,
        per_url_timeout_s=settings.per_url_timeout_s,
    )


def build_storage() -> PostgresStorageAdapter:
    return PostgresStorageAdapter()


def build_filesystem(settings: Settings) -> FilesystemAdapter:
    return FilesystemAdapter(data_dir=settings.data_dir)


async def build_adapters(
    config: RunConfig,
    settings: Settings | None = None,
) -> AdapterBundle:
    """
    Wire up all adapters for a run. The scraper is returned without being
    entered as a context manager — the caller is responsible for `async with`.
    The DB engine is initialised lazily on first call.
    """
    s = settings or get_settings()
    if s.database_url:
        # Idempotent — safe to call multiple times; subsequent calls no-op.
        try:
            init_db(s.database_url)
        except Exception:  # noqa: BLE001 — already initialised
            pass

    return AdapterBundle(
        llm=build_llm(s),
        search=build_search(s, config.search_backends),
        scraper=build_scraper(s),
        storage=build_storage(),
        filesystem=build_filesystem(s),
    )
