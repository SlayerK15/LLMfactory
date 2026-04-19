from __future__ import annotations

import asyncio

from collection_system.bootstrap import build_adapters
from collection_system.core.models import RunConfig
from collection_system.infra.config import Settings


def test_build_adapters_ensures_schema_when_database_configured(monkeypatch):
    calls: dict[str, int] = {"init_db": 0, "ensure_schema": 0}

    def _fake_init_db(_url: str) -> None:
        calls["init_db"] += 1

    async def _fake_ensure_schema() -> None:
        calls["ensure_schema"] += 1

    monkeypatch.setattr("collection_system.bootstrap.init_db", _fake_init_db)
    monkeypatch.setattr("collection_system.bootstrap.ensure_schema", _fake_ensure_schema)
    monkeypatch.setattr("collection_system.bootstrap.build_llm", lambda _s: object())
    monkeypatch.setattr("collection_system.bootstrap.build_search", lambda _s, _b: object())
    monkeypatch.setattr("collection_system.bootstrap.build_scraper", lambda _s: object())
    monkeypatch.setattr("collection_system.bootstrap.build_storage", lambda: object())
    monkeypatch.setattr("collection_system.bootstrap.build_filesystem", lambda _s: object())

    settings = Settings(
        llm_provider="ollama",
        DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db",
    )
    cfg = RunConfig(topic="AI")

    asyncio.run(build_adapters(cfg, settings=settings))

    assert calls["init_db"] == 1
    assert calls["ensure_schema"] == 1
