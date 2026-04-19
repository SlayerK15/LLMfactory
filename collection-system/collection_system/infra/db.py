"""SQLAlchemy async engine + session factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    pass


def init_db(database_url: str) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
    )
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )


async def ensure_schema() -> None:
    """
    Create persistence tables if they do not exist yet.
    Safe to call repeatedly; used as a lightweight bootstrap guardrail when
    migrations were not applied in a fresh environment.
    """
    engine = get_engine()
    # Import ORM models so SQLAlchemy metadata is fully registered.
    from collection_system.adapters.storage import orm  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    if _engine:
        await _engine.dispose()
