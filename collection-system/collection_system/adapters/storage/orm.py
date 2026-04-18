"""
SQLAlchemy ORM table classes.
These are the persistence-layer mirror of the Pydantic domain models in
`core.models`. They stay in the adapters/ layer — core never imports them.
The schema is kept in lockstep with `migrations/versions/0001_initial_schema.py`.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from collection_system.infra.db import Base


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)


class QueryRow(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("queries.id"), nullable=True
    )
    depth: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="expansion")


class URLRow(Base):
    __tablename__ = "urls"
    __table_args__ = (
        UniqueConstraint("run_id", "url_hash", name="uq_urls_dedup"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("queries.id"), nullable=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(Text, nullable=False)
    source_backend: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="PENDING")
    discovered_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )


class DocRow(Base):
    __tablename__ = "docs"
    __table_args__ = (
        UniqueConstraint("run_id", "content_hash", name="uq_docs_content_dedup"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    url_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("urls.id"), nullable=True
    )
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )
    scrape_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class FailureRow(Base):
    __tablename__ = "failures"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    target: Mapped[str] = mapped_column(Text, nullable=False)
    error_type: Mapped[str] = mapped_column(Text, nullable=False)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    retries: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    failed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )


class StageStatsRow(Base):
    __tablename__ = "stage_stats"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    input_count: Mapped[int] = mapped_column(Integer, default=0)
    output_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
