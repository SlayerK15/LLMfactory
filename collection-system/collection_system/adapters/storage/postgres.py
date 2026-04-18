"""PostgreSQL storage adapter — metadata persistence."""
from __future__ import annotations

from typing import AsyncIterator
from uuid import uuid5, UUID

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from collection_system.adapters.storage.orm import (
    DocRow,
    FailureRow,
    QueryRow,
    RunRow,
    StageStatsRow,
    URLRow,
)
from collection_system.core.constants import RunStatus, Stage
from collection_system.core.errors import RunNotFoundError, StorageError
from collection_system.core.models import (
    DiscoveredURL,
    Failure,
    Query,
    RunConfig,
    RunManifest,
    RunSummary,
    ScrapedDoc,
    StageStats,
)
from collection_system.infra.db import get_session

log = structlog.get_logger()


class PostgresStorageAdapter:
    """
    Stores all pipeline metadata (runs, queries, URLs, doc records, failures)
    in PostgreSQL via SQLAlchemy async.
    Raw document content is stored on the filesystem — see FilesystemAdapter.
    """

    def __init__(self, session_factory: object | None = None) -> None:
        # session_factory is optional; if not provided we use the global one from infra.db.
        # Kept as a parameter for future test injection without breaking the public constructor.
        self._session_factory = session_factory

    # ---------- writes ----------

    async def save_run(self, config: RunConfig) -> None:
        try:
            async with get_session() as session:
                session.add(
                    RunRow(
                        id=config.run_id,
                        topic=config.topic,
                        config=config.model_dump(mode="json"),
                        status=RunStatus.RUNNING.value,
                    )
                )
        except Exception as exc:
            raise StorageError(f"save_run failed: {exc}") from exc

    async def save_query(self, query: Query) -> None:
        for parent_id in (query.parent_id, None):
            try:
                stmt = (
                    pg_insert(QueryRow)
                    .values(
                        id=query.id,
                        run_id=query.run_id,
                        text=query.text,
                        parent_id=parent_id,
                        depth=query.depth,
                        relevance_score=query.relevance_score,
                        source=query.source,
                    )
                    .on_conflict_do_nothing(index_elements=["id"])
                )
                async with get_session() as session:
                    await session.execute(stmt)
                return
            except Exception as exc:
                if parent_id is not None and "ForeignKeyViolation" in str(exc):
                    continue  # parent not committed yet — retry without parent_id
                raise StorageError(f"save_query failed: {exc}") from exc

    async def save_url(self, url: DiscoveredURL) -> None:
        """
        Idempotent insert — duplicate (run_id, url_hash) is silently ignored.
        This makes url_seen() + save_url() safe under races without a transaction.
        """
        for query_id in (url.query_id, None):
            try:
                stmt = (
                    pg_insert(URLRow)
                    .values(
                        id=url.id,
                        run_id=url.run_id,
                        query_id=query_id,
                        url=url.url,
                        url_hash=url.url_hash,
                        domain=url.domain,
                        source_backend=url.source_backend.value,
                        status=url.status.value,
                    )
                    .on_conflict_do_nothing(index_elements=["run_id", "url_hash"])
                )
                async with get_session() as session:
                    await session.execute(stmt)
                return
            except Exception as exc:
                if query_id is not None and "ForeignKeyViolation" in str(exc):
                    continue
                raise StorageError(f"save_url failed: {exc}") from exc

    async def save_doc(self, doc: ScrapedDoc) -> None:
        """Idempotent insert by (run_id, content_hash)."""
        for url_id in (doc.url_id, None):
            try:
                stmt = (
                    pg_insert(DocRow)
                    .values(
                        id=doc.id,
                        run_id=doc.run_id,
                        url_id=url_id,
                        content_hash=doc.content_hash,
                        path=str(doc.path),
                        token_count=doc.token_count,
                        extraction_confidence=doc.extraction_confidence,
                        scrape_duration_ms=doc.scrape_duration_ms,
                    )
                    .on_conflict_do_nothing(index_elements=["run_id", "content_hash"])
                )
                async with get_session() as session:
                    await session.execute(stmt)
                return
            except Exception as exc:
                if url_id is not None and "ForeignKeyViolation" in str(exc):
                    continue
                raise StorageError(f"save_doc failed: {exc}") from exc

    async def save_failure(self, failure: Failure) -> None:
        try:
            async with get_session() as session:
                session.add(
                    FailureRow(
                        id=failure.id,
                        run_id=failure.run_id,
                        stage=failure.stage.value,
                        target=failure.target,
                        error_type=failure.error_type,
                        error_msg=failure.error_msg,
                        retries=failure.retries,
                    )
                )
        except Exception as exc:
            raise StorageError(f"save_failure failed: {exc}") from exc

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        error_msg: str | None = None,
    ) -> None:
        from datetime import datetime, timezone

        values: dict = {"status": status.value}
        if status in (RunStatus.COMPLETED, RunStatus.FAILED):
            values["completed_at"] = datetime.now(timezone.utc)
        if error_msg is not None:
            values["error_msg"] = error_msg
        try:
            async with get_session() as session:
                await session.execute(
                    update(RunRow).where(RunRow.id == run_id).values(**values)
                )
        except Exception as exc:
            raise StorageError(f"update_run_status failed: {exc}") from exc

    async def save_stage_stats(self, run_id: str, stats: StageStats) -> None:
        try:
            async with get_session() as session:
                session.add(
                    StageStatsRow(
                        id=str(uuid5(UUID(run_id), stats.stage.value)),
                        run_id=run_id,
                        stage=stats.stage.value,
                        started_at=stats.started_at,
                        completed_at=stats.completed_at,
                        input_count=stats.input_count,
                        output_count=stats.output_count,
                        failure_count=stats.failure_count,
                    )
                )
        except Exception as exc:
            raise StorageError(f"save_stage_stats failed: {exc}") from exc

    # ---------- reads ----------

    async def load_run(self, run_id: str) -> RunManifest:
        async with get_session() as session:
            run_row = (
                await session.execute(select(RunRow).where(RunRow.id == run_id))
            ).scalar_one_or_none()
            if run_row is None:
                raise RunNotFoundError(run_id)

            stats_rows = (
                await session.execute(
                    select(StageStatsRow).where(StageStatsRow.run_id == run_id)
                )
            ).scalars().all()

            stages = {
                Stage(row.stage): StageStats(
                    stage=Stage(row.stage),
                    started_at=row.started_at,
                    completed_at=row.completed_at,
                    input_count=row.input_count,
                    output_count=row.output_count,
                    failure_count=row.failure_count,
                )
                for row in stats_rows
            }

            config = RunConfig.model_validate(run_row.config)
            return RunManifest(
                run_id=run_row.id,
                config=config,
                status=RunStatus(run_row.status),
                started_at=run_row.started_at,
                completed_at=run_row.completed_at,
                stages=stages,
                error_msg=run_row.error_msg,
            )

    async def list_runs(self, limit: int = 50) -> list[RunSummary]:
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(RunRow).order_by(RunRow.started_at.desc()).limit(limit)
                )
            ).scalars().all()
            summaries: list[RunSummary] = []
            for row in rows:
                doc_count_row = await session.execute(
                    select(DocRow).where(DocRow.run_id == row.id)
                )
                docs = doc_count_row.scalars().all()
                duration: float | None = None
                if row.completed_at:
                    duration = (row.completed_at - row.started_at).total_seconds()
                summaries.append(
                    RunSummary(
                        run_id=row.id,
                        topic=row.topic,
                        status=RunStatus(row.status),
                        docs_collected=len(docs),
                        started_at=row.started_at,
                        duration_s=duration,
                    )
                )
            return summaries

    async def load_docs(self, run_id: str) -> AsyncIterator[ScrapedDoc]:
        from pathlib import Path

        async with get_session() as session:
            result = await session.stream(
                select(DocRow, URLRow)
                .join(URLRow, DocRow.url_id == URLRow.id)
                .where(DocRow.run_id == run_id)
            )
            async for row in result:
                doc_row, url_row = row
                yield ScrapedDoc(
                    id=doc_row.id,
                    run_id=doc_row.run_id,
                    url_id=doc_row.url_id,
                    url=url_row.url,
                    markdown="",  # content is on disk; readers call FilesystemAdapter
                    content_hash=doc_row.content_hash,
                    token_count=doc_row.token_count or 0,
                    extraction_confidence=doc_row.extraction_confidence or 1.0,
                    scraped_at=doc_row.scraped_at,
                    scrape_duration_ms=doc_row.scrape_duration_ms or 0,
                    path=Path(doc_row.path),
                )

    async def load_failures(self, run_id: str) -> list[Failure]:
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(FailureRow).where(FailureRow.run_id == run_id)
                )
            ).scalars().all()
            return [
                Failure(
                    id=r.id,
                    run_id=r.run_id,
                    stage=Stage(r.stage),
                    target=r.target,
                    error_type=r.error_type,
                    error_msg=r.error_msg or "",
                    retries=r.retries,
                    failed_at=r.failed_at,
                )
                for r in rows
            ]

    async def run_exists(self, run_id: str) -> bool:
        async with get_session() as session:
            row = (
                await session.execute(select(RunRow.id).where(RunRow.id == run_id))
            ).scalar_one_or_none()
            return row is not None

    # ---------- dedup helpers ----------

    async def url_seen(self, run_id: str, url_hash: str) -> bool:
        async with get_session() as session:
            row = (
                await session.execute(
                    select(URLRow.id).where(
                        URLRow.run_id == run_id, URLRow.url_hash == url_hash
                    )
                )
            ).scalar_one_or_none()
            return row is not None

    async def content_hash_seen(self, run_id: str, content_hash: str) -> bool:
        async with get_session() as session:
            row = (
                await session.execute(
                    select(DocRow.id).where(
                        DocRow.run_id == run_id, DocRow.content_hash == content_hash
                    )
                )
            ).scalar_one_or_none()
            return row is not None
