"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-18
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.create_table(
        "runs",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("topic", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="RUNNING"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error_msg", sa.Text, nullable=True),
    )

    op.create_table(
        "queries",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", UUID, sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("parent_id", UUID, sa.ForeignKey("queries.id"), nullable=True),
        sa.Column("depth", sa.SmallInteger, nullable=False),
        sa.Column("relevance_score", sa.Float, nullable=True),
        sa.Column("source", sa.Text, nullable=False, server_default="expansion"),
    )
    op.create_index("idx_queries_run", "queries", ["run_id"])

    op.create_table(
        "urls",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", UUID, sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("query_id", UUID, sa.ForeignKey("queries.id"), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("url_hash", sa.Text, nullable=False),
        sa.Column("domain", sa.Text, nullable=False),
        sa.Column("source_backend", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="PENDING"),
        sa.Column("discovered_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_urls_run", "urls", ["run_id"])
    op.create_unique_constraint("uq_urls_dedup", "urls", ["run_id", "url_hash"])

    op.create_table(
        "docs",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", UUID, sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("url_id", UUID, sa.ForeignKey("urls.id"), nullable=False),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("extraction_confidence", sa.Float, nullable=True),
        sa.Column("scraped_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("scrape_duration_ms", sa.Integer, nullable=True),
    )
    op.create_index("idx_docs_run", "docs", ["run_id"])
    op.create_unique_constraint("uq_docs_content_dedup", "docs", ["run_id", "content_hash"])

    op.create_table(
        "failures",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", UUID, sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("target", sa.Text, nullable=False),
        sa.Column("error_type", sa.Text, nullable=False),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.Column("retries", sa.SmallInteger, nullable=False, server_default="0"),
        sa.Column("failed_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_failures_run", "failures", ["run_id"])

    op.create_table(
        "stage_stats",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", UUID, sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("input_count", sa.Integer, server_default="0"),
        sa.Column("output_count", sa.Integer, server_default="0"),
        sa.Column("failure_count", sa.Integer, server_default="0"),
    )
    op.create_index("idx_stage_stats_run", "stage_stats", ["run_id"])


def downgrade() -> None:
    op.drop_table("stage_stats")
    op.drop_table("failures")
    op.drop_table("docs")
    op.drop_table("urls")
    op.drop_table("queries")
    op.drop_table("runs")
