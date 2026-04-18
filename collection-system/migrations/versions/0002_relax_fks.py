"""Make urls.query_id and docs.url_id nullable to tolerate concurrent saves."""
from __future__ import annotations

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("urls", "query_id", nullable=True)
    op.alter_column("docs", "url_id", nullable=True)


def downgrade() -> None:
    op.alter_column("urls", "query_id", nullable=False)
    op.alter_column("docs", "url_id", nullable=False)
