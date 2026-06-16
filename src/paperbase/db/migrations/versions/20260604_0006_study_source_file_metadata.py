"""Track local file metadata for study sources."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260604_0006"
down_revision = "20260529_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("study_sources", sa.Column("source_size_bytes", sa.Integer(), nullable=True))
    op.add_column("study_sources", sa.Column("source_mtime_ns", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("study_sources", "source_mtime_ns")
    op.drop_column("study_sources", "source_size_bytes")
