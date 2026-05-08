"""Add explicit study source context."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260508_0003"
down_revision = "20260508_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("read_status", sa.String(length=64), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_study_sources_workspace_id"), "study_sources", ["workspace_id"], unique=False)
    op.create_index(op.f("ix_study_sources_source_type"), "study_sources", ["source_type"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_study_sources_source_type"), table_name="study_sources")
    op.drop_index(op.f("ix_study_sources_workspace_id"), table_name="study_sources")
    op.drop_table("study_sources")
