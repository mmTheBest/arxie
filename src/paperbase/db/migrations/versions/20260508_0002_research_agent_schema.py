"""Add research agent persistence tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260508_0002"
down_revision = "20260415_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_threads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=True),
        sa.Column("selected_paper_ids_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_threads_collection_id"), "research_threads", ["collection_id"], unique=False)

    op.create_table(
        "research_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("input_payload_json", sa.JSON(), nullable=False),
        sa.Column("output_payload_json", sa.JSON(), nullable=False),
        sa.Column("evidence_payload_json", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["thread_id"], ["research_threads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_artifacts_collection_id"), "research_artifacts", ["collection_id"], unique=False)
    op.create_index(op.f("ix_research_artifacts_thread_id"), "research_artifacts", ["thread_id"], unique=False)
    op.create_index(op.f("ix_research_artifacts_artifact_type"), "research_artifacts", ["artifact_type"], unique=False)

    op.create_table(
        "research_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["research_artifacts.id"]),
        sa.ForeignKeyConstraint(["thread_id"], ["research_threads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_messages_thread_id"), "research_messages", ["thread_id"], unique=False)

    op.create_table(
        "paper_research_labels",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("paper_id", sa.String(length=36), nullable=False),
        sa.Column("user_label", sa.String(length=64), nullable=False),
        sa.Column("inferred_label", sa.String(length=64), nullable=True),
        sa.Column("inferred_signals_json", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "paper_id", name="uq_paper_research_labels_collection_paper"),
    )
    op.create_index(op.f("ix_paper_research_labels_collection_id"), "paper_research_labels", ["collection_id"], unique=False)
    op.create_index(op.f("ix_paper_research_labels_paper_id"), "paper_research_labels", ["paper_id"], unique=False)

    op.create_table(
        "research_design_elements",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("paper_id", sa.String(length=36), nullable=False),
        sa.Column("element_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_design_elements_paper_id"), "research_design_elements", ["paper_id"], unique=False)
    op.create_index(op.f("ix_research_design_elements_element_type"), "research_design_elements", ["element_type"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_research_design_elements_element_type"), table_name="research_design_elements")
    op.drop_index(op.f("ix_research_design_elements_paper_id"), table_name="research_design_elements")
    op.drop_table("research_design_elements")
    op.drop_index(op.f("ix_paper_research_labels_paper_id"), table_name="paper_research_labels")
    op.drop_index(op.f("ix_paper_research_labels_collection_id"), table_name="paper_research_labels")
    op.drop_table("paper_research_labels")
    op.drop_index(op.f("ix_research_messages_thread_id"), table_name="research_messages")
    op.drop_table("research_messages")
    op.drop_index(op.f("ix_research_artifacts_artifact_type"), table_name="research_artifacts")
    op.drop_index(op.f("ix_research_artifacts_thread_id"), table_name="research_artifacts")
    op.drop_index(op.f("ix_research_artifacts_collection_id"), table_name="research_artifacts")
    op.drop_table("research_artifacts")
    op.drop_index(op.f("ix_research_threads_collection_id"), table_name="research_threads")
    op.drop_table("research_threads")
