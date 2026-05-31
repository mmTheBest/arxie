"""Add durable research intelligence memory tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260529_0005"
down_revision = "20260520_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_briefs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("brief_json", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_by", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", name="uq_study_briefs_workspace"),
    )

    op.create_table(
        "research_memory_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=True),
        sa.Column("paper_id", sa.String(length=36), nullable=True),
        sa.Column("memory_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("source_refs_json", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("version_key", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id",
            "workspace_id",
            "memory_type",
            "version_key",
            name="uq_research_memory_records_scope_type_version",
        ),
    )
    op.create_index(
        op.f("ix_research_memory_records_collection_id"),
        "research_memory_records",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_memory_records_workspace_id"),
        "research_memory_records",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_memory_records_paper_id"),
        "research_memory_records",
        ["paper_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_memory_records_memory_type"),
        "research_memory_records",
        ["memory_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_memory_records_version_key"),
        "research_memory_records",
        ["version_key"],
        unique=False,
    )
    op.create_index(
        "uq_research_memory_records_null_workspace_type_version",
        "research_memory_records",
        ["collection_id", "memory_type", "version_key"],
        unique=True,
        sqlite_where=sa.text("workspace_id IS NULL"),
        postgresql_where=sa.text("workspace_id IS NULL"),
    )

    op.create_table(
        "research_graph_nodes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=True),
        sa.Column("node_type", sa.String(length=64), nullable=False),
        sa.Column("stable_key", sa.String(length=255), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id",
            "workspace_id",
            "node_type",
            "stable_key",
            name="uq_research_graph_nodes_scope_type_key",
        ),
    )
    op.create_index(
        op.f("ix_research_graph_nodes_collection_id"),
        "research_graph_nodes",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_nodes_workspace_id"),
        "research_graph_nodes",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_nodes_node_type"),
        "research_graph_nodes",
        ["node_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_nodes_stable_key"),
        "research_graph_nodes",
        ["stable_key"],
        unique=False,
    )
    op.create_index(
        "uq_research_graph_nodes_null_workspace_type_key",
        "research_graph_nodes",
        ["collection_id", "node_type", "stable_key"],
        unique=True,
        sqlite_where=sa.text("workspace_id IS NULL"),
        postgresql_where=sa.text("workspace_id IS NULL"),
    )

    op.create_table(
        "research_graph_edges",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=True),
        sa.Column("source_node_id", sa.String(length=36), nullable=False),
        sa.Column("target_node_id", sa.String(length=36), nullable=False),
        sa.Column("edge_type", sa.String(length=64), nullable=False),
        sa.Column("evidence_refs_json", sa.JSON(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["source_node_id"], ["research_graph_nodes.id"]),
        sa.ForeignKeyConstraint(["target_node_id"], ["research_graph_nodes.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id",
            "workspace_id",
            "source_node_id",
            "target_node_id",
            "edge_type",
            name="uq_research_graph_edges_scope_nodes_type",
        ),
    )
    op.create_index(
        op.f("ix_research_graph_edges_collection_id"),
        "research_graph_edges",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_edges_workspace_id"),
        "research_graph_edges",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_edges_source_node_id"),
        "research_graph_edges",
        ["source_node_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_edges_target_node_id"),
        "research_graph_edges",
        ["target_node_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_research_graph_edges_edge_type"),
        "research_graph_edges",
        ["edge_type"],
        unique=False,
    )
    op.create_index(
        "uq_research_graph_edges_null_workspace_nodes_type",
        "research_graph_edges",
        ["collection_id", "source_node_id", "target_node_id", "edge_type"],
        unique=True,
        sqlite_where=sa.text("workspace_id IS NULL"),
        postgresql_where=sa.text("workspace_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_research_graph_edges_null_workspace_nodes_type",
        table_name="research_graph_edges",
    )
    op.drop_index(op.f("ix_research_graph_edges_edge_type"), table_name="research_graph_edges")
    op.drop_index(op.f("ix_research_graph_edges_target_node_id"), table_name="research_graph_edges")
    op.drop_index(op.f("ix_research_graph_edges_source_node_id"), table_name="research_graph_edges")
    op.drop_index(op.f("ix_research_graph_edges_workspace_id"), table_name="research_graph_edges")
    op.drop_index(op.f("ix_research_graph_edges_collection_id"), table_name="research_graph_edges")
    op.drop_table("research_graph_edges")
    op.drop_index(
        "uq_research_graph_nodes_null_workspace_type_key",
        table_name="research_graph_nodes",
    )
    op.drop_index(op.f("ix_research_graph_nodes_stable_key"), table_name="research_graph_nodes")
    op.drop_index(op.f("ix_research_graph_nodes_node_type"), table_name="research_graph_nodes")
    op.drop_index(op.f("ix_research_graph_nodes_workspace_id"), table_name="research_graph_nodes")
    op.drop_index(op.f("ix_research_graph_nodes_collection_id"), table_name="research_graph_nodes")
    op.drop_table("research_graph_nodes")
    op.drop_index(
        "uq_research_memory_records_null_workspace_type_version",
        table_name="research_memory_records",
    )
    op.drop_index(
        op.f("ix_research_memory_records_version_key"),
        table_name="research_memory_records",
    )
    op.drop_index(
        op.f("ix_research_memory_records_memory_type"),
        table_name="research_memory_records",
    )
    op.drop_index(op.f("ix_research_memory_records_paper_id"), table_name="research_memory_records")
    op.drop_index(
        op.f("ix_research_memory_records_workspace_id"),
        table_name="research_memory_records",
    )
    op.drop_index(
        op.f("ix_research_memory_records_collection_id"),
        table_name="research_memory_records",
    )
    op.drop_table("research_memory_records")
    op.drop_table("study_briefs")
