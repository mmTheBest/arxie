"""Add first-class research agent runtime traces."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260520_0004"
down_revision = "20260508_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_agent_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=True),
        sa.Column("skill_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("model_policy", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("input_json", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["research_artifacts.id"]),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["thread_id"], ["research_threads.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_agent_runs_thread_id"), "research_agent_runs", ["thread_id"], unique=False)
    op.create_index(op.f("ix_research_agent_runs_artifact_id"), "research_agent_runs", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_research_agent_runs_collection_id"), "research_agent_runs", ["collection_id"], unique=False)
    op.create_index(op.f("ix_research_agent_runs_workspace_id"), "research_agent_runs", ["workspace_id"], unique=False)
    op.create_index(op.f("ix_research_agent_runs_skill_id"), "research_agent_runs", ["skill_id"], unique=False)
    op.create_index(op.f("ix_research_agent_runs_artifact_type"), "research_agent_runs", ["artifact_type"], unique=False)
    op.create_index(op.f("ix_research_agent_runs_status"), "research_agent_runs", ["status"], unique=False)

    op.create_table(
        "research_agent_steps",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("input_json", sa.JSON(), nullable=False),
        sa.Column("output_json", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_agent_steps_run_id"), "research_agent_steps", ["run_id"], unique=False)
    op.create_index(op.f("ix_research_agent_steps_step_type"), "research_agent_steps", ["step_type"], unique=False)

    op.create_table(
        "study_context_packs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=True),
        sa.Column("task_type", sa.String(length=128), nullable=False),
        sa.Column("cache_key", sa.String(length=255), nullable=True),
        sa.Column("context_json", sa.JSON(), nullable=False),
        sa.Column("selected_item_counts_json", sa.JSON(), nullable=False),
        sa.Column("readiness_warnings_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["research_agent_runs.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_study_context_packs_run_id"), "study_context_packs", ["run_id"], unique=False)
    op.create_index(op.f("ix_study_context_packs_collection_id"), "study_context_packs", ["collection_id"], unique=False)
    op.create_index(op.f("ix_study_context_packs_workspace_id"), "study_context_packs", ["workspace_id"], unique=False)
    op.create_index(op.f("ix_study_context_packs_task_type"), "study_context_packs", ["task_type"], unique=False)
    op.create_index(op.f("ix_study_context_packs_cache_key"), "study_context_packs", ["cache_key"], unique=False)

    op.create_table(
        "research_validation_reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("harness_status", sa.String(length=64), nullable=False),
        sa.Column("missing_evidence_json", sa.JSON(), nullable=False),
        sa.Column("unsupported_claims_json", sa.JSON(), nullable=False),
        sa.Column("readiness_blockers_json", sa.JSON(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["research_artifacts.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["research_agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_validation_reports_run_id"), "research_validation_reports", ["run_id"], unique=False)
    op.create_index(op.f("ix_research_validation_reports_artifact_id"), "research_validation_reports", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_research_validation_reports_harness_status"), "research_validation_reports", ["harness_status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_research_validation_reports_harness_status"), table_name="research_validation_reports")
    op.drop_index(op.f("ix_research_validation_reports_artifact_id"), table_name="research_validation_reports")
    op.drop_index(op.f("ix_research_validation_reports_run_id"), table_name="research_validation_reports")
    op.drop_table("research_validation_reports")
    op.drop_index(op.f("ix_study_context_packs_cache_key"), table_name="study_context_packs")
    op.drop_index(op.f("ix_study_context_packs_task_type"), table_name="study_context_packs")
    op.drop_index(op.f("ix_study_context_packs_workspace_id"), table_name="study_context_packs")
    op.drop_index(op.f("ix_study_context_packs_collection_id"), table_name="study_context_packs")
    op.drop_index(op.f("ix_study_context_packs_run_id"), table_name="study_context_packs")
    op.drop_table("study_context_packs")
    op.drop_index(op.f("ix_research_agent_steps_step_type"), table_name="research_agent_steps")
    op.drop_index(op.f("ix_research_agent_steps_run_id"), table_name="research_agent_steps")
    op.drop_table("research_agent_steps")
    op.drop_index(op.f("ix_research_agent_runs_status"), table_name="research_agent_runs")
    op.drop_index(op.f("ix_research_agent_runs_artifact_type"), table_name="research_agent_runs")
    op.drop_index(op.f("ix_research_agent_runs_skill_id"), table_name="research_agent_runs")
    op.drop_index(op.f("ix_research_agent_runs_workspace_id"), table_name="research_agent_runs")
    op.drop_index(op.f("ix_research_agent_runs_collection_id"), table_name="research_agent_runs")
    op.drop_index(op.f("ix_research_agent_runs_artifact_id"), table_name="research_agent_runs")
    op.drop_index(op.f("ix_research_agent_runs_thread_id"), table_name="research_agent_runs")
    op.drop_table("research_agent_runs")
