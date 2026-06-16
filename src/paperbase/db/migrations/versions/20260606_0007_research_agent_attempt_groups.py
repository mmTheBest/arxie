"""Group retried research-agent trace records by attempt."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260606_0007"
down_revision = "20260604_0006"
branch_labels = None
depends_on = None

ATTEMPT_TABLES = (
    "research_agent_steps",
    "study_context_packs",
    "research_validation_reports",
)


def upgrade() -> None:
    for table_name in ATTEMPT_TABLES:
        op.add_column(
            table_name,
            sa.Column(
                "attempt_number",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("1"),
            ),
        )


def downgrade() -> None:
    for table_name in reversed(ATTEMPT_TABLES):
        op.drop_column(table_name, "attempt_number")
