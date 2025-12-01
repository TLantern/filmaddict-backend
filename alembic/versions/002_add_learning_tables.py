"""add learning tables

Revision ID: 002
Revises: 001
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "highlight_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("highlight_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feedback_type", sa.String(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["highlight_id"], ["highlights.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_highlight_feedback_highlight_id", "highlight_feedback", ["highlight_id"])
    op.create_index("idx_highlight_feedback_created_at", "highlight_feedback", ["created_at"])
    op.create_index("idx_highlight_feedback_type", "highlight_feedback", ["feedback_type"])

    op.create_table(
        "prompt_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version_name", sa.String(), nullable=False, unique=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("user_prompt_template", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=False),
        sa.Column("performance_metrics", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_prompt_versions_version_name", "prompt_versions", ["version_name"])
    op.create_index("idx_prompt_versions_is_active", "prompt_versions", ["is_active"])

    op.create_table(
        "calibration_config",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("score_offset", sa.Float(), nullable=False, default=0.0),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False, default=0),
    )

    op.add_column("highlights", sa.Column("prompt_version_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_highlights_prompt_version",
        "highlights",
        "prompt_versions",
        ["prompt_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("idx_highlights_prompt_version_id", "highlights", ["prompt_version_id"])


def downgrade() -> None:
    op.drop_index("idx_highlights_prompt_version_id", table_name="highlights")
    op.drop_constraint("fk_highlights_prompt_version", "highlights", type_="foreignkey")
    op.drop_column("highlights", "prompt_version_id")

    op.drop_table("calibration_config")
    op.drop_index("idx_prompt_versions_is_active", table_name="prompt_versions")
    op.drop_index("idx_prompt_versions_version_name", table_name="prompt_versions")
    op.drop_table("prompt_versions")
    op.drop_index("idx_highlight_feedback_type", table_name="highlight_feedback")
    op.drop_index("idx_highlight_feedback_created_at", table_name="highlight_feedback")
    op.drop_index("idx_highlight_feedback_highlight_id", table_name="highlight_feedback")
    op.drop_table("highlight_feedback")

