"""add clip feedback features

Revision ID: 003
Revises: 002
Create Date: 2024-01-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "485f0e48d1d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # Add text_feedback column to highlight_feedback table if it doesn't exist
    highlight_feedback_columns = [col['name'] for col in inspector.get_columns('highlight_feedback')]
    if 'text_feedback' not in highlight_feedback_columns:
        op.add_column("highlight_feedback", sa.Column("text_feedback", sa.Text(), nullable=True))
    
    # Create saved_clips table if it doesn't exist
    tables = inspector.get_table_names()
    if 'saved_clips' not in tables:
        op.create_table(
            "saved_clips",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("clip_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("highlight_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["clip_id"], ["clips.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["highlight_id"], ["highlights.id"], ondelete="SET NULL"),
        )
        op.create_index("idx_saved_clips_clip_id", "saved_clips", ["clip_id"])
        op.create_index("idx_saved_clips_highlight_id", "saved_clips", ["highlight_id"])


def downgrade() -> None:
    # Drop saved_clips table and indexes
    op.drop_index("idx_saved_clips_highlight_id", table_name="saved_clips")
    op.drop_index("idx_saved_clips_clip_id", table_name="saved_clips")
    op.drop_table("saved_clips")
    
    # Drop text_feedback column from highlight_feedback
    op.drop_column("highlight_feedback", "text_feedback")

