"""initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_videos_status", "videos", ["status"])
    op.create_index("idx_videos_created_at", "videos", ["created_at"])

    op.create_table(
        "transcripts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("segments", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_transcripts_video_id", "transcripts", ["video_id"])

    op.create_table(
        "highlights",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_highlights_video_id", "highlights", ["video_id"])
    op.create_index("idx_highlights_score", "highlights", ["score"])

    op.create_table(
        "clips",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("thumbnail_path", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_clips_video_id", "clips", ["video_id"])


def downgrade() -> None:
    op.drop_index("idx_clips_video_id", table_name="clips")
    op.drop_table("clips")
    op.drop_index("idx_highlights_score", table_name="highlights")
    op.drop_index("idx_highlights_video_id", table_name="highlights")
    op.drop_table("highlights")
    op.drop_index("idx_transcripts_video_id", table_name="transcripts")
    op.drop_table("transcripts")
    op.drop_index("idx_videos_created_at", table_name="videos")
    op.drop_index("idx_videos_status", table_name="videos")
    op.drop_table("videos")

