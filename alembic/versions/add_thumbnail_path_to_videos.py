"""add_thumbnail_path_to_videos

Revision ID: add_thumbnail_path_videos
Revises: merge_heads_001
Create Date: 2025-01-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_thumbnail_path_videos'
down_revision: Union[str, None] = 'merge_heads_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if column already exists (idempotent migration)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'videos' AND column_name = 'thumbnail_path')"
    ))
    column_exists = result.scalar()
    
    if not column_exists:
        op.add_column('videos', sa.Column('thumbnail_path', sa.String(), nullable=True))


def downgrade() -> None:
    # Check if column exists before dropping
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'videos' AND column_name = 'thumbnail_path')"
    ))
    column_exists = result.scalar()
    
    if column_exists:
        op.drop_column('videos', 'thumbnail_path')

