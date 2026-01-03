"""add_clerk_user_id_to_videos

Revision ID: a1b2c3d4e5f6
Revises: 483fa25f6dc0
Create Date: 2025-01-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '483fa25f6dc0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if column already exists (idempotent migration)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'videos' AND column_name = 'clerk_user_id')"
    ))
    column_exists = result.scalar()
    
    if not column_exists:
        op.add_column('videos', sa.Column('clerk_user_id', sa.String(), nullable=True))
    
    # Check if index exists
    result = conn.execute(sa.text(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'videos' AND indexname = 'idx_videos_clerk_user_id'"
    ))
    index_exists = result.scalar()
    
    if not index_exists:
        op.create_index('idx_videos_clerk_user_id', 'videos', ['clerk_user_id'])


def downgrade() -> None:
    op.drop_index('idx_videos_clerk_user_id', table_name='videos')
    op.drop_column('videos', 'clerk_user_id')

