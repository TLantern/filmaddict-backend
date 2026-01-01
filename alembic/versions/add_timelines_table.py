"""add_timelines_table

Revision ID: add_timelines_table
Revises: add_thumbnail_path_videos
Create Date: 2025-12-30 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'add_timelines_table'
down_revision: Union[str, None] = 'add_thumbnail_path_videos'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if table already exists (idempotent migration)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'timelines')"
    ))
    table_exists = result.scalar()
    
    if not table_exists:
        op.create_table(
            'timelines',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
            sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('clerk_user_id', sa.String(), nullable=True),
            sa.Column('project_name', sa.String(), nullable=True),
            sa.Column('markers', postgresql.JSONB(), nullable=True),
            sa.Column('selections', postgresql.JSONB(), nullable=True),
            sa.Column('current_time', sa.Float(), nullable=True, server_default='0.0'),
            sa.Column('in_point', sa.Float(), nullable=True),
            sa.Column('out_point', sa.Float(), nullable=True),
            sa.Column('zoom', sa.Float(), nullable=True, server_default='1.0'),
            sa.Column('view_preferences', postgresql.JSONB(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
            sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
            sa.UniqueConstraint('video_id', name='uq_timelines_video_id'),
        )
        op.create_index('idx_timelines_video_id', 'timelines', ['video_id'])
        op.create_index('idx_timelines_clerk_user_id', 'timelines', ['clerk_user_id'])
        op.create_index('idx_timelines_created_at', 'timelines', ['created_at'])


def downgrade() -> None:
    op.drop_index('idx_timelines_created_at', table_name='timelines')
    op.drop_index('idx_timelines_clerk_user_id', table_name='timelines')
    op.drop_index('idx_timelines_video_id', table_name='timelines')
    op.drop_table('timelines')

