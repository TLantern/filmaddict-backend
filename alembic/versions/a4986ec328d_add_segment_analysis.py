"""add_segment_analysis

Revision ID: a4986ec328d
Revises: 44f68d512870
Create Date: 2025-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a4986ec328d'
down_revision: Union[str, None] = '44f68d512870'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'video_segments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('segment_id', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),  # FLUFF, REPEATED, USEFUL
        sa.Column('rating', sa.Float(), nullable=False),  # 0.0-1.0
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('repetition_score', sa.Float(), nullable=False),  # 0.0-1.0
        sa.Column('filler_density', sa.Float(), nullable=False),  # 0.0-1.0
        sa.Column('visual_change_score', sa.Float(), nullable=False),  # 0.0-1.0
        sa.Column('usefulness_score', sa.Float(), nullable=False),  # 0.0-1.0
        sa.Column('embedding', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
    )
    op.create_index('idx_video_segments_video_id', 'video_segments', ['video_id'])
    op.create_index('idx_video_segments_label', 'video_segments', ['label'])
    op.create_index('idx_video_segments_rating', 'video_segments', ['rating'])


def downgrade() -> None:
    op.drop_index('idx_video_segments_rating', table_name='video_segments')
    op.drop_index('idx_video_segments_label', table_name='video_segments')
    op.drop_index('idx_video_segments_video_id', table_name='video_segments')
    op.drop_table('video_segments')
