"""add_segment_feedback

Revision ID: d12a13dc6648
Revises: fb0789785290
Create Date: 2025-01-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd12a13dc6648'
down_revision: Union[str, None] = 'fb0789785290'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'segment_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('video_segment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('feedback_type', sa.String(), nullable=False),  # GREAT, FINE, WRONG
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['video_segment_id'], ['video_segments.id'], ondelete='CASCADE'),
    )
    op.create_index('idx_segment_feedback_video_segment_id', 'segment_feedback', ['video_segment_id'])
    op.create_index('idx_segment_feedback_created_at', 'segment_feedback', ['created_at'])
    op.create_index('idx_segment_feedback_type', 'segment_feedback', ['feedback_type'])


def downgrade() -> None:
    op.drop_index('idx_segment_feedback_type', table_name='segment_feedback')
    op.drop_index('idx_segment_feedback_created_at', table_name='segment_feedback')
    op.drop_index('idx_segment_feedback_video_segment_id', table_name='segment_feedback')
    op.drop_table('segment_feedback')
