"""add_retention_metrics

Revision ID: fb0789785290
Revises: fb358a6319f1
Create Date: 2025-01-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fb0789785290'
down_revision: Union[str, None] = 'fb358a6319f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if table already exists (idempotent migration)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'retention_metrics')"
    ))
    table_exists = result.scalar()
    
    if not table_exists:
        op.create_table(
            'retention_metrics',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
            sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('segment_id', sa.Integer(), nullable=False),
            sa.Column('time_range', postgresql.JSONB(), nullable=False),  # {start, end, duration}
            sa.Column('text', sa.Text(), nullable=False),
            sa.Column('metrics', postgresql.JSONB(), nullable=False),  # {semantic_novelty, information_density, emotional_delta, narrative_momentum}
            sa.Column('retention_value', sa.Float(), nullable=False),
            sa.Column('decision', postgresql.JSONB(), nullable=False),  # {action, reason}
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
            sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        )
        op.create_index('idx_retention_metrics_video_id', 'retention_metrics', ['video_id'])
        op.create_index('idx_retention_metrics_segment_id', 'retention_metrics', ['segment_id'])
        op.create_index('idx_retention_metrics_retention_value', 'retention_metrics', ['retention_value'])
    else:
        # Table exists, check and create indexes if missing
        result = conn.execute(sa.text(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'retention_metrics'"
        ))
        existing_indexes = [row[0] for row in result.fetchall()]
        
        if 'idx_retention_metrics_video_id' not in existing_indexes:
            op.create_index('idx_retention_metrics_video_id', 'retention_metrics', ['video_id'])
        if 'idx_retention_metrics_segment_id' not in existing_indexes:
            op.create_index('idx_retention_metrics_segment_id', 'retention_metrics', ['segment_id'])
        if 'idx_retention_metrics_retention_value' not in existing_indexes:
            op.create_index('idx_retention_metrics_retention_value', 'retention_metrics', ['retention_value'])


def downgrade() -> None:
    op.drop_index('idx_retention_metrics_retention_value', table_name='retention_metrics')
    op.drop_index('idx_retention_metrics_segment_id', table_name='retention_metrics')
    op.drop_index('idx_retention_metrics_video_id', table_name='retention_metrics')
    op.drop_table('retention_metrics')

