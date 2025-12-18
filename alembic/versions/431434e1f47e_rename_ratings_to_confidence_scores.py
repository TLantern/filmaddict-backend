"""rename_ratings_to_confidence_scores

Revision ID: 431434e1f47e
Revises: 007
Create Date: 2025-12-11 21:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '431434e1f47e'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if old columns exist and rename them, or create new ones if they don't exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col['name'] for col in inspector.get_columns('prompt_versions')}
    
    # Rename sum_ratings to sum_confidence_scores if it exists
    if 'sum_ratings' in columns and 'sum_confidence_scores' not in columns:
        op.alter_column('prompt_versions', 'sum_ratings',
                       new_column_name='sum_confidence_scores')
    elif 'sum_confidence_scores' not in columns:
        # Create the column if neither exists
        op.add_column('prompt_versions',
                     sa.Column('sum_confidence_scores', sa.Float(), nullable=False, server_default='0.0'))
    
    # Rename avg_rating to avg_confidence_score if it exists
    if 'avg_rating' in columns and 'avg_confidence_score' not in columns:
        op.alter_column('prompt_versions', 'avg_rating',
                       new_column_name='avg_confidence_score')
    elif 'avg_confidence_score' not in columns:
        # Create the column if neither exists
        op.add_column('prompt_versions',
                     sa.Column('avg_confidence_score', sa.Float(), nullable=False, server_default='0.0'))


def downgrade() -> None:
    # Rename back
    op.alter_column('prompt_versions', 'sum_confidence_scores',
                   new_column_name='sum_ratings')
    op.alter_column('prompt_versions', 'avg_confidence_score',
                   new_column_name='avg_rating')
