"""rename_rating_to_confidence_score_in_feedback

Revision ID: 44f68d512870
Revises: 2ff6e4a1d3a8
Create Date: 2025-12-12 17:09:20.364215

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '44f68d512870'
down_revision: Union[str, None] = '2ff6e4a1d3a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if old column exists and rename it, or create new one if it doesn't exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col['name'] for col in inspector.get_columns('highlight_feedback')}
    
    # Rename rating to confidence_score if it exists
    if 'rating' in columns and 'confidence_score' not in columns:
        op.alter_column('highlight_feedback', 'rating',
                       new_column_name='confidence_score')
    elif 'confidence_score' not in columns:
        # Create the column if neither exists
        op.add_column('highlight_feedback',
                     sa.Column('confidence_score', sa.Float(), nullable=True))


def downgrade() -> None:
    # Rename back
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col['name'] for col in inspector.get_columns('highlight_feedback')}
    
    if 'confidence_score' in columns:
        op.alter_column('highlight_feedback', 'confidence_score',
                       new_column_name='rating')

