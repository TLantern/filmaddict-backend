"""add_sequences_to_timelines

Revision ID: add_sequences_to_timelines
Revises: add_highlight_explanation_cache
Create Date: 2026-01-02 22:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'add_sequences_to_timelines'
down_revision: Union[str, None] = 'add_highlight_explanation_cache'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if column already exists (idempotent migration)
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'timelines' AND column_name = 'sequences')"
    ))
    column_exists = result.scalar()
    
    if not column_exists:
        op.add_column('timelines', sa.Column('sequences', postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    # Check if column exists before dropping
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'timelines' AND column_name = 'sequences')"
    ))
    column_exists = result.scalar()
    
    if column_exists:
        op.drop_column('timelines', 'sequences')

