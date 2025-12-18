"""Add title and summary columns, remove reason from highlights table

Revision ID: 006_add_highlight_summary
Revises: 005_add_prompt_version_rolling_metrics
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('highlights', sa.Column('title', sa.String(), nullable=True))
    op.add_column('highlights', sa.Column('summary', sa.Text(), nullable=True))
    op.drop_column('highlights', 'reason')


def downgrade() -> None:
    op.add_column('highlights', sa.Column('reason', sa.String(), nullable=False, server_default=''))
    op.drop_column('highlights', 'summary')
    op.drop_column('highlights', 'title')

