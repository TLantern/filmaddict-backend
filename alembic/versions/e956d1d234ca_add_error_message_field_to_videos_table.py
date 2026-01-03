"""add_error_message_field_to_videos_table

Revision ID: e956d1d234ca
Revises: 483fa25f6dc0
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e956d1d234ca'
down_revision: Union[str, None] = '483fa25f6dc0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('error_message', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('videos', 'error_message')

