"""add_pending_cuts_to_videos

Revision ID: 483fa25f6dc0
Revises: fb358a6319f1
Create Date: 2025-12-27 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '483fa25f6dc0'
down_revision: Union[str, None] = 'fb358a6319f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('pending_cuts', postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column('videos', 'pending_cuts')

