"""add_grade_to_video_segments

Revision ID: fb358a6319f1
Revises: a4986ec328d
Create Date: 2025-12-24 11:05:48.371430

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fb358a6319f1'
down_revision: Union[str, None] = 'a4986ec328d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('video_segments', sa.Column('grade', sa.String(), nullable=False, server_default='C'))


def downgrade() -> None:
    op.drop_column('video_segments', 'grade')

