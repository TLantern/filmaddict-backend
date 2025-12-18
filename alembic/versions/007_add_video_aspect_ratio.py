"""Add aspect_ratio column to videos table

Revision ID: 007_add_video_aspect_ratio
Revises: 006_add_highlight_summary
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('aspect_ratio', sa.String(), nullable=True, server_default='16:9'))


def downgrade() -> None:
    op.drop_column('videos', 'aspect_ratio')

