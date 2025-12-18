"""change_default_aspect_ratio_to_16_9

Revision ID: 2ff6e4a1d3a8
Revises: 431434e1f47e
Create Date: 2025-12-11 21:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2ff6e4a1d3a8'
down_revision: Union[str, None] = '431434e1f47e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update server default from 9:16 to 16:9
    op.alter_column('videos', 'aspect_ratio',
                   server_default='16:9')
    
    # Update existing videos with 9:16 to 16:9
    op.execute("UPDATE videos SET aspect_ratio = '16:9' WHERE aspect_ratio = '9:16'")


def downgrade() -> None:
    # Revert server default back to 9:16
    op.alter_column('videos', 'aspect_ratio',
                   server_default='9:16')
    
    # Revert existing videos back to 9:16
    op.execute("UPDATE videos SET aspect_ratio = '9:16' WHERE aspect_ratio = '16:9'")
