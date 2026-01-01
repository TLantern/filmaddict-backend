"""merge_heads_clerk_user_id

Revision ID: merge_heads_001
Revises: a1b2c3d4e5f6, d12a13dc6648
Create Date: 2025-01-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'merge_heads_001'
down_revision: Union[str, tuple[str, ...], None] = ('a1b2c3d4e5f6', 'd12a13dc6648')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

