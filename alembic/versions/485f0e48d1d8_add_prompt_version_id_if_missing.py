"""add_prompt_version_id_if_missing

Revision ID: 485f0e48d1d8
Revises: 002
Create Date: 2025-11-24 23:09:54.997777

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '485f0e48d1d8'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if column exists before adding
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('highlights')]
    
    if 'prompt_version_id' not in columns:
        op.add_column('highlights', sa.Column('prompt_version_id', postgresql.UUID(as_uuid=True), nullable=True))
        op.create_foreign_key(
            'fk_highlights_prompt_version',
            'highlights',
            'prompt_versions',
            ['prompt_version_id'],
            ['id'],
            ondelete='SET NULL',
        )
        op.create_index('idx_highlights_prompt_version_id', 'highlights', ['prompt_version_id'])


def downgrade() -> None:
    op.drop_index('idx_highlights_prompt_version_id', table_name='highlights')
    op.drop_constraint('fk_highlights_prompt_version', 'highlights', type_='foreignkey')
    op.drop_column('highlights', 'prompt_version_id')

