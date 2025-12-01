"""add prompt version rolling metrics

Revision ID: 005
Revises: 004
Create Date: 2024-01-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("prompt_versions", sa.Column("total_rated", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("prompt_versions", sa.Column("sum_ratings", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("prompt_versions", sa.Column("avg_rating", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("prompt_versions", sa.Column("num_positive", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("prompt_versions", sa.Column("num_negative", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("prompt_versions", sa.Column("positive_rate", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("prompt_versions", sa.Column("negative_rate", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("prompt_versions", sa.Column("total_saves", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("prompt_versions", sa.Column("save_rate", sa.Float(), nullable=False, server_default="0.0"))


def downgrade() -> None:
    op.drop_column("prompt_versions", "save_rate")
    op.drop_column("prompt_versions", "total_saves")
    op.drop_column("prompt_versions", "negative_rate")
    op.drop_column("prompt_versions", "positive_rate")
    op.drop_column("prompt_versions", "num_negative")
    op.drop_column("prompt_versions", "num_positive")
    op.drop_column("prompt_versions", "avg_rating")
    op.drop_column("prompt_versions", "sum_ratings")
    op.drop_column("prompt_versions", "total_rated")

