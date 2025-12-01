"""add online calibration fields

Revision ID: 004
Revises: 003
Create Date: 2024-01-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("calibration_config", sa.Column("feedback_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("calibration_config", sa.Column("sum_predicted", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("calibration_config", sa.Column("sum_actual", sa.Float(), nullable=False, server_default="0.0"))


def downgrade() -> None:
    op.drop_column("calibration_config", "sum_actual")
    op.drop_column("calibration_config", "sum_predicted")
    op.drop_column("calibration_config", "feedback_count")

