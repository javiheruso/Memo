"""add conversation owner session

Revision ID: a4d8f2c6b9e1
Revises: c3d4e5f6a7b8
Create Date: 2026-11-05 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d8f2c6b9e1"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("owner_session_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        op.f("ix_conversations_owner_session_id"),
        "conversations",
        ["owner_session_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_conversations_owner_session_id"), table_name="conversations"
    )
    op.drop_column("conversations", "owner_session_id")
