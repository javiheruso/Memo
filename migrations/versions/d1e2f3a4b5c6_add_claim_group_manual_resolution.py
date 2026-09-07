"""add claim group manual resolution status

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-08-31 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'd1e2f3a4b5c6'
down_revision: Union[str, None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

claim_group_status = postgresql.ENUM(
    'active',
    'resolved',
    'ignored',
    name='claim_group_status',
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind is not None:
        claim_group_status.create(bind, checkfirst=True)
    op.add_column(
        'claim_groups',
        sa.Column(
            'status',
            claim_group_status,
            server_default='active',
            nullable=False,
        ),
    )
    op.add_column(
        'claim_groups',
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'claim_groups',
        sa.Column('resolution_note', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('claim_groups', 'resolution_note')
    op.drop_column('claim_groups', 'resolved_at')
    op.drop_column('claim_groups', 'status')
    bind = op.get_bind()
    if bind is not None:
        claim_group_status.drop(bind, checkfirst=True)
