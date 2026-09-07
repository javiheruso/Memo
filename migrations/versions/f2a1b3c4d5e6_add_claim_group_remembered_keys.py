"""add claim group remembered keys

Revision ID: f2a1b3c4d5e6
Revises: e1f2a3b4c5d6
Create Date: 2026-09-07 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'f2a1b3c4d5e6'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'claim_group_remembered_keys',
        sa.Column('canonical_text', sa.String(length=500), nullable=False),
        sa.Column('claim_group_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['claim_group_id'], ['claim_groups.id'], name=op.f('fk_claim_group_remembered_keys_claim_group_id_claim_groups')),
        sa.PrimaryKeyConstraint('canonical_text', name=op.f('pk_claim_group_remembered_keys')),
    )
    op.create_index(op.f('ix_claim_group_remembered_keys_claim_group_id'), 'claim_group_remembered_keys', ['claim_group_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_claim_group_remembered_keys_claim_group_id'), table_name='claim_group_remembered_keys')
    op.drop_table('claim_group_remembered_keys')
