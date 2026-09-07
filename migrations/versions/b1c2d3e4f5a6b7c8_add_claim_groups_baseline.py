"""add claim groups baseline

Revision ID: b1c2d3e4f5a6b7c8
Revises: a9b1c2d3e4f5
Create Date: 2026-08-31 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b1c2d3e4f5a6b7c8'
down_revision: Union[str, None] = 'a9b1c2d3e4f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'claim_groups',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('canonical_text', sa.String(length=500), nullable=False),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('total_mentions', sa.Integer(), nullable=False),
        sa.Column('distinct_conversations', sa.Integer(), nullable=False),
        sa.Column('distinct_days', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_groups')),
        sa.UniqueConstraint('canonical_text', name=op.f('uq_claim_groups_canonical_text')),
    )

    op.create_table(
        'claim_group_members',
        sa.Column('claim_group_id', sa.Uuid(), nullable=False),
        sa.Column('claim_id', sa.Uuid(), nullable=False),
        sa.Column('added_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['claim_group_id'], ['claim_groups.id'], name=op.f('fk_claim_group_members_claim_group_id_claim_groups')),
        sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_claim_group_members_claim_id_claims')),
        sa.PrimaryKeyConstraint('claim_group_id', 'claim_id', name=op.f('pk_claim_group_members')),
        sa.UniqueConstraint('claim_id', name=op.f('uq_claim_group_members_claim_id')),
    )


def downgrade() -> None:
    op.drop_table('claim_group_members')
    op.drop_table('claim_groups')
