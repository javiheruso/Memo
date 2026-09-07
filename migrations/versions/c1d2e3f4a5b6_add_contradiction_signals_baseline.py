"""add contradiction signals baseline

Revision ID: c1d2e3f4a5b6
Revises: b1c2d3e4f5a6b7c8
Create Date: 2026-08-31 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'c1d2e3f4a5b6'
down_revision: Union[str, None] = 'b1c2d3e4f5a6b7c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'claim_groups',
        sa.Column(
            'needs_review',
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )

    op.create_table(
        'claim_group_contradictions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('claim_group_id', sa.Uuid(), nullable=False),
        sa.Column('fact_id', sa.Uuid(), nullable=False),
        sa.Column('detector_name', sa.String(length=50), nullable=False),
        sa.Column('matched_terms', sa.Text(), nullable=True),
        sa.Column('first_detected_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_detected_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('detection_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['claim_group_id'], ['claim_groups.id'], name=op.f('fk_claim_group_contradictions_claim_group_id_claim_groups')),
        sa.ForeignKeyConstraint(['fact_id'], ['facts.id'], name=op.f('fk_claim_group_contradictions_fact_id_facts')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_group_contradictions')),
        sa.UniqueConstraint('claim_group_id', 'fact_id', 'detector_name', name=op.f('uq_claim_group_contradictions_group_fact_detector')),
    )
    op.create_index(op.f('ix_claim_group_contradictions_claim_group_id'), 'claim_group_contradictions', ['claim_group_id'], unique=False)
    op.create_index(op.f('ix_claim_group_contradictions_fact_id'), 'claim_group_contradictions', ['fact_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_claim_group_contradictions_fact_id'), table_name='claim_group_contradictions')
    op.drop_index(op.f('ix_claim_group_contradictions_claim_group_id'), table_name='claim_group_contradictions')
    op.drop_table('claim_group_contradictions')
    op.drop_column('claim_groups', 'needs_review')
