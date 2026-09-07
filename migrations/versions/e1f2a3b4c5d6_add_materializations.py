"""add materializations provenance table

Revision ID: e1f2a3b4c5d6
Revises: d1e2f3a4b5c6
Create Date: 2026-08-31 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

materialization_source_type = postgresql.ENUM(
    'claim_group',
    'knowledge_gap',
    name='materialization_source_type',
    create_type=False,
)

materialization_target_type = postgresql.ENUM(
    'fact',
    'memory',
    name='materialization_target_type',
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind is not None:
        materialization_source_type.create(bind, checkfirst=True)
        materialization_target_type.create(bind, checkfirst=True)
    op.create_table(
        'materializations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('source_type', materialization_source_type, nullable=False),
        sa.Column('source_id', sa.Uuid(), nullable=False),
        sa.Column('target_type', materialization_target_type, nullable=False),
        sa.Column('target_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_materializations')),
        sa.UniqueConstraint(
            'source_type', 'source_id', 'target_type',
            name='uq_materializations_source_target_type',
        ),
    )
    op.create_index('ix_materializations_source_id', 'materializations', ['source_id'])
    op.create_index('ix_materializations_target_id', 'materializations', ['target_id'])


def downgrade() -> None:
    op.drop_table('materializations')
    bind = op.get_bind()
    if bind is not None:
        materialization_target_type.drop(bind, checkfirst=True)
        materialization_source_type.drop(bind, checkfirst=True)
