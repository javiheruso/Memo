"""add knowledge gap conversation association

Revision ID: 2c0f8a9b1d3e
Revises: 9a2ff7929222
Create Date: 2026-08-30 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '2c0f8a9b1d3e'
down_revision: Union[str, None] = '9a2ff7929222'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'knowledge_gap_conversations',
        sa.Column('gap_id', sa.Uuid(), nullable=False),
        sa.Column('conversation_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], name=op.f('fk_knowledge_gap_conversations_conversation_id_conversations')),
        sa.ForeignKeyConstraint(['gap_id'], ['knowledge_gaps.id'], name=op.f('fk_knowledge_gap_conversations_gap_id_knowledge_gaps')),
        sa.PrimaryKeyConstraint('gap_id', 'conversation_id', name=op.f('pk_knowledge_gap_conversations')),
    )


def downgrade() -> None:
    op.drop_table('knowledge_gap_conversations')
