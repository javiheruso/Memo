"""add claims ingestion baseline

Revision ID: a9b1c2d3e4f5
Revises: 2c0f8a9b1d3e
Create Date: 2026-08-31 07:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a9b1c2d3e4f5'
down_revision: Union[str, None] = '2c0f8a9b1d3e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'claims',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('speaker_person_id', sa.Uuid(), nullable=True),
        sa.Column('canonical_subject', sa.String(length=255), nullable=True),
        sa.Column('canonical_predicate', sa.String(length=255), nullable=True),
        sa.Column('canonical_object', sa.String(length=500), nullable=True),
        sa.Column('original_text', sa.Text(), nullable=False),
        sa.Column('conversation_id', sa.Uuid(), nullable=False),
        sa.Column('message_id', sa.Uuid(), nullable=False),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.Enum('candidate', 'rejected', name='claim_status'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], name=op.f('fk_claims_conversation_id_conversations')),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], name=op.f('fk_claims_message_id_messages')),
        sa.ForeignKeyConstraint(['speaker_person_id'], ['persons.id'], name=op.f('fk_claims_speaker_person_id_persons')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_claims')),
    )


def downgrade() -> None:
    op.drop_table('claims')
    op.execute('DROP TYPE IF EXISTS claim_status')
