"""add conversation profiles and sensitive topics

Revision ID: a1b2c3d4e5f6
Revises: f2a1b3c4d5e6
Create Date: 2026-10-31 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f2a1b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversation_profiles',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('person_id', sa.Uuid(), nullable=False),
        sa.Column('preferred_name', sa.String(length=255), nullable=True),
        sa.Column('language', sa.String(length=50), nullable=True),
        sa.Column('conversational_style', sa.Text(), nullable=True),
        sa.Column('positive_topics', sa.JSON(), nullable=False),
        sa.Column('beliefs_policy', sa.Text(), nullable=True),
        sa.Column('never_do', sa.JSON(), nullable=False),
        sa.Column('orientation', sa.Text(), nullable=True),
        sa.Column('data_origin', sa.Text(), nullable=True),
        sa.Column('source_id', sa.Uuid(), nullable=True),
        sa.Column('is_fictional', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_conversation_profiles_person_id_persons')),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_conversation_profiles_source_id_sources')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_conversation_profiles')),
        sa.UniqueConstraint('person_id', name=op.f('uq_conversation_profiles_person_id')),
    )
    op.create_index(
        op.f('ix_conversation_profiles_is_active'),
        'conversation_profiles',
        ['is_active'],
        unique=False,
    )
    op.create_table(
        'conversation_profile_sensitive_topics',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('profile_id', sa.Uuid(), nullable=False),
        sa.Column('topic', sa.String(length=255), nullable=False),
        sa.Column('handling', sa.Text(), nullable=True),
        sa.Column('consent_review_required', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('triggers', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['profile_id'], ['conversation_profiles.id'], name=op.f('fk_conversation_profile_sensitive_topics_profile_id_conversation_profiles')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_conversation_profile_sensitive_topics')),
        sa.UniqueConstraint(
            'profile_id', 'topic',
            name='uq_conversation_profile_sensitive_topics_profile_topic',
        ),
    )


def downgrade() -> None:
    op.drop_table('conversation_profile_sensitive_topics')
    op.drop_index(
        op.f('ix_conversation_profiles_is_active'),
        table_name='conversation_profiles',
    )
    op.drop_table('conversation_profiles')
