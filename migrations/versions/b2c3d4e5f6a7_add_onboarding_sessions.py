"""add onboarding sessions

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-11-01 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

onboarding_status = sa.Enum(
    'draft',
    'activated',
    name='onboarding_status',
)


def upgrade() -> None:
    op.create_table(
        'onboarding_sessions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('status', onboarding_status, nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('consent_confirmed', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('consent_confirmed_by', sa.String(length=255), nullable=True),
        sa.Column('consent_confirmed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('consent_scope', sa.Text(), nullable=True),
        sa.Column('subject_person_id', sa.Uuid(), nullable=True),
        sa.Column('profile_id', sa.Uuid(), nullable=True),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['subject_person_id'], ['persons.id'], name=op.f('fk_onboarding_sessions_subject_person_id_persons')),
        sa.ForeignKeyConstraint(['profile_id'], ['conversation_profiles.id'], name=op.f('fk_onboarding_sessions_profile_id_conversation_profiles')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_onboarding_sessions')),
    )
    op.create_index(
        op.f('ix_onboarding_sessions_status'),
        'onboarding_sessions',
        ['status'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_onboarding_sessions_status'),
        table_name='onboarding_sessions',
    )
    op.drop_table('onboarding_sessions')
    op.execute('DROP TYPE IF EXISTS onboarding_status')
