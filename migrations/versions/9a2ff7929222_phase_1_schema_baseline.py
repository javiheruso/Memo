"""phase 1 schema baseline

Revision ID: 9a2ff7929222
Revises:
Create Date: 2026-08-30 17:31:32.040539

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = '9a2ff7929222'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        'conversations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('initiated_by', sa.String(length=50), nullable=False),
        sa.Column('initiation_reason', sa.String(length=500), nullable=True),
        sa.Column('device_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_conversations')),
    )

    op.create_table(
        'persons',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('first_name', sa.String(length=255), nullable=False),
        sa.Column('last_name', sa.String(length=255), nullable=True),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('birth_date', sa.Date(), nullable=True),
        sa.Column('death_date', sa.Date(), nullable=True),
        sa.Column('is_deceased', sa.Boolean(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('primary_media_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_persons')),
    )

    op.create_table(
        'sources',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('source_type', sa.Enum('family_member', 'user', 'conversation', 'document', 'photo', 'calendar', 'system', 'other', name='source_type'), nullable=False),
        sa.Column('person_id', sa.Uuid(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('external_reference', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_sources_person_id_persons')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_sources')),
    )

    op.create_table(
        'media',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('media_type', sa.String(length=50), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('approx_date', sa.Date(), nullable=True),
        sa.Column('place_text', sa.String(length=500), nullable=True),
        sa.Column('source_id', sa.Uuid(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_media_source_id_sources')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_media')),
    )

    op.create_foreign_key(
        op.f('fk_persons_primary_media_id_media'),
        'persons', 'media',
        ['primary_media_id'], ['id'],
    )

    op.create_table(
        'person_aliases',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('person_id', sa.Uuid(), nullable=False),
        sa.Column('alias', sa.String(length=255), nullable=False),
        sa.Column('alias_type', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_person_aliases_person_id_persons')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_person_aliases')),
    )

    op.create_table(
        'relationships',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('person_a_id', sa.Uuid(), nullable=False),
        sa.Column('person_b_id', sa.Uuid(), nullable=False),
        sa.Column('relationship_type', sa.Enum('parent_of', 'child_of', 'grandparent_of', 'grandchild_of', 'sibling_of', 'spouse_of', 'partner_of', 'friend_of', 'other', name='relationship_type'), nullable=False),
        sa.Column('status', sa.Enum('verified', 'probable', 'pending', 'rejected', name='relationship_status'), nullable=False),
        sa.Column('source_id', sa.Uuid(), nullable=True),
        sa.Column('valid_from', sa.Date(), nullable=True),
        sa.Column('valid_to', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['person_a_id'], ['persons.id'], name=op.f('fk_relationships_person_a_id_persons')),
        sa.ForeignKeyConstraint(['person_b_id'], ['persons.id'], name=op.f('fk_relationships_person_b_id_persons')),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_relationships_source_id_sources')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_relationships')),
    )

    op.create_table(
        'facts',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('subject_person_id', sa.Uuid(), nullable=False),
        sa.Column('fact_type', sa.String(length=50), nullable=False),
        sa.Column('value_text', sa.Text(), nullable=True),
        sa.Column('value_date', sa.Date(), nullable=True),
        sa.Column('value_number', sa.Numeric(), nullable=True),
        sa.Column('related_person_id', sa.Uuid(), nullable=True),
        sa.Column('place_text', sa.String(length=500), nullable=True),
        sa.Column('valid_from', sa.Date(), nullable=True),
        sa.Column('valid_to', sa.Date(), nullable=True),
        sa.Column('status', sa.Enum('verified', 'probable', 'pending', 'rejected', name='fact_status'), nullable=False),
        sa.Column('source_id', sa.Uuid(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['related_person_id'], ['persons.id'], name=op.f('fk_facts_related_person_id_persons')),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_facts_source_id_sources')),
        sa.ForeignKeyConstraint(['subject_person_id'], ['persons.id'], name=op.f('fk_facts_subject_person_id_persons')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_facts')),
    )

    op.create_table(
        'memories',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('narrative', sa.Text(), nullable=False),
        sa.Column('approx_start_date', sa.Date(), nullable=True),
        sa.Column('approx_end_date', sa.Date(), nullable=True),
        sa.Column('approx_period_text', sa.String(length=255), nullable=True),
        sa.Column('place_text', sa.String(length=500), nullable=True),
        sa.Column('memory_status', sa.Enum('candidate', 'recurring', 'consolidated', 'archived', name='memory_status'), nullable=False),
        sa.Column('sensitivity_level', sa.Enum('low', 'medium', 'high', name='sensitivity_level'), nullable=False),
        sa.Column('source_id', sa.Uuid(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_memories_source_id_sources')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_memories')),
    )

    op.create_table(
        'memory_people',
        sa.Column('memory_id', sa.Uuid(), nullable=False),
        sa.Column('person_id', sa.Uuid(), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['memory_id'], ['memories.id'], name=op.f('fk_memory_people_memory_id_memories')),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_memory_people_person_id_persons')),
        sa.PrimaryKeyConstraint('memory_id', 'person_id', name=op.f('pk_memory_people')),
    )

    op.create_table(
        'media_people',
        sa.Column('media_id', sa.Uuid(), nullable=False),
        sa.Column('person_id', sa.Uuid(), nullable=False),
        sa.Column('is_primary', sa.Boolean(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['media_id'], ['media.id'], name=op.f('fk_media_people_media_id_media')),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_media_people_person_id_persons')),
        sa.PrimaryKeyConstraint('media_id', 'person_id', name=op.f('pk_media_people')),
    )

    op.create_table(
        'memory_media',
        sa.Column('memory_id', sa.Uuid(), nullable=False),
        sa.Column('media_id', sa.Uuid(), nullable=False),
        sa.Column('relation_type', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['media_id'], ['media.id'], name=op.f('fk_memory_media_media_id_media')),
        sa.ForeignKeyConstraint(['memory_id'], ['memories.id'], name=op.f('fk_memory_media_memory_id_memories')),
        sa.PrimaryKeyConstraint('memory_id', 'media_id', name=op.f('pk_memory_media')),
    )

    op.create_table(
        'events',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('start_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Enum('planned', 'confirmed', 'cancelled', 'completed', name='event_status'), nullable=False),
        sa.Column('source_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id'], name=op.f('fk_events_source_id_sources')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_events')),
    )

    op.create_table(
        'event_people',
        sa.Column('event_id', sa.Uuid(), nullable=False),
        sa.Column('person_id', sa.Uuid(), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['event_id'], ['events.id'], name=op.f('fk_event_people_event_id_events')),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_event_people_person_id_persons')),
        sa.PrimaryKeyConstraint('event_id', 'person_id', name=op.f('pk_event_people')),
    )

    op.create_table(
        'messages',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('conversation_id', sa.Uuid(), nullable=False),
        sa.Column('speaker', sa.String(length=50), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('intent', sa.String(length=100), nullable=True),
        sa.Column('audio_path', sa.String(length=500), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], name=op.f('fk_messages_conversation_id_conversations')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_messages')),
    )

    op.create_table(
        'response_sources',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('message_id', sa.Uuid(), nullable=False),
        sa.Column('source_entity_type', sa.String(length=50), nullable=False),
        sa.Column('source_entity_id', sa.Uuid(), nullable=False),
        sa.Column('usage_type', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], name=op.f('fk_response_sources_message_id_messages')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_response_sources')),
    )

    op.create_table(
        'knowledge_gaps',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('gap_type', sa.String(length=50), nullable=False),
        sa.Column('canonical_reference', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('occurrence_count', sa.Integer(), nullable=False),
        sa.Column('distinct_sessions', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('observed', 'recurrent', 'family_review', 'resolved', 'ignored', name='knowledge_gap_status'), nullable=False),
        sa.Column('resolved_person_id', sa.Uuid(), nullable=True),
        sa.Column('resolved_fact_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['resolved_fact_id'], ['facts.id'], name=op.f('fk_knowledge_gaps_resolved_fact_id_facts')),
        sa.ForeignKeyConstraint(['resolved_person_id'], ['persons.id'], name=op.f('fk_knowledge_gaps_resolved_person_id_persons')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_knowledge_gaps')),
    )


def downgrade() -> None:
    op.drop_constraint(op.f('fk_persons_primary_media_id_media'), 'persons', type_='foreignkey')

    op.drop_table('response_sources')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('event_people')
    op.drop_table('events')
    op.drop_table('memory_media')
    op.drop_table('media_people')
    op.drop_table('memory_people')
    op.drop_table('memories')
    op.drop_table('knowledge_gaps')
    op.drop_table('facts')
    op.drop_table('person_aliases')
    op.drop_table('relationships')
    op.drop_table('media')
    op.drop_table('sources')
    op.drop_table('persons')

    op.execute('DROP TYPE IF EXISTS knowledge_gap_status')
    op.execute('DROP TYPE IF EXISTS event_status')
    op.execute('DROP TYPE IF EXISTS sensitivity_level')
    op.execute('DROP TYPE IF EXISTS memory_status')
    op.execute('DROP TYPE IF EXISTS fact_status')
    op.execute('DROP TYPE IF EXISTS source_type')
    op.execute('DROP TYPE IF EXISTS relationship_type')
