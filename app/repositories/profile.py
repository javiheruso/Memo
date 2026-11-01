import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.models import ConversationProfile, ConversationProfileSensitiveTopic
from app.repositories.base import SQLRepository
from app.schemas.profile import ConversationProfileCreate


class ConversationProfileRepository(SQLRepository[ConversationProfile]):
    model = ConversationProfile
    load_options = (selectinload(ConversationProfile.sensitive_topics),)

    async def get_by_person(self, person_id: uuid.UUID) -> ConversationProfile | None:
        stmt = (
            select(ConversationProfile)
            .where(ConversationProfile.person_id == person_id)
            .options(*self.load_options)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_active(self) -> ConversationProfile | None:
        stmt = (
            select(ConversationProfile)
            .where(ConversationProfile.is_active.is_(True))
            .options(*self.load_options)
            .order_by(ConversationProfile.created_at)
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def has_active(self) -> bool:
        return await self.get_active() is not None

    async def create(
        self,
        data: ConversationProfileCreate,
        *,
        entity_id: uuid.UUID | None = None,
        commit: bool = True,
    ) -> ConversationProfile:
        if data.is_active:
            await self._deactivate_all()
        id_kwargs = {"id": entity_id} if entity_id is not None else {}
        profile = ConversationProfile(
            **id_kwargs,
            person_id=data.person_id,
            preferred_name=data.preferred_name,
            language=data.language,
            conversational_style=data.conversational_style,
            positive_topics=list(data.positive_topics),
            beliefs_policy=data.beliefs_policy,
            never_do=list(data.never_do),
            orientation=data.orientation,
            data_origin=data.data_origin,
            source_id=data.source_id,
            is_fictional=data.is_fictional,
            is_active=data.is_active,
            sensitive_topics=[
                ConversationProfileSensitiveTopic(
                    topic=topic.topic,
                    handling=topic.handling,
                    consent_review_required=topic.consent_review_required,
                    triggers=list(topic.triggers),
                )
                for topic in data.sensitive_topics
            ],
        )
        self._session.add(profile)
        if commit:
            await self._session.commit()
        else:
            await self._session.flush()
        fresh = await self.get(profile.id)
        if fresh is None:
            raise RuntimeError("ConversationProfile disappeared after write")
        return fresh

    async def activate(self, profile_id: uuid.UUID) -> ConversationProfile | None:
        profile = await self.get(profile_id)
        if profile is None:
            return None
        await self._deactivate_all()
        profile.is_active = True
        await self._session.commit()
        return await self.get(profile_id)

    async def _deactivate_all(self) -> None:
        await self._session.execute(
            update(ConversationProfile).values(is_active=False)
        )
