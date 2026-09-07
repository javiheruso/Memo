import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.domain.enums import MediaType
from app.models import Media, MediaPerson
from app.repositories.base import SQLRepository
from app.schemas.media import MediaCreate


class MediaRepository(SQLRepository[Media]):
    model = Media
    load_options = (selectinload(Media.people),)

    async def create(self, data: MediaCreate) -> Media:
        media_type = (
            data.media_type.value
            if isinstance(data.media_type, MediaType)
            else str(data.media_type)
        )
        return await self._create(
            lambda: Media(
                media_type=media_type,
                storage_path=data.storage_path,
                title=data.title,
                description=data.description,
                approx_date=data.approx_date,
                place_text=data.place_text,
                source_id=data.source_id,
                people=[
                    MediaPerson(
                        person_id=person.person_id,
                        is_primary=person.is_primary,
                        notes=person.notes,
                    )
                    for person in data.people
                ],
            )
        )

    async def for_person(self, person_id: uuid.UUID) -> list[Media]:
        stmt = (
            select(Media)
            .join(MediaPerson, MediaPerson.media_id == Media.id)
            .where(MediaPerson.person_id == person_id)
            .options(*self.load_options)
            .order_by(MediaPerson.is_primary.desc(), Media.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
