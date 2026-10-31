import uuid

from app.models import Source
from app.repositories.base import SQLRepository
from app.schemas.source import SourceCreate


class SourceRepository(SQLRepository[Source]):
    model = Source

    async def create(
        self,
        data: SourceCreate,
        *,
        entity_id: uuid.UUID | None = None,
        commit: bool = True,
    ) -> Source:
        id_kwargs = {"id": entity_id} if entity_id is not None else {}
        return await self._create(
            lambda: Source(
                **id_kwargs,
                source_type=data.source_type,
                person_id=data.person_id,
                description=data.description,
                external_reference=data.external_reference,
            ),
            commit=commit,
        )
