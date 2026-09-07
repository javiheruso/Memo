from app.models import Source
from app.repositories.base import SQLRepository
from app.schemas.source import SourceCreate


class SourceRepository(SQLRepository[Source]):
    model = Source

    async def create(self, data: SourceCreate) -> Source:
        return await self._create(
            lambda: Source(
                source_type=data.source_type,
                person_id=data.person_id,
                description=data.description,
                external_reference=data.external_reference,
            )
        )
