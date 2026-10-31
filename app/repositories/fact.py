import uuid

from sqlalchemy import select

from app.domain.enums import FactStatus
from app.models import Fact
from app.repositories.base import SQLRepository
from app.schemas.fact import FactCreate


class FactRepository(SQLRepository[Fact]):
    model = Fact

    async def create(
        self, data: FactCreate, *, entity_id: uuid.UUID | None = None
    ) -> Fact:
        id_kwargs = {"id": entity_id} if entity_id is not None else {}
        return await self._create(
            lambda: Fact(
                **id_kwargs,
                subject_person_id=data.subject_person_id,
                fact_type=data.fact_type,
                value_text=data.value_text,
                value_date=data.value_date,
                value_number=data.value_number,
                related_person_id=data.related_person_id,
                place_text=data.place_text,
                valid_from=data.valid_from,
                valid_to=data.valid_to,
                status=data.status,
                source_id=data.source_id,
                notes=data.notes,
            )
        )

    async def for_subject_person(self, person_id: uuid.UUID) -> list[Fact]:
        stmt = (
            select(Fact)
            .where(Fact.subject_person_id == person_id)
            .order_by(Fact.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def list_verified(self) -> list[Fact]:
        """All verified facts, the only authority a contradiction signal may
        reference. Lower-status facts (probable/pending/rejected) are not
        authoritative and are never compared against claims."""
        stmt = (
            select(Fact)
            .where(Fact.status == FactStatus.verified)
            .order_by(Fact.created_at, Fact.id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
