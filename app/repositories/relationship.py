import uuid

from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from app.domain.enums import RelationshipStatus
from app.models import Relationship
from app.repositories.base import SQLRepository
from app.schemas.relationship import RelationshipCreate


class RelationshipRepository(SQLRepository[Relationship]):
    model = Relationship
    # The admin relationship page renders both endpoints, so eager-load them
    # instead of triggering async lazy loads during template rendering.
    load_options = (
        selectinload(Relationship.person_a),
        selectinload(Relationship.person_b),
    )

    async def create(
        self,
        data: RelationshipCreate,
        *,
        entity_id: uuid.UUID | None = None,
        commit: bool = True,
    ) -> Relationship:
        if data.person_a_id == data.person_b_id:
            raise ValueError("A relationship must connect two different persons")
        id_kwargs = {"id": entity_id} if entity_id is not None else {}
        return await self._create(
            lambda: Relationship(
                **id_kwargs,
                person_a_id=data.person_a_id,
                person_b_id=data.person_b_id,
                relationship_type=data.relationship_type,
                status=data.status,
                source_id=data.source_id,
                valid_from=data.valid_from,
                valid_to=data.valid_to,
            ),
            commit=commit,
        )

    async def set_status(
        self, relationship_id: uuid.UUID, status: RelationshipStatus
    ) -> Relationship | None:
        """Update only the trust status of an existing relationship.

        Returns ``None`` when the id does not exist. Endpoints, relationship
        type, provenance (``source_id``), validity dates and timestamps are
        left untouched; the row is reloaded after commit.
        """
        relationship = await self.get(relationship_id)
        if relationship is None:
            return None
        relationship.status = status
        await self._session.commit()
        return await self.get(relationship_id)

    async def for_person(self, person_id: uuid.UUID) -> list[Relationship]:
        """Relationships for one person that may still reach an answer.

        Rejected relationships are excluded at the query boundary. Pending
        relationships remain for the context layer to withhold explicitly.
        """
        stmt = (
            select(Relationship)
            .where(
                or_(
                    Relationship.person_a_id == person_id,
                    Relationship.person_b_id == person_id,
                ),
                Relationship.status != RelationshipStatus.rejected,
            )
            .options(
                selectinload(Relationship.person_a),
                selectinload(Relationship.person_b),
            )
            .order_by(Relationship.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
