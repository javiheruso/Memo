import uuid
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class SQLRepository(Generic[ModelT]):
    """Minimal async CRUD base shared by the Phase 1 repositories."""

    model: type[ModelT]
    load_options: tuple[Any, ...] = ()

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        stmt = (
            select(self.model)
            .where(self.model.id == entity_id)
            .options(*self.load_options)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = (
            select(self.model)
            .options(*self.load_options)
            .order_by(self.model.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def _create(
        self, build: Callable[[], ModelT], *, commit: bool = True
    ) -> ModelT:
        model = build()
        self._session.add(model)
        if commit:
            await self._session.commit()
        else:
            await self._session.flush()
        fresh = await self.get(model.id)
        if fresh is None:
            raise RuntimeError(f"{self.model.__name__} disappeared after write")
        return fresh
