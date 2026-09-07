from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.config import settings
from app.db.base import Base
from app.main import app
from tests.fixtures.universe import SeedUniverse, seed_universe


@pytest.fixture(scope="session")
async def postgres_engine() -> AsyncEngine | None:
    """Session-scoped PostgreSQL engine when MEMO_TEST_DATABASE_URL is set.

    Returns ``None`` when the profile is not requested, so the default suite
    keeps running on fast in-memory SQLite.
    """
    if not settings.test_database_url:
        yield None
        return
    engine = create_async_engine(settings.test_database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_engine(postgres_engine: AsyncEngine | None) -> AsyncGenerator[AsyncEngine, None]:
    if postgres_engine is not None:
        await _truncate_all(postgres_engine)
        yield postgres_engine
        return

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(_enable_foreign_keys)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


async def _truncate_all(engine: AsyncEngine) -> None:
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.exec_driver_sql(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE")


def _enable_foreign_keys(connection) -> None:
    connection.exec_driver_sql("PRAGMA foreign_keys = ON")


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
async def seed(db_session: AsyncSession) -> SeedUniverse:
    return await seed_universe(db_session)


@pytest.fixture
async def app_client(db_engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
