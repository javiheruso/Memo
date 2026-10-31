"""Dialect-portable SQLAlchemy column types."""

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """A ``TIMESTAMP WITH TIME ZONE`` column that always yields aware UTC.

    PostgreSQL round-trips ``timestamptz`` values with their timezone
    information. SQLite has no timezone-aware datetime type: it stores the
    value as text and reads it back naive, even for ``DateTime(timezone=True)``.
    The application treats every stored timestamp as UTC, so this type restores
    the UTC marker on read and normalizes values to UTC on write. Model
    datetimes are therefore comparable and serializable on both backends.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
