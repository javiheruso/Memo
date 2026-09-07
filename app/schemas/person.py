import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PersonAliasCreate(BaseModel):
    alias: str = Field(min_length=1, max_length=255)
    alias_type: str = Field(default="nickname", max_length=50)


class PersonCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=255)
    last_name: str | None = Field(default=None, max_length=255)
    display_name: str | None = Field(default=None, max_length=255)
    birth_date: date | None = None
    death_date: date | None = None
    is_deceased: bool = False
    description: str | None = None
    aliases: list[PersonAliasCreate] = Field(default_factory=list)


class PersonAliasRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    alias: str
    alias_type: str


class PersonPrimaryMediaUpdate(BaseModel):
    """Set (or clear, with ``null``) a person's primary media row."""

    media_id: uuid.UUID | None = None


class PersonRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    first_name: str
    last_name: str | None
    display_name: str
    birth_date: date | None
    death_date: date | None
    is_deceased: bool
    description: str | None
    primary_media_id: uuid.UUID | None
    created_at: datetime
    aliases: list[PersonAliasRead] = Field(default_factory=list)
