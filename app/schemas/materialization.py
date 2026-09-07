import uuid
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from app.domain.enums import FactStatus, MemoryStatus, SensitivityLevel
from app.schemas.memory import MemoryPersonCreate


class FactMaterializeRequest(BaseModel):
    """Fields an admin supplies to materialize a fact from a reviewed source.

    ``value_text`` is optional: when omitted the service defaults it to the
    source's canonical text (a claim group's ``canonical_text`` or a knowledge
    gap's ``canonical_reference``). ``status`` defaults to ``pending`` so an
    explicit materialization never silently verifies a claim — the admin must
    choose ``verified`` consciously.
    """

    subject_person_id: uuid.UUID
    fact_type: str = Field(max_length=50)
    value_text: str | None = None
    value_date: date | None = None
    value_number: Decimal | None = None
    related_person_id: uuid.UUID | None = None
    place_text: str | None = Field(default=None, max_length=500)
    valid_from: date | None = None
    valid_to: date | None = None
    status: FactStatus = FactStatus.pending
    source_id: uuid.UUID
    notes: str | None = None


class MemoryMaterializeRequest(BaseModel):
    """Fields an admin supplies to materialize a memory from a reviewed source.

    ``title`` and ``narrative`` default to the source's canonical text when
    omitted. ``memory_status`` defaults to ``candidate`` and the sensitivity
    level to ``low``, keeping the trust-model defaults of the manual memory
    creation path.
    """

    title: str | None = Field(default=None, max_length=255)
    narrative: str | None = None
    approx_start_date: date | None = None
    approx_end_date: date | None = None
    approx_period_text: str | None = Field(default=None, max_length=255)
    place_text: str | None = Field(default=None, max_length=500)
    memory_status: MemoryStatus = MemoryStatus.candidate
    sensitivity_level: SensitivityLevel = SensitivityLevel.low
    source_id: uuid.UUID
    people: list[MemoryPersonCreate] = Field(default_factory=list)
