import uuid
from dataclasses import dataclass
from enum import Enum

from app.repositories.person import PersonRepository


class ResolutionKind(str, Enum):
    resolved = "resolved"
    ambiguous = "ambiguous"
    not_found = "not_found"


@dataclass(frozen=True)
class PersonCandidate:
    id: uuid.UUID
    display_name: str


@dataclass(frozen=True)
class Resolution:
    kind: ResolutionKind
    term: str
    candidates: list[PersonCandidate]
    person: PersonCandidate | None = None

    @classmethod
    def from_people(cls, term: str, people: list) -> "Resolution":
        candidates = [
            PersonCandidate(id=person.id, display_name=person.display_name)
            for person in people
        ]
        if not candidates:
            return cls(kind=ResolutionKind.not_found, term=term, candidates=[])
        if len(candidates) == 1:
            return cls(
                kind=ResolutionKind.resolved,
                term=term,
                candidates=candidates,
                person=candidates[0],
            )
        return cls(kind=ResolutionKind.ambiguous, term=term, candidates=candidates)


class EntityResolver:
    """Resolve a person by name or alias with explicit ambiguity handling.

    When several people match, the resolution is ``ambiguous`` and lists all
    candidates: the pipeline never picks one arbitrarily (dossier: "No elegir
    arbitrariamente si existen dos candidatos plausibles").
    """

    def __init__(self, person_repository: PersonRepository) -> None:
        self._person_repository = person_repository

    async def resolve(self, term: str) -> Resolution:
        people = await self._person_repository.search_by_name(term)
        return Resolution.from_people(term, people)
