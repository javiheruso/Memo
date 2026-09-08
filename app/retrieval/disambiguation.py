"""Multi-turn disambiguation of an ambiguous person query.

When the previous assistant turn surfaced several candidates for a person
("¿A cuál te refieres?"), a follow-up such as "el panadero", "el segundo" or
"Paco García" must pick one of the pending candidates without re-running
entity resolution over the whole corpus. Each strategy below is
accent-folding (accent-insensitive) and only ever returns a person on a
confident match; otherwise it returns ``None`` so the caller re-asks or falls
back to the normal routing.
"""

import re
import uuid

from app.models import Person
from app.repositories.person import PersonRepository, _fold

_ARTICLE_STOPWORDS = frozenset(
    {"el", "la", "los", "las", "un", "una", "unos", "unas"}
)

# Accent-folded ordinal tokens -> index into the pending candidate list.
# "último"/"última" fold to "ultimo"/"ultima" via _fold.
_ORDINAL_TO_INDEX = {
    "primero": 0,
    "primera": 0,
    "1": 0,
    "segundo": 1,
    "segunda": 1,
    "2": 1,
    "tercero": 2,
    "tercera": 2,
    "3": 2,
    "ultimo": -1,
    "ultima": -1,
}
_ORDINAL_STOPWORDS = frozenset(_ORDINAL_TO_INDEX)

# Words that carry no descriptive signal when scanning a description: the
# ordinal words are excluded so "el segundo" can never match a description.
_CONTENT_STOPWORDS = _ARTICLE_STOPWORDS | _ORDINAL_STOPWORDS

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _whole_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match ``phrase`` as a whole word/phrase inside already-folded text."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])")


def _name_span(surface: str) -> tuple[int, int]:
    """Rank a folded name surface by (word count, length) for specificity."""
    return (len(surface.split()), len(surface))


def _content_tokens(text: str) -> set[str]:
    """Fold ``text`` and return its descriptive tokens minus stopwords."""
    return {
        token
        for token in _TOKEN_RE.findall(_fold(text))
        if token not in _CONTENT_STOPWORDS
    }


class DisambiguationResolver:
    """Pick the intended person among the pending ambiguity candidates.

    Strategies run in order and the first confident match wins:
    ordinal ("el segundo"), a whole name/alias/display phrase ("Paco García"),
    then a description keyword ("el panadero" -> a candidate whose description
    mentions "panadero"). Zero or several matching candidates never yield a
    pick, so the pipeline re-asks instead of guessing arbitrarily.
    """

    def __init__(self, person_repository: PersonRepository) -> None:
        self._person_repository = person_repository

    async def resolve(
        self, text: str, candidate_ids: list[uuid.UUID]
    ) -> Person | None:
        people: list[Person] = []
        for person_id in candidate_ids:
            person = await self._person_repository.get(person_id)
            if person is not None:
                people.append(person)
        if not people:
            return None

        ordinal = self._select_by_ordinal(text, people)
        if ordinal is not None:
            return ordinal
        name = self._select_by_name(text, people)
        if name is not None:
            return name
        return self._select_by_description(text, people)

    @staticmethod
    def _select_by_ordinal(text: str, people: list[Person]) -> Person | None:
        """Map "el primero"/"la segunda"/"1" onto the candidate order."""
        folded = _fold(text)
        tokens = {
            token
            for token in _TOKEN_RE.findall(folded)
            if token not in _ARTICLE_STOPWORDS
        }
        if len(tokens) != 1:
            return None
        index = _ORDINAL_TO_INDEX.get(next(iter(tokens)))
        if index is None:
            return None
        try:
            return people[index]
        except IndexError:
            return None

    @classmethod
    def _select_by_name(cls, text: str, people: list[Person]) -> Person | None:
        """Pick the candidate whose name/alias/display phrase the text names.

        The most specific phrase wins: "Paco García" beats the shared alias
        "Paco", so it resolves to the García person. A tie at the same
        specificity (bare "paco" matches both) yields no pick.
        """
        folded_text = _fold(text)
        matches: list[tuple[Person, str]] = []
        for person in people:
            surface = cls._best_name_surface(folded_text, person)
            if surface is not None:
                matches.append((person, surface))
        if not matches:
            return None

        matches.sort(key=lambda item: _name_span(item[1]), reverse=True)
        top_surface = matches[0][1]
        if len(matches) == 1:
            return matches[0][0]
        second_surface = matches[1][1]
        if _name_span(top_surface) > _name_span(second_surface):
            return matches[0][0]
        return None

    @classmethod
    def _select_by_description(
        cls, text: str, people: list[Person]
    ) -> Person | None:
        """Pick the candidate whose description carries a follow-up keyword.

        Only the descriptive content tokens of the follow-up are used (folded,
        minus articles and ordinal words), and any token that is part of a
        candidate name/alias is ignored: a bare "paco" that ties in the name
        strategy must re-ask, not resolve through a description that happens to
        mention the alias. Exactly one candidate whose folded description
        contains any of the remaining keywords wins.
        """
        keywords = _content_tokens(text)
        if not keywords:
            return None
        keywords = keywords - cls._candidate_name_tokens(people)
        if not keywords:
            return None
        matched: list[Person] = []
        for person in people:
            folded_description = _fold(person.description)
            if folded_description and any(
                _whole_phrase_pattern(keyword).search(folded_description)
                for keyword in keywords
            ):
                matched.append(person)
        if len(matched) == 1:
            return matched[0]
        return None

    @staticmethod
    def _candidate_name_tokens(people: list[Person]) -> set[str]:
        """Fold the candidate name/alias surfaces into their word tokens."""
        tokens: set[str] = set()
        for person in people:
            for value in (
                person.first_name,
                person.last_name,
                person.display_name,
            ):
                tokens.update(_TOKEN_RE.findall(_fold(value)))
            for alias in person.aliases:
                tokens.update(_TOKEN_RE.findall(_fold(alias.alias)))
        return tokens

    @staticmethod
    def _best_name_surface(folded_text: str, person: Person) -> str | None:
        """Longest whole-phrase name surface of ``person`` found in the text."""
        best: str | None = None
        for value in (person.first_name, person.last_name, person.display_name):
            surface = _fold(value)
            if (
                surface
                and _whole_phrase_pattern(surface).search(folded_text)
                and (best is None or _name_span(surface) > _name_span(best))
            ):
                best = surface
        for alias in person.aliases:
            surface = _fold(alias.alias)
            if (
                surface
                and _whole_phrase_pattern(surface).search(folded_text)
                and (best is None or _name_span(surface) > _name_span(best))
            ):
                best = surface
        return best
