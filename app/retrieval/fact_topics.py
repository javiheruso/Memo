"""Deterministic relevance rules for conversational fact questions.

A fact query such as "¿Qué le gusta a Elena?" must not dump every stored fact
about the person. This module maps the question to a small, explicit topic
family and decides whether a stored fact belongs to that family.

The rules are intentionally shallow and auditable: the fact ``fact_type`` is
tokenized (``comida_favorita`` -> ``comida``, ``favorita``) and intersected
with the family's accepted tokens. There is no semantic model, embedding, or
free-text guessing, so an unrelated fact can never be surfaced just because it
belongs to the same person. Questions that ask for a person's data in general
("¿Qué datos tienes de Elena?") use the ``general`` family, which is the only
case where all stored facts are considered relevant.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum


class FactTopic(StrEnum):
    """Question family used to select contextually relevant facts."""

    preference = "preference"
    dislike = "dislike"
    hobby = "hobby"
    food = "food"
    drink = "drink"
    music = "music"
    place = "place"
    occupation = "occupation"
    habit = "habit"
    belief = "belief"
    value = "value"
    conversation = "conversation"
    family = "family"
    general = "general"


_TOKEN = re.compile(r"[a-z0-9]+")


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).lower()
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(_fold(value)))


# Ordered from specific to broad: the first matching family wins, so a
# question about food or music never falls back to the generic preference set.
_QUERY_RULES: tuple[tuple[FactTopic, re.Pattern[str]], ...] = (
    (
        FactTopic.dislike,
        re.compile(
            r"\b(?:disgust\w*|molest\w*|no\s+le\s+gust\w*|"
            r"dislike\w*|hate\w*|bother\w*)"
        ),
    ),
    (
        FactTopic.food,
        re.compile(
            r"\b(?:comid\w*|comer\w*|plat\w*|gastronom\w*|"
            r"food\w*|meal\w*|dish\w*)"
        ),
    ),
    (FactTopic.drink, re.compile(r"\b(?:bebid\w*|beber\w*|drink\w*)")),
    (
        FactTopic.music,
        re.compile(r"\b(?:music\w*|cancion\w*|song\w*|songs?)"),
    ),
    (
        FactTopic.hobby,
        re.compile(
            r"\b(?:aficion\w*|hobb\w*|interes\w*|"
            r"tiempo\s+libre|free\s+time)"
        ),
    ),
    (
        FactTopic.place,
        re.compile(
            r"\b(?:donde|nacio\w*|viv\w*|residenc\w*|lugar\w*|"
            r"ciudad\w*|pueblo\w*|domicili\w*|direccion\w*|"
            r"where|born|lives?|lived|hometown|residence|place)"
        ),
    ),
    (
        FactTopic.occupation,
        re.compile(
            r"\b(?:oficio\w*|profesion\w*|trabaj\w*|dedic\w*|"
            r"ocupacion\w*|occupation\w*|job\w*|work\w*|profession\w*)"
        ),
    ),
    (
        FactTopic.habit,
        re.compile(r"\b(?:costumbre\w*|rutina\w*|habito\w*|habit\w*|routine\w*)"),
    ),
    (
        FactTopic.conversation,
        re.compile(
            r"\b(?:conversacion\w*|conversational\w*|trato|"
            r"habl\w*|conversation\w*)"
        ),
    ),
    (
        FactTopic.belief,
        re.compile(
            r"\b(?:cree|creencia\w*|opina\w*|opinion\w*|piensa\w*|"
            r"belief\w*|believe\w*|think\w*)"
        ),
    ),
    (FactTopic.value, re.compile(r"\b(?:valor\w*|value\w*|values?)")),
    (
        FactTopic.family,
        re.compile(
            r"\b(?:familia\w*|parentesco\w*|familiar\w*|"
            r"family|relatives?)"
        ),
    ),
    (
        FactTopic.preference,
        re.compile(
            r"\b(?:gust\w*|encant\w*|prefier\w*|preferenci\w*|"
            r"favorit\w*|like\w*|love\w*|enjoy\w*|prefer\w*)"
        ),
    ),
    (
        FactTopic.general,
        re.compile(
            r"\b(?:datos?|informacion\w*|sabes|conoces|"
            r"know\w*|information\w*|facts?)"
        ),
    ),
)

# Accepted ``fact_type`` tokens per family, Spanish importer vocabulary plus
# the English seed vocabulary. A missing family means "all facts".
_FACT_TYPES: dict[FactTopic, frozenset[str]] = {
    FactTopic.preference: frozenset(
        {
            "aficion",
            "aficiones",
            "hobby",
            "hobbies",
            "interes",
            "intereses",
            "gusto",
            "gustos",
            "preferencia",
            "preferencias",
            "preferencia_conversacional",
            "favorit",
            "favorito",
            "favorita",
            "tema",
            "temas",
            "musica",
            "comida",
            "bebida",
            "like",
            "likes",
            "favorite",
            "interest",
        }
    ),
    FactTopic.dislike: frozenset(
        {"desagrado", "desagrados", "disgusto", "rechazo", "molestia", "dislike"}
    ),
    FactTopic.hobby: frozenset(
        {
            "aficion",
            "aficiones",
            "hobby",
            "hobbies",
            "interes",
            "intereses",
            "tiempo",
            "libre",
            "interest",
        }
    ),
    FactTopic.food: frozenset(
        {
            "comida",
            "comidas",
            "plato",
            "platos",
            "gastronomia",
            "cocina",
            "food",
            "meal",
            "dish",
        }
    ),
    FactTopic.drink: frozenset({"bebida", "bebidas", "drink"}),
    FactTopic.music: frozenset(
        {"musica", "musical", "cancion", "canciones", "music", "song"}
    ),
    FactTopic.place: frozenset(
        {
            "lugar",
            "lugares",
            "residencia",
            "historica",
            "infancia",
            "vivienda",
            "domicilio",
            "direccion",
            "ciudad",
            "pueblo",
            "residence",
            "place",
            "born",
            "hometown",
            "move",
        }
    ),
    FactTopic.occupation: frozenset(
        {
            "oficio",
            "profesion",
            "profesional",
            "trabajo",
            "ocupacion",
            "laboral",
            "labor",
            "occupation",
            "job",
            "work",
            "profession",
        }
    ),
    FactTopic.habit: frozenset(
        {"costumbre", "costumbres", "rutina", "habito", "habitos", "habit", "routine"}
    ),
    FactTopic.belief: frozenset(
        {
            "creencia",
            "creencias",
            "opinion",
            "opiniones",
            "ideologia",
            "belief",
            "believes",
        }
    ),
    FactTopic.value: frozenset({"valor", "valores", "value", "values"}),
    FactTopic.conversation: frozenset(
        {
            "preferencia_conversacional",
            "conversacional",
            "conversacion",
            "trato",
            "hablar",
            "conversation",
        }
    ),
    FactTopic.family: frozenset(
        {
            "parentesco",
            "familia",
            "familiar",
            "familiares",
            "rol_familiar",
            "rol",
            "family",
            "kinship",
            "relationship",
        }
    ),
}


def classify_fact_question(text: str) -> FactTopic | None:
    """Return the question family for a fact question, or ``None``.

    The caller still has to confirm a question shape and a subject; this
    function only classifies the topic vocabulary.
    """
    folded = _fold(text)
    for topic, pattern in _QUERY_RULES:
        if pattern.search(folded):
            return topic
    return None


def fact_matches_topic(
    fact_type: str | None,
    *,
    topic: FactTopic,
) -> bool:
    """True when a stored fact belongs to the question's family."""
    if topic is FactTopic.general:
        return True
    accepted = _FACT_TYPES.get(topic)
    if not accepted:
        return True
    if not fact_type:
        return False
    tokens = _tokens(fact_type.replace("_", " "))
    return bool(tokens & accepted)


__all__ = ["FactTopic", "classify_fact_question", "fact_matches_topic"]
