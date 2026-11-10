"""Strict contract for an optional low-confidence routing suggestion.

The interpreter may only propose a route. It never returns an answer, a source,
a person id, a confidence score or any write instruction, and the model output
is rejected as a whole when it contains anything outside this schema. Every
field is verified against Memo's own data and policies before it can influence a
turn, so a malformed or over-eager suggestion simply falls back to the existing
deterministic behavior.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.retrieval.fact_topics import FactTopic
from app.retrieval.intent_router import FOLLOW_UP_TOPIC_LABELS

_SUGGESTED_INTENTS = Literal["person_query", "fact_query"]
_MAX_TERM_CHARS = 120
_MAX_FOLLOW_UP_TOPIC_CHARS = 64


class InterpreterSuggestion(BaseModel):
    """A bounded route proposal produced by the optional interpreter.

    ``extra="forbid"`` is intentional: model output that includes an answer,
    sources, person ids, confidence, or any other field is invalid and is
    discarded instead of being partially trusted.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: _SUGGESTED_INTENTS | None = None
    term: str | None = Field(default=None, max_length=_MAX_TERM_CHARS)
    fact_topic: FactTopic | None = None
    follow_up_topic: str | None = Field(
        default=None, max_length=_MAX_FOLLOW_UP_TOPIC_CHARS
    )

    @field_validator("term", "follow_up_topic")
    @classmethod
    def _single_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(ord(char) < 32 for char in value):
            raise ValueError("interpreter text fields must be a single line")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> InterpreterSuggestion:
        if self.intent is None:
            if any(
                value is not None
                for value in (self.term, self.fact_topic, self.follow_up_topic)
            ):
                raise ValueError("a declined suggestion must not carry route fields")
            return self

        if not self.term:
            raise ValueError("term is required for a route suggestion")

        if self.intent == "person_query":
            if self.fact_topic is not None or self.follow_up_topic is not None:
                raise ValueError("person_query suggestions cannot carry fact fields")
            return self

        if self.fact_topic is None:
            raise ValueError("fact_topic is required for a fact_query suggestion")
        if (
            self.follow_up_topic is not None
            and self.follow_up_topic not in FOLLOW_UP_TOPIC_LABELS.values()
        ):
            raise ValueError(
                "follow_up_topic must be one of the known natural topic labels"
            )
        return self


__all__ = ["InterpreterSuggestion"]
