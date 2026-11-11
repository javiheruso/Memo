"""Real LLM responder behind the provider registry, fail-closed.

Calls any OpenAI-compatible ``/chat/completions`` endpoint with a prompt built
exclusively from the ``AnswerContext`` (no database handle or tool is ever
exposed), parses the model's JSON answer, validates every returned source id
against the context's allowed set, and on any invalid output or
provider/network problem falls back to the deterministic responder for that
turn. A model mistake can therefore degrade a turn to the conservative text
responder but can never inject an invented source into ``response_sources``.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.config import Settings
from app.domain.conversation_context import ConversationStyle, ConversationTurn
from app.domain.enums import Intent
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.llm.prompt_builder import allowed_sources, build_prompt
from app.policies.positive_topics import (
    context_topic_grounding,
    positive_topic_violation,
)
from app.repositories.conversation import ResponseSourceRef
from app.retrieval.context_builder import AnswerContext

logger = logging.getLogger(__name__)


class InvalidModelOutput(ValueError):
    """Raised when model output cannot be turned into a trusted response."""


def _log_fallback(exc: Exception) -> None:
    """Log a safe fallback signal without prompts, bodies, or credentials.

    Only the failure category/class and, when the provider answered with an
    HTTP error, the status code are recorded. Exception messages are never
    logged because they may embed model output, provider response text, or
    request details.
    """
    category = (
        "invalid_model_output"
        if isinstance(exc, InvalidModelOutput)
        else "http_status_error"
        if isinstance(exc, httpx.HTTPStatusError)
        else type(exc).__name__
    )
    status = (
        exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    )
    if status is None:
        logger.warning(
            "LLM responder fell back to deterministic (category=%s)",
            category,
        )
    else:
        logger.warning(
            "LLM responder fell back to deterministic (category=%s, http_status=%s)",
            category,
            status,
        )


class OpenAICompatibleResponder(Responder):
    """Responder backed by any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
        fallback: Responder | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for the OpenAI-compatible responder")
        if not model:
            raise ValueError("model is required for the OpenAI-compatible responder")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = client
        self._fallback = fallback or DeterministicResponder()

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleResponder:
        return cls(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    async def respond(
        self,
        context: AnswerContext,
        *,
        message: str | None = None,
        history: Sequence[ConversationTurn] = (),
        style: ConversationStyle | None = None,
    ) -> Response:
        try:
            prompt = build_prompt(
                context, message=message, history=history, style=style
            )
            messages = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            raw = await self._call_model(messages)
            response = self._validate_response(raw, context)
            has_stored_data = bool(allowed_sources(context))
            if has_stored_data and not response.sources:
                raise InvalidModelOutput(
                    "model cited no context source although stored data exists"
                )
            if (
                style is not None
                and context.sensitive_topic_requested is None
                and positive_topic_violation(
                    answer=response.text,
                    message=message or "",
                    topics=style.positive_topics,
                    offer_allowed=style.offer_positive_topics,
                    grounded_text=context_topic_grounding(context),
                )
            ):
                raise InvalidModelOutput(
                    "model broke the bounded positive-topic suggestion policy"
                )
            if (
                not has_stored_data
                and context.intent in (Intent.general_chat, Intent.unknown)
                and _NO_STORED_DATA_RECORD_CLAIM.search(response.text)
            ):
                raise InvalidModelOutput(
                    "model claimed stored records although no stored data exists"
                )
            if (
                not has_stored_data
                and _similar_to_previous(response.text, history)
            ):
                raise InvalidModelOutput(
                    "model repeated the previous answer with no stored data"
                )
            if context.accepted_offer and _repeats_offer_or_question(
                response.text, history
            ):
                raise InvalidModelOutput(
                    "model repeated the offer or asked again after acceptance"
                )
            if (
                context.intent == Intent.memory_query
                and not context.memories
                and (
                    response.text.strip().endswith(("?", "¿"))
                    or _MEMORY_NO_DATA_OFFER.search(response.text)
                )
            ):
                raise InvalidModelOutput(
                    "model offered or asked a question with no stored memory"
                )
            if (
                context.intent == Intent.memory_query
                and context.memories
                and not _memory_answer_is_grounded(
                    response.text, context.memories
                )
            ):
                raise InvalidModelOutput(
                    "model memory answer added details absent from the source"
                )
            return response
        except (httpx.HTTPError, InvalidModelOutput) as exc:
            # Fail closed: the deterministic responder handles this turn.
            _log_fallback(exc)
            return await self._fallback.respond(
                context, message=message, history=history, style=style
            )

    async def _call_model(self, messages: list[dict[str, str]]) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            # Ask OpenAI-compatible servers for a JSON object so the model does
            # not wrap or pad the required ``{answer, sources}`` contract. This
            # only shapes output; every field is still validated fail-closed.
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._client is not None:
            response = await self._client.post(url, headers=headers, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InvalidModelOutput(
                "model response did not match the chat completions shape"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise InvalidModelOutput("model returned an empty answer")
        return content

    def _validate_response(self, raw: str, context: AnswerContext) -> Response:
        payload = _parse_json_payload(raw)
        if payload is None:
            raise InvalidModelOutput("model output is not a JSON object")

        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise InvalidModelOutput("model output has no text answer")

        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            raise InvalidModelOutput("model output 'sources' is not a list")

        allowed = allowed_sources(context)
        sources: list[ResponseSourceRef] = []
        for entry in raw_sources:
            if not isinstance(entry, dict):
                raise InvalidModelOutput("model output contains a malformed source")
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            if not isinstance(entity_type, str) or not isinstance(entity_id, str):
                raise InvalidModelOutput(
                    "model output source is missing entity_type or entity_id"
                )
            try:
                parsed_id = uuid.UUID(entity_id)
            except (ValueError, TypeError) as exc:
                raise InvalidModelOutput(
                    f"model output has an invalid source id: {entity_id!r}"
                ) from exc
            usage = allowed.get((entity_type, str(parsed_id)))
            if usage is None:
                raise InvalidModelOutput(
                    "model output references a source not in the context: "
                    f"{entity_type}/{entity_id}"
                )
            sources.append(
                ResponseSourceRef(
                    entity_type=entity_type, entity_id=parsed_id, usage_type=usage
                )
            )
        return Response(text=answer.strip(), sources=sources)


_GROUNDING_TOKEN = re.compile(r"[a-záéíóúñü0-9]+")
_GROUNDING_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "algo",
        "como",
        "con",
        "de",
        "del",
        "el",
        "ella",
        "en",
        "era",
        "es",
        "esta",
        "este",
        "fue",
        "ha",
        "la",
        "las",
        "lo",
        "los",
        "más",
        "mas",
        "me",
        "mi",
        "mis",
        "que",
        "qué",
        "se",
        "sin",
        "sobre",
        "son",
        "su",
        "sus",
        "te",
        "tu",
        "tus",
        "un",
        "una",
        "y",
        "ya",
    }
)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _GROUNDING_TOKEN.findall(text.lower())
        if len(token) >= 2 and token not in _GROUNDING_STOPWORDS
    }


def _memory_answer_is_grounded(answer: str, memories) -> bool:
    """True when the answer adds no content absent from the memory texts.

    Memory narratives are exact stored text. A model that embellishes them
    (for example, inventing a detail that is not in the narrative) is rejected
    and the deterministic renderer answers instead.
    """
    allowed: set[str] = set()
    for memory in memories:
        allowed |= _content_tokens(f"{memory.title or ''} {memory.narrative}")
    return _content_tokens(answer) <= allowed


def _normalize_reply(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?¿¡")


def _previous_assistant_text(
    history: Sequence[ConversationTurn],
) -> str | None:
    previous = next(
        (turn for turn in reversed(history) if turn.role == "assistant"),
        None,
    )
    return previous.text if previous is not None else None


def _similar_to_previous(answer: str, history: Sequence[ConversationTurn]) -> bool:
    previous = _previous_assistant_text(history)
    if previous is None:
        return False
    ratio = difflib.SequenceMatcher(
        None,
        _normalize_reply(answer),
        _normalize_reply(previous),
    ).ratio()
    return ratio >= 0.9


def _repeats_offer_or_question(
    answer: str, history: Sequence[ConversationTurn]
) -> bool:
    """Detect the acceptance-loop failure in a model answer.

    After the person accepts, a new question or a near-copy of the previous
    assistant offer is not a continuation. The responder falls back to the
    deterministic, context-grounded answer instead.
    """
    stripped = answer.strip()
    if stripped.endswith(("?", "¿")):
        return True
    return _similar_to_previous(answer, history)


_MEMORY_NO_DATA_OFFER = re.compile(
    r"\b(?:podemos|puedo|quieres|quiere|te\s+gustar[íi]a|te\s+apetece|"
    r"te\s+interesa|si\s+quieres|te\s+cuento|hablamos\s+de)\b",
    re.IGNORECASE,
)

# A spontaneous turn with no stored data must never imply that Memo holds the
# person's projects, memories, records, or archives. This is deliberately
# generic: it protects the trust boundary without naming any profile topic.
_NO_STORED_DATA_RECORD_CLAIM = re.compile(
    r"\b(?:proyect\w*|recuerd\w*|memori\w*|registr\w*|archiv\w*|guardad\w*)\b",
    re.IGNORECASE,
)


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    """Parse a model answer that may carry markdown fences or prose around it."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            candidate = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return candidate if isinstance(candidate, dict) else None
    return None
