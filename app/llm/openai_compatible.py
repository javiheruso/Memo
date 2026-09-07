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

import json
import re
import uuid
from typing import Any

import httpx

from app.core.config import Settings
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.llm.prompt_builder import allowed_sources, build_prompt
from app.repositories.conversation import ResponseSourceRef
from app.retrieval.context_builder import AnswerContext


class InvalidModelOutput(ValueError):
    """Raised when model output cannot be turned into a trusted response."""


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

    async def respond(self, context: AnswerContext) -> Response:
        try:
            prompt = build_prompt(context)
            messages = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            raw = await self._call_model(messages)
            return self._validate_response(raw, context)
        except (httpx.HTTPError, InvalidModelOutput):
            # Fail closed: the deterministic responder handles this turn.
            return await self._fallback.respond(context)

    async def _call_model(self, messages: list[dict[str, str]]) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
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
