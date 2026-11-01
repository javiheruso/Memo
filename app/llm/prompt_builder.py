"""Build chat-completion prompts from an ``AnswerContext``.

The prompt is the only thing a model ever sees: a flat, epistemically-tagged
rendering of the context plus the exact list of entity ids it may reference.
The model never receives a database handle, repository, or tool, so it cannot
query storage directly. Trust-model policy decisions (what is surfaced, what
epistemic kind each item has, how ambiguity is resolved) are made upstream in
code and only rendered here; the prompt repeats them as instructions, but
``OpenAICompatibleResponder`` enforces them again by validating every returned
source id against ``allowed_sources``.
"""

from dataclasses import dataclass

from app.retrieval.context_builder import AnswerContext

_SYSTEM = """You are the conversational companion of an older adult. You know nothing except the CONTEXT data the user supplies.

Rules:
- Never invent facts, people, relationships, memories, or events.
- Never claim stored knowledge that is not present in CONTEXT.
- Only reference entity ids listed under "Allowed source ids" in CONTEXT.
- Present probable or uncertain data as uncertain. Never present a memory or a probable item as a verified fact.
- Never present a planned or confirmed event as already completed.
- For a fact_query, answer from the Facts list only. Do not substitute a relation, memory, event, or media item for a missing fact. If no fact matches, say there is no stored data about that.
- When a fact lists a stored source type, you may mention it as provenance. Never invent a source.
- Base any mention of today's date or time ONLY on the code-provided clock in CONTEXT, never on your own knowledge.
- If several people are listed as candidates, ask which one the user means; never pick one yourself.
- If there is no stored data about the person, say you do not have details stored and never invent any.
- If a stored photograph is referenced, only ever mention its stored metadata (type, title, date, place). Never describe what the photograph shows or invent visual details.
- Never initiate a configured sensitive topic. Only discuss one when the person explicitly raises it; then do not invent details and gently redirect to a safer subject.
- The user speaks Spanish. Reply in Spanish, naturally and with warmth.

Reply with a single JSON object and nothing else:
{"answer": "<your answer text>", "sources": [{"entity_type": "<type>", "entity_id": "<uuid>"}]}
Include in "sources" only the ids listed under "Allowed source ids" that your answer actually uses. You may return an empty "sources" list.
"""


@dataclass(frozen=True)
class BuiltPrompt:
    system: str
    user: str


def build_prompt(context: AnswerContext) -> BuiltPrompt:
    """Serialize an ``AnswerContext`` into system/user prompt messages."""
    lines: list[str] = ["CONTEXT", ""]
    lines.append(f"- Intent: {context.intent.value}")
    lines.append(
        f"- Current date and time (code-provided clock, UTC): {context.now.isoformat()}"
    )
    if context.term is not None:
        lines.append(f"- Query term: {context.term}")
    if context.resolution is not None:
        lines.append(f"- Resolution: {context.resolution.value}")

    if context.person is not None:
        lines.append(
            f"- Subject person: {context.person.display_name} (id {context.person.id})"
        )

    if context.candidates:
        names = ", ".join(
            f"{candidate.display_name} (id {candidate.id})"
            for candidate in context.candidates
        )
        lines.append(f"- Candidate people: {names}")

    if context.verified_relations:
        lines.append("- Verified relations (verified_fact, may be stated as fact):")
        for item in context.verified_relations:
            lines.append(
                f"  - id {item.entity_id} | {item.other_display_name} is "
                f"{item.relationship_type.value}"
            )

    if context.probable_relations:
        lines.append("- Probable relations (probable_fact, must be stated as uncertain):")
        for item in context.probable_relations:
            lines.append(
                f"  - id {item.entity_id} | {item.other_display_name} possibly is "
                f"{item.relationship_type.value}"
            )

    if context.facts:
        lines.append("- Facts (epistemic kind in parentheses):")
        for item in context.facts:
            fact_type = f"type {item.fact_type} | " if item.fact_type else ""
            provenance = (
                f" | source {item.source_type}" if item.source_type else ""
            )
            lines.append(
                f"  - id {item.entity_id} | {fact_type}{item.text} "
                f"({item.epistemic.value}){provenance}"
            )

    if context.memories:
        lines.append("- Memories (memory, never present as a verified fact):")
        for item in context.memories:
            lines.append(f"  - id {item.entity_id} | {item.narrative}")

    if context.events:
        lines.append("- Events (status in parentheses, phrase by status only):")
        for item in context.events:
            lines.append(
                f"  - id {item.entity_id} | '{item.title}' on "
                f"{item.start_at.isoformat()} ({item.status.value})"
            )

    if context.media:
        lines.append("- Media (stored metadata only; never describe image content):")
        for item in context.media:
            metadata = []
            if item.title:
                metadata.append(f"title '{item.title}'")
            if item.description:
                metadata.append(f"description '{item.description}'")
            if item.approx_date is not None:
                metadata.append(f"date {item.approx_date.isoformat()}")
            if item.place_text:
                metadata.append(f"place '{item.place_text}'")
            rendered = ", ".join(metadata) if metadata else "(no extra metadata)"
            lines.append(
                f"  - id {item.entity_id} | type {item.media_type} | {rendered}"
            )

    if context.blocked_sensitive_topics:
        lines.append(
            "- Sensitive topics that must not be initiated: "
            + "; ".join(context.blocked_sensitive_topics)
        )

    if context.sensitive_topic_requested is not None:
        lines.append(
            "- Sensitive topic explicitly raised by the person: "
            f"{context.sensitive_topic_requested}"
        )
        redirect = context.sensitive_redirect_topic or "a safer subject"
        lines.append(
            "- Respond carefully without inventing details and gently redirect "
            f"toward {redirect}."
        )

    if context.policies:
        lines.append("- Policies already applied (do not override):")
        for policy in context.policies:
            lines.append(f"  - {policy}")

    lines.append("")
    lines.append("Allowed source ids you may reference:")
    for (entity_type, entity_id), usage in allowed_sources(context).items():
        lines.append(f"  - {entity_type} {entity_id} ({usage})")

    return BuiltPrompt(system=_SYSTEM, user="\n".join(lines))


def allowed_sources(context: AnswerContext) -> dict[tuple[str, str], str]:
    """Map every referenceable entity to its truthful ``usage_type``.

    ``usage_type`` is derived from the context item itself (subject, candidate,
    epistemic kind, or event status), never from the model: the caller assigns
    it after matching a returned id against this map, so a model cannot make up
    a role for an entity it was allowed to use.
    """
    allowed: dict[tuple[str, str], str] = {}
    if context.person is not None:
        allowed[("person", str(context.person.id))] = "subject"
    for candidate in context.candidates:
        key = ("person", str(candidate.id))
        if key not in allowed:
            allowed[key] = "candidate"
    for item in context.verified_relations:
        allowed[("relationship", str(item.entity_id))] = item.epistemic.value
    for item in context.probable_relations:
        allowed[("relationship", str(item.entity_id))] = item.epistemic.value
    for item in context.facts:
        allowed[("fact", str(item.entity_id))] = item.epistemic.value
        if item.source_id is not None:
            allowed[("source", str(item.source_id))] = "provenance"
    for item in context.memories:
        allowed[("memory", str(item.entity_id))] = "memory"
    for item in context.events:
        allowed[("event", str(item.entity_id))] = item.status.value
    for item in context.media:
        allowed[("media", str(item.entity_id))] = "media"
    return {key: allowed[key] for key in sorted(allowed)}
