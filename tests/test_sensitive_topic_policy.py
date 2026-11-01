import uuid

from app.domain.enums import EpistemicKind, Intent, MemoryStatus
from app.policies.sensitive_topics import (
    SensitiveTopic,
    answer_leaks_blocked_topic,
    apply_to_context,
    evaluate,
    matches_topic,
)
from app.retrieval.context_builder import AnswerContext, FactItem, MemoryItem

TOPIC = SensitiveTopic(
    topic="fallecimiento de Miguel",
    handling="Do not initiate; redirect gently if asked.",
    triggers=("murió",),
)


def test_matches_full_phrase_theme_and_subject_only_cases() -> None:
    assert matches_topic("Háblame del fallecimiento de Miguel.", TOPIC) is True
    assert matches_topic("¿Cuándo murió Miguel?", TOPIC) is True
    assert matches_topic("Miguel, fallecimiento", TOPIC) is True
    assert matches_topic("¿Quién era Miguel?", TOPIC) is False


def test_evaluate_marks_first_requested_topic_and_blocks_the_rest() -> None:
    other = SensitiveTopic(topic="deudas de la familia", triggers=("deuda",))

    evaluation = evaluate("Prefiero no hablar de deudas", (TOPIC, other))

    assert evaluation.requested == other
    assert evaluation.blocked == (TOPIC,)


def test_apply_withholds_blocked_items_and_keeps_existing_policies() -> None:
    context = AnswerContext(
        intent=Intent.person_query,
        memories=[
            MemoryItem(
                entity_id=uuid.uuid4(),
                title="Una despedida",
                narrative="Elena recuerda el fallecimiento de Miguel.",
                memory_status=MemoryStatus.consolidated,
                epistemic=EpistemicKind.memory,
            ),
            MemoryItem(
                entity_id=uuid.uuid4(),
                title="La costura",
                narrative="Elena cosía con su madre.",
                memory_status=MemoryStatus.consolidated,
                epistemic=EpistemicKind.memory,
            ),
        ],
        facts=[
            FactItem(
                entity_id=uuid.uuid4(),
                text="Miguel murió en 1998",
                status="probable",
                epistemic=EpistemicKind.probable_fact,
            )
        ],
        policies=["sensitive_content_filtered: 1 high-sensitivity memories excluded"],
    )

    result = apply_to_context(
        context,
        evaluate("¿Quién era Elena?", (TOPIC,)),
        positive_topics=("costura",),
    )

    assert [item.title for item in result.memories] == ["La costura"]
    assert result.facts == []
    assert result.blocked_sensitive_topics == ("fallecimiento de Miguel",)
    assert "sensitive_content_filtered: 1 high-sensitivity memories excluded" in (
        result.policies
    )
    assert any("sensitive_topic_not_initiated" in line for line in result.policies)


def test_answer_leak_detection_respects_the_requested_topic() -> None:
    blocked = evaluate("¿Quién era Elena?", (TOPIC,))
    requested = evaluate("Háblame del fallecimiento de Miguel", (TOPIC,))

    assert answer_leaks_blocked_topic("Miguel murió en 1998", blocked) is True
    assert answer_leaks_blocked_topic("Miguel murió en 1998", requested) is False
