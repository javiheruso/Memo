from app.domain.enums import Intent
from app.retrieval.fact_topics import FactTopic
from app.retrieval.intent_router import IntentRouter, is_explicit_pronoun_question


def test_explicit_pronoun_question_is_not_a_person_name() -> None:
    for phrase in (
        "¿Quién es él?",
        "¿Quién era ella?",
        "¿Quién fue ese?",
        "¿Y quién es él?",
        "¿Quién es ella?",
    ):
        assert is_explicit_pronoun_question(phrase), phrase
        result = IntentRouter().route(phrase)
        assert result.intent == Intent.general_chat, phrase
        assert result.term is None, phrase
        # The existing pronoun follow-up recognizer still claims the turn, so
        # the service can reuse the verified previous subject.
        follow_up = IntentRouter().route_pronoun_follow_up(phrase)
        assert follow_up is not None, phrase
        assert follow_up.intent == Intent.person_query, phrase
        assert follow_up.term is None, phrase

    assert not is_explicit_pronoun_question("¿Quién es Antonio?")
    assert not is_explicit_pronoun_question("¿Y él?")
    named = IntentRouter().route("¿Quién es Antonio?")
    assert named.intent == Intent.person_query
    assert named.term == "Antonio"


def test_routes_spanish_person_question():
    result = IntentRouter().route("¿Quién era Paco?")
    assert result.intent == Intent.person_query
    assert result.term == "Paco"


def test_routes_hablame_de_person_as_person_query():
    result = IntentRouter().route("Háblame de Elena")
    assert result.intent == Intent.person_query
    assert result.term == "Elena"


def test_routes_cuentame_sobre_person_as_person_query():
    result = IntentRouter().route("Cuéntame sobre Miguel")
    assert result.intent == Intent.person_query
    assert result.term == "Miguel"


def test_routes_person_about_request_variants():
    cases = [
        ("Háblame sobre Elena", "Elena"),
        ("Háblame acerca de Elena", "Elena"),
        ("Cuéntame de Miguel", "Miguel"),
        ("Cuéntame algo sobre Miguel", "Miguel"),
        ("Dime algo de Elena", "Elena"),
        ("Háblame de Paco García", "Paco García"),
        ("Háblame de Elena, por favor", "Elena"),
    ]
    for phrase, expected_term in cases:
        result = IntentRouter().route(phrase)
        assert result.intent == Intent.person_query, phrase
        assert result.term == expected_term, phrase


def test_routes_describe_person_request_as_person_query():
    result = IntentRouter().route("Descríbeme a Elena")
    assert result.intent == Intent.person_query
    assert result.term == "Elena"


def test_routes_polite_person_request_as_person_query():
    cases = [
        ("¿Me puedes contar sobre Elena?", "Elena"),
        ("¿Puedes contarme sobre Elena?", "Elena"),
        ("¿Podrías decirme algo sobre Elena?", "Elena"),
    ]
    for phrase, expected_term in cases:
        result = IntentRouter().route(phrase)
        assert result.intent == Intent.person_query, phrase
        assert result.term == expected_term, phrase


def test_person_request_without_proper_name_stays_general_chat():
    for phrase in [
        "Háblame de la vida",
        "Háblame de ti",
        "Háblame de mi familia",
        "Cuéntame un chiste",
        "Cuéntame sobre lo que pasó",
        "Cuéntame sobre el viaje",
        "Dime algo bonito",
        "Descríbeme la casa",
    ]:
        result = IntentRouter().route(phrase)
        assert result.intent == Intent.general_chat, phrase
        assert result.term is None, phrase


def test_lowercase_stt_person_requests_keep_working():
    cases = [
        ("hablame de elena", "elena"),
        ("cuentame sobre miguel", "miguel"),
        ("dime algo de paco garcia", "paco garcia"),
        ("me gustaria saber sobre elena", "elena"),
        ("quiero saber quien es miguel", "miguel"),
        ("tengo curiosidad por antonio", "antonio"),
    ]
    for phrase, expected_term in cases:
        result = IntentRouter().route(phrase)
        assert result.intent == Intent.person_query, phrase
        assert result.term == expected_term, phrase
        assert IntentRouter().is_personal_lookup_request(phrase), phrase


def test_narrative_mentions_are_not_personal_lookups():
    for phrase in [
        "Ayer vi a María en el mercado.",
        "María viene mañana a visitarme.",
        "Viene María el domingo.",
        "Mi marido viene a recogerme.",
        "El domingo viene mi hija.",
        "Mi marido se llamaba Miguel.",
        "Ayer estuve pensando en Carmen.",
        "Hoy estuve recordando la mudanza de Carmen.",
        "Qué bien que a María le gusta el mar.",
        "¿Sabes que a María le gusta el mar?",
        "¿Ayer vi a María?",
    ]:
        result = IntentRouter().route(phrase)
        assert result.intent == Intent.general_chat, phrase
        assert result.term is None, phrase
        assert not IntentRouter().is_personal_lookup_request(phrase), phrase


def test_explicit_event_and_fact_requests_survive_lowercase_and_no_punctuation():
    event = IntentRouter().route("cuando viene maria")
    assert event.intent == Intent.event_query
    assert event.term == "maria"

    events = IntentRouter().route("que eventos tiene antonio")
    assert events.intent == Intent.event_query
    assert events.term == "antonio"

    fact = IntentRouter().route("que le gusta a elena")
    assert fact.intent == Intent.fact_query
    assert fact.term == "elena"
    assert fact.fact_topic == FactTopic.preference


def test_person_request_does_not_steal_memory_recall():
    result = IntentRouter().route("¿Te acuerdas de cuando venía María?")
    assert result.intent == Intent.memory_query


def test_person_request_does_not_steal_fact_question():
    result = IntentRouter().route("¿Qué le gusta a Elena?")
    assert result.intent == Intent.fact_query
    assert result.term == "Elena"


def test_routes_english_person_question():
    result = IntentRouter().route("Who was Paco?")
    assert result.intent == Intent.person_query
    assert result.term == "Paco"


def test_routes_person_question_without_accent():
    result = IntentRouter().route("Quien es Francisco?")
    assert result.intent == Intent.person_query
    assert result.term == "Francisco"


def test_extracts_multi_word_term():
    result = IntentRouter().route("¿Quién es Paco García?")
    assert result.intent == Intent.person_query
    assert result.term == "Paco García"


def test_routes_event_question():
    result = IntentRouter().route("¿Viene María hoy?")
    assert result.intent == Intent.event_query
    assert result.term == "María"


def test_routes_event_question_with_cuando():
    result = IntentRouter().route("¿Cuándo viene María?")
    assert result.intent == Intent.event_query
    assert result.term == "María"


def test_routes_event_question_with_future_verb():
    result = IntentRouter().route("¿Vendrá Antonio esta semana?")
    assert result.intent == Intent.event_query
    assert result.term == "Antonio"


def test_routes_events_of_person_question():
    result = IntentRouter().route("¿Qué eventos tiene Antonio esta semana?")
    assert result.intent == Intent.event_query
    assert result.term == "Antonio"


def test_routes_event_question_without_person_keeps_no_term():
    result = IntentRouter().route("¿Cuándo hay eventos?")
    assert result.intent == Intent.event_query
    assert result.term is None


def test_routes_preference_question_as_fact_query():
    result = IntentRouter().route("¿Qué le gusta a Elena?")
    assert result.intent == Intent.fact_query
    assert result.term == "Elena"
    assert result.fact_topic == FactTopic.preference


def test_routes_specific_hobby_question_as_fact_query():
    result = IntentRouter().route("¿Qué aficiones tiene Elena?")
    assert result.intent == Intent.fact_query
    assert result.term == "Elena"
    assert result.fact_topic == FactTopic.hobby


def test_fact_question_wins_over_event_marker_for_birth_question():
    result = IntentRouter().route("¿Cuándo nació Elena?")
    assert result.intent == Intent.fact_query
    assert result.term == "Elena"
    assert result.fact_topic == FactTopic.place


def test_routes_english_preference_question_as_fact_query():
    result = IntentRouter().route("What does Elena like?")
    assert result.intent == Intent.fact_query
    assert result.term == "Elena"
    assert result.fact_topic == FactTopic.preference


def test_declarative_preference_statement_is_not_a_fact_query():
    assert IntentRouter().route("Me gusta Elena").intent == Intent.general_chat


def test_routes_general_chat_by_default():
    result = IntentRouter().route("Me encanta la tortilla de patatas")
    assert result.intent == Intent.general_chat
    assert result.term is None


def test_routes_blank_as_unknown():
    assert IntentRouter().route("   ").intent == Intent.unknown
    assert IntentRouter().route("").intent == Intent.unknown


def test_routes_spanish_memory_recall_question():
    result = IntentRouter().route("¿Te acuerdas de cuando íbamos a bailar?")
    assert result.intent == Intent.memory_query
    assert result.term is None


def test_routes_recuerdas_question():
    result = IntentRouter().route("¿Recuerdas cuando íbamos a pescar?")
    assert result.intent == Intent.memory_query


def test_routes_english_memory_recall_question():
    result = IntentRouter().route("Do you remember when we went dancing?")
    assert result.intent == Intent.memory_query
    assert result.term is None


def test_routes_english_remember_when_question():
    result = IntentRouter().route("Remember when we went fishing?")
    assert result.intent == Intent.memory_query


def test_memory_recall_wins_over_event_marker():
    result = IntentRouter().route("¿Te acuerdas de cuando venía María a bailar?")
    assert result.intent == Intent.memory_query


def test_person_question_wins_over_memory_marker():
    result = IntentRouter().route("¿Recuerdas quién era Paco?")
    assert result.intent == Intent.person_query
    assert result.term == "Paco"


def test_routes_orientation_day_question():
    result = IntentRouter().route("¿Qué día es hoy?")
    assert result.intent == Intent.orientation_query
    assert result.term is None


def test_routes_orientation_weekday_question():
    result = IntentRouter().route("¿Qué día de la semana es?")
    assert result.intent == Intent.orientation_query


def test_routes_orientation_date_question():
    result = IntentRouter().route("¿Qué fecha es hoy?")
    assert result.intent == Intent.orientation_query


def test_routes_orientation_time_question():
    result = IntentRouter().route("¿Qué hora es?")
    assert result.intent == Intent.orientation_query


def test_routes_orientation_english_question():
    assert IntentRouter().route("What time is it?").intent == Intent.orientation_query
    assert IntentRouter().route("What day is it?").intent == Intent.orientation_query
    assert IntentRouter().route("What's the date?").intent == Intent.orientation_query


def test_event_question_wins_over_day_phrase():
    result = IntentRouter().route("¿Qué día viene María?")
    assert result.intent == Intent.event_query
    assert result.term == "María"


def test_memory_recall_wins_over_orientation():
    result = IntentRouter().route("¿Te acuerdas de qué día es hoy?")
    assert result.intent == Intent.memory_query


def test_general_chat_with_hoy_is_not_orientation():
    assert (
        IntentRouter().route("Hoy hace buen día para pasear").intent
        == Intent.general_chat
    )


def test_routes_elliptical_food_follow_up_without_subject():
    result = IntentRouter().route_fact_follow_up("¿Y de comer?")
    assert result is not None
    assert result.intent == Intent.fact_query
    assert result.term is None
    assert result.fact_topic == FactTopic.food


def test_routes_elliptical_place_follow_up_without_subject():
    result = IntentRouter().route_fact_follow_up("¿Y dónde vivió?")
    assert result is not None
    assert result.intent == Intent.fact_query
    assert result.term is None
    assert result.fact_topic == FactTopic.place


def test_follow_up_requires_continuation_marker():
    assert IntentRouter().route_fact_follow_up("¿Qué música le gusta?") is None


def test_follow_up_with_explicit_subject_is_not_elliptical():
    assert IntentRouter().route_fact_follow_up("¿Y dónde vivió Miguel?") is None


def test_follow_up_without_fact_topic_returns_none():
    assert IntentRouter().route_fact_follow_up("¿Y tú?") is None


def test_routes_possessive_childhood_follow_up():
    result = IntentRouter().route_possessive_follow_up("Háblame sobre su infancia")
    assert result is not None
    assert result.intent == Intent.fact_query
    assert result.term is None
    assert result.fact_topic == FactTopic.childhood
    assert result.follow_up_topic == "la infancia"


def test_routes_lowercase_unaccented_possessive_follow_up():
    result = IntentRouter().route_possessive_follow_up("hablame sobre su infancia")
    assert result is not None
    assert result.intent == Intent.fact_query
    assert result.fact_topic == FactTopic.childhood


def test_routes_possessive_family_follow_up():
    result = IntentRouter().route_possessive_follow_up("Cuéntame de su familia")
    assert result is not None
    assert result.intent == Intent.fact_query
    assert result.fact_topic == FactTopic.family
    assert result.follow_up_topic == "la familia"


def test_possessive_follow_up_ignores_explicit_subject():
    assert (
        IntentRouter().route_possessive_follow_up("Háblame sobre su infancia de Miguel")
        is None
    )


def test_possessive_follow_up_requires_recognizable_topic():
    assert IntentRouter().route_possessive_follow_up("Háblame sobre su casa") is None
    assert IntentRouter().route_possessive_follow_up("Hola") is None


def test_routes_only_the_four_close_relative_follow_ups():
    router = IntentRouter()
    cases = {
        "¿Y su hija?": ("hija", True),
        "¿Y su hijo?": ("hijo", True),
        "¿Y su madre?": ("madre", False),
        "¿Y su padre?": ("padre", False),
        "su hija": ("hija", True),
    }
    for phrase, (role, active_is_parent) in cases.items():
        query = router.route_relative_follow_up(phrase)
        assert query is not None, phrase
        assert query.role == role, phrase
        assert query.active_is_parent is active_is_parent, phrase


def test_relative_follow_up_rejects_extra_content_and_other_relatives():
    router = IntentRouter()
    for phrase in (
        "¿Y su nieta?",
        "¿Y su hija mayor?",
        "Háblame de su hija",
        "¿Y su perro?",
        "Hola",
    ):
        assert router.route_relative_follow_up(phrase) is None, phrase
