from app.domain.enums import Intent
from app.retrieval.fact_topics import FactTopic
from app.retrieval.intent_router import IntentRouter


def test_routes_spanish_person_question():
    result = IntentRouter().route("¿Quién era Paco?")
    assert result.intent == Intent.person_query
    assert result.term == "Paco"


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
