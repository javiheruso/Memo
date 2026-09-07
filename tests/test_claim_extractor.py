from app.claims.deterministic import DeterministicClaimExtractor
from app.claims.provider import EXTRACTORS, get_claim_extractor


def _extract(text: str):
    return DeterministicClaimExtractor().extract(text)


def test_extracts_first_person_past_claim():
    claims = _extract("Vivíamos en Madrid en 1972.")
    assert len(claims) == 1
    assert claims[0].original_text == "Vivíamos en Madrid en 1972"


def test_extracts_each_claim_sentence():
    claims = _extract("Vivíamos en Madrid. Teníamos un perro llamado Toby.")
    assert [claim.original_text for claim in claims] == [
        "Vivíamos en Madrid",
        "Teníamos un perro llamado Toby",
    ]


def test_extracts_recuerdo_que_phrase():
    claims = _extract("Recuerdo que íbamos al Rastro los domingos.")
    assert len(claims) == 1


def test_extracts_me_acuerdo_phrase():
    claims = _extract("Me acuerdo de que teníamos un perro.")
    assert len(claims) == 1


def test_extracts_unaccented_past_form():
    claims = _extract("viviamos en un pueblo pequeño")
    assert len(claims) == 1
    assert claims[0].original_text == "viviamos en un pueblo pequeño"


def test_skips_question():
    assert _extract("¿Quién era Paco?") == []
    assert _extract("Vivimos en Madrid?") == []


def test_skips_trivial_greeting():
    assert _extract("Buenos días") == []
    assert _extract("Hola") == []
    assert _extract("Muchas gracias") == []


def test_skips_short_trivial_acknowledgments():
    assert _extract("Sí") == []
    assert _extract("Vale") == []
    assert _extract("Ok") == []


def test_skips_request_and_imperative():
    assert _extract("Dime cuándo viene María.") == []
    assert _extract("Cuéntame qué hiciste ayer.") == []
    assert _extract("Recuerdas cuando vivíamos allí") == []


def test_skips_unpunctuated_query():
    assert _extract("quien era paco") == []
    assert _extract("donde esta antonio") == []


def test_skips_generic_chat_without_past_signal():
    assert _extract("Me encanta la tortilla de patatas") == []
    assert _extract("Hoy hace mucho calor") == []


def test_preserves_original_casing_and_accents():
    claims = _extract("Vivíamos en MADRID en 1972")
    assert claims[0].original_text == "Vivíamos en MADRID en 1972"


def test_skips_future_or_non_past_statement():
    assert _extract("Viene María el domingo") == []
    assert _extract("María viene el domingo") == []
    assert _extract("Mañana vamos al médico") == []


def test_extractor_is_deterministic():
    text = "Vivíamos en Madrid. ¿Quién era Paco?"
    assert _extract(text) == _extract(text)


def test_provider_defaults_to_deterministic():
    assert EXTRACTORS["deterministic"] is DeterministicClaimExtractor
    assert isinstance(get_claim_extractor(), DeterministicClaimExtractor)


def test_provider_rejects_unknown_extractor(monkeypatch):
    import pytest

    class _FakeSettings:
        claim_extractor = "nonexistent"

    monkeypatch.setattr(
        "app.claims.provider.get_settings", lambda: _FakeSettings()
    )
    with pytest.raises(ValueError, match="Unknown claim extractor 'nonexistent'"):
        get_claim_extractor()
