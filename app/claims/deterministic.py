"""Conservative deterministic claim extractor.

Extracts candidate claims from declarative conversational statements about the
past, mirroring the dossier definition of an "afirmación": content said in
conversation whose veracity is not yet known. It is deliberately conservative
— false negatives are fine, false positives are not — so the ingestion side
stays safe by default:

- questions (with or without ``?``) are skipped,
- greetings, thanks and short acknowledgments are skipped,
- requests, imperatives and memory/recall prompts are skipped,
- any sentence without a first-person past-tense signal is skipped.

The signal set targets first-person Spanish past forms: imperfect and preterite
verb suffixes (``-ábamos``, ``-íamos``, ``-ía``, ``-aba``, and their
unaccented spellings) plus a curated set of strong irregular past verbs and
narrative phrases ("recuerdo que", "me acuerdo"). Canonicalization is deferred
to the claim-grouping phase, so each candidate keeps only its verbatim
``original_text``.
"""

import re
import unicodedata

from app.claims.base import ClaimCandidate, ClaimExtractor

_SENTENCE_SPLIT = re.compile(r"[.!;\n]+")

_TRIVIAL = {
    "hola",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "hasta luego",
    "hasta manana",
    "adios",
    "gracias",
    "muchas gracias",
    "de nada",
    "vale",
    "ok",
    "okay",
    "perfecto",
    "muy bien",
    "bien",
    "claro",
    "genial",
    "estupendo",
    "que bien",
    "que alegria",
    "guay",
}

_REQUEST_PREFIXES = (
    "dime",
    "dime si",
    "cuentame",
    "dame",
    "muestrame",
    "pasame",
    "repiteme",
    "repite",
    "busca",
    "mira",
    "muestra",
    "cuenta",
    "hablame",
    "ayudame",
    "ensename",
    "pon",
    "abre",
    "di",
    "escucha",
    "quiero que",
    "necesito que",
    "puedes",
    "podrias",
    "puede",
    "podria",
    "quieres",
    "te acuerdas",
    "recuerdas",
    "sabes",
    "sabrias",
    "hazme",
    "preparame",
    "recuerdame",
)

# Unambiguous question words used as a backstop for questions that arrive
# without a "?". Ambiguous ones ("que", "como", "cuando") are excluded on
# purpose: "cuando vivíamos..." is a claim, not a question.
_INTERROGATIVE_WORDS = re.compile(
    r"\b(?:quien|quienes|cual|cuales|cuanto|cuantos|cuanta|cuantas|donde)\b",
    re.IGNORECASE,
)
_INTERROGATIVE_PHRASES = (
    "por que",
    "para que",
    "a que",
    "que dia",
    "que hora",
    "que fecha",
    "que es",
    "que era",
    "que fue",
    "quien es",
    "quien era",
    "quien fue",
    "donde esta",
)

# Accent-preserved patterns: the 1st-person-plural imperfect suffixes are
# high-precision past-tense signals. The generic accented "-ía" ending is
# deliberately NOT used because it also ends proper names ("María", "Sofía");
# accented singular imperfect verbs are covered by the curated set below.
_ACCENTED_PAST = re.compile(r"\b\w{2,}(?:ábamos|íamos)\b")

# Accent-stripped signals: curated first-person past verbs plus the
# unaccented regular suffixes ("abamos", "iamos", "aba").
_PLAIN_PAST_WORDS = (
    "fui",
    "fuimos",
    "tuve",
    "estuve",
    "estuvimos",
    "conoci",
    "vivi",
    "vi",
    "dije",
    "hice",
    "pude",
    "quise",
    "supe",
    "conocimos",
    "vivimos",
    "visitamos",
    "viajamos",
    "salimos",
    "llegamos",
    "cenamos",
    "paseamos",
    "bailamos",
    "jugamos",
    "cantamos",
    "compramos",
    "trabajamos",
    "hablamos",
    "aprendimos",
    "comimos",
    "lloramos",
    "reimos",
    "estudiamos",
    "era",
    "eramos",
    "estaba",
    "tenia",
    "vivia",
    "iba",
    "ibamos",
    "solia",
    "soliamos",
    "queria",
    "sabia",
    "conocia",
    "recordaba",
    "acordaba",
    "ayudaba",
    "habia",
    "fue",
    "recuerdo",
    "comia",
    "bebia",
    "subia",
    "corria",
    "veia",
    "leia",
    "decia",
    "ponia",
    "salia",
    "oia",
    "creia",
)
_PLAIN_PAST_SUFFIX = re.compile(r"\b\w{3,}(?:abamos|iamos)\b|\b\w{4,}aba\b")
_PLAIN_PAST_PHRASES = (
    "recuerdo que",
    "me acuerdo",
    "me acordaba",
    "nos acordamos",
)
_PLAIN_PAST_WORDS_PATTERN = re.compile(
    r"\b(?:" + "|".join(_PLAIN_PAST_WORDS) + r")\b"
)

_MIN_SENTENCE_LENGTH = 6


def _fold(text: str) -> str:
    """Lowercase and strip accents for matching; keeps the original for storage."""
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if unicodedata.category(char) != "Mn"
    )


class DeterministicClaimExtractor(ClaimExtractor):
    """Rules-only extractor: deterministic, conservative, no network."""

    def extract(self, text: str) -> list[ClaimCandidate]:
        candidates: list[ClaimCandidate] = []
        for sentence in _SENTENCE_SPLIT.split(text):
            candidate = self._extract_sentence(sentence)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _extract_sentence(sentence: str) -> ClaimCandidate | None:
        original = sentence.strip()
        folded = _fold(original)
        if not folded or len(folded) < _MIN_SENTENCE_LENGTH:
            return None

        if "?" in original or "¿" in original:
            return None

        if folded in _TRIVIAL:
            return None

        if any(folded.startswith(prefix) for prefix in _REQUEST_PREFIXES):
            return None

        if _INTERROGATIVE_WORDS.search(folded) or any(
            phrase in folded for phrase in _INTERROGATIVE_PHRASES
        ):
            return None

        if not DeterministicClaimExtractor._has_past_signal(original, folded):
            return None

        return ClaimCandidate(original_text=original)

    @staticmethod
    def _has_past_signal(original: str, folded: str) -> bool:
        if _ACCENTED_PAST.search(original):
            return True
        if _PLAIN_PAST_SUFFIX.search(folded):
            return True
        if _PLAIN_PAST_WORDS_PATTERN.search(folded):
            return True
        return any(phrase in folded for phrase in _PLAIN_PAST_PHRASES)
