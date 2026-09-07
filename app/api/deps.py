from typing import Annotated

from fastapi import Depends

from app.claims.base import ClaimExtractor
from app.claims.contradiction import ContradictionDetector
from app.claims.grouping import ClaimGrouper
from app.claims.provider import (
    get_claim_contradiction_detector,
    get_claim_extractor,
    get_claim_grouper,
)
from app.db.session import get_db
from app.embeddings.base import Embedder
from app.embeddings.provider import get_embedder
from app.llm.base import Responder
from app.llm.provider import get_responder
from app.voice.base import SpeechToText, TextToSpeech
from app.voice.provider import get_speech_to_text, get_text_to_speech

ResponderDep = Annotated[Responder, Depends(get_responder)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ClaimExtractorDep = Annotated[ClaimExtractor, Depends(get_claim_extractor)]
ClaimGrouperDep = Annotated[ClaimGrouper, Depends(get_claim_grouper)]
ClaimContradictionDetectorDep = Annotated[
    ContradictionDetector, Depends(get_claim_contradiction_detector)
]
SttDep = Annotated[SpeechToText | None, Depends(get_speech_to_text)]
TtsDep = Annotated[TextToSpeech | None, Depends(get_text_to_speech)]

__all__ = [
    "ClaimContradictionDetectorDep",
    "ClaimExtractorDep",
    "ClaimGrouperDep",
    "EmbedderDep",
    "ResponderDep",
    "SttDep",
    "TtsDep",
    "get_db",
]
