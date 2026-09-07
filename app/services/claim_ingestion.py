"""Claim ingestion with deterministic grouping and contradiction signals.

Wraps claim persistence (append-only, no promotion), claim-group routing and
contradiction detection in fail-safe units used by the conversation pipeline.
Any failure here is caught upstream and can never change the response behavior.
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimCandidate
from app.claims.contradiction import (
    ContradictionDetector,
    DeterministicContradictionDetector,
    fact_surface_text,
)
from app.claims.grouping import ClaimGrouper, DeterministicClaimGrouper
from app.domain.enums import ClaimGroupStatus
from app.models import Claim, ClaimGroup
from app.repositories.claim import ClaimRepository
from app.repositories.claim_group import ClaimGroupRepository
from app.repositories.claim_group_contradiction import (
    ClaimGroupContradictionRepository,
)
from app.repositories.fact import FactRepository

logger = logging.getLogger(__name__)


class ClaimIngestionService:
    """Persists candidate claims, routes each into its claim group, and flags
    contradictions against verified facts.

    Grouping is additive: claims are always stored append-only, and a grouping
    failure leaves the claims persisted but ungrouped rather than dropping or
    mutating them. Contradiction detection is a separate, best-effort step that
    runs only after the claims and groups are committed, so a detector or
    persistence error there can never roll back the ingestion or alter the
    answer.
    """

    def __init__(
        self,
        session: AsyncSession,
        grouper: ClaimGrouper | None = None,
        contradiction_detector: ContradictionDetector | None = None,
    ) -> None:
        self._session = session
        self._grouper = grouper or DeterministicClaimGrouper()
        self._detector = (
            contradiction_detector or DeterministicContradictionDetector()
        )
        self._claims = ClaimRepository(session)
        self._groups = ClaimGroupRepository(session)
        self._contradictions = ClaimGroupContradictionRepository(session)
        self._facts = FactRepository(session)

    async def ingest(
        self,
        candidates: list[ClaimCandidate],
        *,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        speaker_person_id: uuid.UUID | None = None,
    ) -> list[Claim]:
        claims = await self._claims.add_many(
            candidates,
            conversation_id=conversation_id,
            message_id=message_id,
            speaker_person_id=speaker_person_id,
        )
        touched_groups: list[ClaimGroup] = []
        seen_group_ids: set[uuid.UUID] = set()
        for claim in claims:
            group_key = self._grouper.group_key(claim.original_text)
            group = await self._groups.attach(claim, group_key)
            if group.id not in seen_group_ids:
                seen_group_ids.add(group.id)
                touched_groups.append(group)
        await self._session.commit()
        await self._detect_contradictions(touched_groups)
        return claims

    async def _detect_contradictions(self, groups: list[ClaimGroup]) -> None:
        """Flag contradiction signals against verified facts, best-effort.

        Runs after the ingestion commit: claims and groups are already durable,
        so any detector or persistence failure here only skips the signals. A
        signal never promotes or rejects anything — it only records a
        contradiction row and lifts ``needs_review`` on the group.
        """
        try:
            facts = await self._facts.list_verified()
            for group in groups:
                # Manually resolved/ignored groups are out of the review queue:
                # detection never re-flags them, even when a new claim attaches.
                if group.status is not ClaimGroupStatus.active:
                    continue
                group_needs_review = False
                for fact in facts:
                    surface = fact_surface_text(
                        value_text=fact.value_text, place_text=fact.place_text
                    )
                    if not surface:
                        continue
                    signal = self._detector.detect(group.canonical_text, surface)
                    if signal is None:
                        continue
                    await self._contradictions.upsert(
                        claim_group_id=group.id,
                        fact_id=fact.id,
                        detector_name=getattr(self._detector, "name", "unknown"),
                        matched_terms=", ".join(signal.matched_terms),
                    )
                    group_needs_review = True
                if group_needs_review:
                    group.needs_review = True
            await self._session.commit()
        except Exception:
            logger.warning(
                "claim contradiction detection failed for %d group(s); "
                "claims and groups were already persisted",
                len(groups),
                exc_info=True,
            )
