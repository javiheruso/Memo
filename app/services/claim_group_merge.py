"""Explicit human canonical merge of duplicate claim groups.

A human reviewing the claim-group queue may decide that two groups that only
share a wording class (paraphrases that deterministic grouping kept apart) are
really the same claim. ``ClaimGroupMergeService.merge`` folds one active group
into another active survivor without any auto-promotion or semantic engine:

- member claims move to the survivor unchanged (never mutated),
- contradiction evidence rows transfer to the survivor, folding rows that hit
  the survivor's ``(claim_group_id, fact_id, detector_name)`` unique key instead
  of duplicating them (min ``first_detected_at``, max ``last_detected_at``,
  summed ``detection_count``),
- remembered keys pointing at the absorbed group re-point to the survivor, and
  the absorbed group's own canonical text is remembered so future claims using
  that wording attach to the survivor (see ``ClaimGroupRepository.attach``),
- only then is the absorbed group row deleted.

Guards keep the review semantics honest: self-merge, ``ignored`` or otherwise
non-``active`` endpoints, groups that already produced a materialization (their
provenance would dangle), and opposite-polarity merges without an explicit
``confirm_opposite_polarity`` flag are all refused. ``suggest`` offers
deterministic targets ranked by shared content tokens.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.grouping import normalize_claim_text
from app.domain.enums import ClaimGroupStatus, MaterializationSourceType
from app.models import (
    ClaimGroup,
    ClaimGroupContradiction,
    ClaimGroupMember,
    ClaimGroupRememberedKey,
    Materialization,
)
from app.repositories.claim_group import ClaimGroupRepository

_NEGATION_TOKENS = frozenset({"no", "nunca", "jamas", "tampoco"})
_NEGATION_PHRASES = ("ya no", "ni siquiera")

# High-frequency framing words that carry no content about the claim, mirroring
# the contradiction detector's vocabulary so suggestions and polarity checks
# agree with the signal the reviewer already sees.
_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "con",
        "cuando",
        "como",
        "de",
        "del",
        "donde",
        "e",
        "el",
        "en",
        "es",
        "era",
        "fue",
        "habia",
        "hay",
        "la",
        "las",
        "le",
        "lo",
        "los",
        "me",
        "mas",
        "mi",
        "mis",
        "muy",
        "nos",
        "o",
        "para",
        "pero",
        "por",
        "que",
        "quien",
        "recuerdo",
        "recuerda",
        "acuerdo",
        "acordaba",
        "se",
        "ser",
        "su",
        "sus",
        "te",
        "tu",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "ya",
    }
)


class ClaimGroupMergeError(Exception):
    """Base error for the merge service; mapped to HTTP by the API."""


class UnknownClaimGroupMergeError(ClaimGroupMergeError):
    """A claim group involved in the merge does not exist."""


class SelfClaimGroupMergeError(ClaimGroupMergeError):
    """A claim group cannot be merged into itself."""


class IgnoredClaimGroupMergeError(ClaimGroupMergeError):
    """An ``ignored`` group is human-closed and cannot take part in a merge."""


class InactiveClaimGroupMergeError(ClaimGroupMergeError):
    """Only ``active`` groups participate in a merge."""


class MaterializedClaimGroupMergeError(ClaimGroupMergeError):
    """The group already produced a materialization; merging would orphan it."""


class OppositePolarityMergeError(ClaimGroupMergeError):
    """The merge would fold a negated wording into a non-negated survivor (or
    vice versa); it requires an explicit ``confirm_opposite_polarity`` flag."""


@dataclass(frozen=True)
class MergeSuggestion:
    """A deterministic merge target for a source group.

    ``shared_terms`` counts the content tokens the two canonical texts have in
    common (stopwords removed) and ``opposite_polarity`` tells the reviewer when
    merging would need the confirmation flag.
    """

    group_id: uuid.UUID
    canonical_text: str
    total_mentions: int
    needs_review: bool
    shared_terms: int
    opposite_polarity: bool


def has_negation(text: str) -> bool:
    """True when the normalized text carries an explicit negation marker."""
    normalized = normalize_claim_text(text)
    tokens = set(normalized.split())
    if any(marker in tokens for marker in _NEGATION_TOKENS):
        return True
    return any(phrase in normalized for phrase in _NEGATION_PHRASES)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_claim_text(text).split()
        if token not in _STOPWORDS
    }


def _merge_matched_terms(left: str | None, right: str | None) -> str | None:
    terms = set()
    for value in (left, right):
        if value:
            terms.update(term for term in value.split(", ") if term)
    return ", ".join(sorted(terms)) or None


class ClaimGroupMergeService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._groups = ClaimGroupRepository(session)

    # ------------------------------------------------------------------
    # Deterministic suggestions
    # ------------------------------------------------------------------

    async def suggest(
        self, group_id: uuid.UUID, *, limit: int = 200
    ) -> list[MergeSuggestion]:
        """Rank the other active groups as merge targets for ``group_id``.

        The ranking is pure token overlap (deterministic, no semantics): groups
        sharing more content tokens with the source come first, so the reviewer
        sees the most plausible duplicates at the top of the form.
        """
        source = await self._session.get(ClaimGroup, group_id)
        if source is None:
            raise UnknownClaimGroupMergeError(
                f"claim group {group_id} not found"
            )
        candidates = await self._groups.list_recent(
            limit=limit, status=ClaimGroupStatus.active
        )
        source_tokens = _content_tokens(source.canonical_text)
        source_negated = has_negation(source.canonical_text)
        suggestions = [
            MergeSuggestion(
                group_id=candidate.id,
                canonical_text=candidate.canonical_text,
                total_mentions=candidate.total_mentions,
                needs_review=candidate.needs_review,
                shared_terms=len(source_tokens & _content_tokens(candidate.canonical_text)),
                opposite_polarity=source_negated ^ has_negation(candidate.canonical_text),
            )
            for candidate in candidates
            if candidate.id != source.id
        ]
        suggestions.sort(key=lambda item: (-item.shared_terms, item.canonical_text))
        return suggestions

    # ------------------------------------------------------------------
    # Explicit merge
    # ------------------------------------------------------------------

    async def merge(
        self,
        group_id: uuid.UUID,
        target_group_id: uuid.UUID,
        *,
        confirm_opposite_polarity: bool = False,
    ) -> ClaimGroup:
        """Fold ``group_id`` into the surviving ``target_group_id``.

        The absorbed group's memberships, contradiction evidence, remembered
        keys and canonical text are preserved (moved/folded/re-pointed) before
        the group row is deleted, and the whole unit commits once. The returned
        survivor is freshly loaded with its members and contradiction evidence.
        """
        if group_id == target_group_id:
            raise SelfClaimGroupMergeError(
                f"claim group {group_id} cannot be merged into itself"
            )

        source = await self._session.get(ClaimGroup, group_id)
        if source is None:
            raise UnknownClaimGroupMergeError(f"claim group {group_id} not found")
        target = await self._session.get(ClaimGroup, target_group_id)
        if target is None:
            raise UnknownClaimGroupMergeError(
                f"target claim group {target_group_id} not found"
            )

        self._guard_statuses(source, target)
        await self._guard_no_materializations(source, target)
        self._guard_opposite_polarity(source, target, confirm_opposite_polarity)

        await self._move_memberships(source.id, target.id)
        await self._fold_contradictions(source.id, target.id)
        await self._repoint_remembered_keys(source.id, target.id)
        await self._remember_absorbed_key(source.canonical_text, target.id)
        if source.needs_review:
            target.needs_review = True

        await self._groups.refresh_group_metrics(target.id)
        await self._session.execute(
            delete(ClaimGroup).where(ClaimGroup.id == source.id)
        )
        await self._session.commit()

        merged = await self._groups.get_with_members(target.id)
        if merged is None:
            raise RuntimeError("surviving claim group disappeared after merge")
        return merged

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    @staticmethod
    def _guard_statuses(source: ClaimGroup, target: ClaimGroup) -> None:
        for group, role in ((source, "source"), (target, "target")):
            if group.status is ClaimGroupStatus.ignored:
                raise IgnoredClaimGroupMergeError(
                    f"cannot merge the {role} claim group {group.id}: it is ignored"
                )
            if group.status is not ClaimGroupStatus.active:
                raise InactiveClaimGroupMergeError(
                    f"only active claim groups can be merged; the {role} group "
                    f"{group.id} is {group.status.value}"
                )

    async def _guard_no_materializations(
        self, source: ClaimGroup, target: ClaimGroup
    ) -> None:
        stmt = (
            select(Materialization.id)
            .where(
                Materialization.source_type == MaterializationSourceType.claim_group,
                Materialization.source_id.in_([source.id, target.id]),
            )
            .limit(1)
        )
        if (await self._session.execute(stmt)).scalar_one_or_none() is not None:
            raise MaterializedClaimGroupMergeError(
                "a claim group that already produced a materialization cannot be "
                "merged: its provenance row would dangle"
            )

    @staticmethod
    def _guard_opposite_polarity(
        source: ClaimGroup,
        target: ClaimGroup,
        confirm_opposite_polarity: bool,
    ) -> None:
        if (
            has_negation(source.canonical_text)
            != has_negation(target.canonical_text)
            and not confirm_opposite_polarity
        ):
            raise OppositePolarityMergeError(
                "this merge folds a negated wording into a non-negated survivor "
                "(or vice versa); pass confirm_opposite_polarity=true to proceed"
            )

    # ------------------------------------------------------------------
    # Evidence-preserving moves
    # ------------------------------------------------------------------

    async def _move_memberships(
        self, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> None:
        claim_ids = list(
            (
                await self._session.execute(
                    select(ClaimGroupMember.claim_id).where(
                        ClaimGroupMember.claim_group_id == source_id
                    )
                )
            ).scalars()
        )
        if not claim_ids:
            return
        await self._session.execute(
            delete(ClaimGroupMember).where(
                ClaimGroupMember.claim_group_id == source_id
            )
        )
        await self._session.execute(
            insert(ClaimGroupMember),
            [
                {"claim_group_id": target_id, "claim_id": claim_id}
                for claim_id in claim_ids
            ],
        )

    async def _fold_contradictions(
        self, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> None:
        columns = (
            ClaimGroupContradiction.id,
            ClaimGroupContradiction.fact_id,
            ClaimGroupContradiction.detector_name,
            ClaimGroupContradiction.matched_terms,
            ClaimGroupContradiction.first_detected_at,
            ClaimGroupContradiction.last_detected_at,
            ClaimGroupContradiction.detection_count,
        )
        rows = (
            await self._session.execute(
                select(*columns).where(
                    ClaimGroupContradiction.claim_group_id == source_id
                )
            )
        ).all()
        for row in rows:
            existing_id = (
                await self._session.execute(
                    select(ClaimGroupContradiction.id).where(
                        ClaimGroupContradiction.claim_group_id == target_id,
                        ClaimGroupContradiction.fact_id == row.fact_id,
                        ClaimGroupContradiction.detector_name
                        == row.detector_name,
                    )
                )
            ).scalar_one_or_none()
            if existing_id is None:
                await self._session.execute(
                    update(ClaimGroupContradiction)
                    .where(ClaimGroupContradiction.id == row.id)
                    .values(claim_group_id=target_id)
                )
                continue
            target_row = await self._session.get(
                ClaimGroupContradiction, existing_id
            )
            if target_row is not None:
                target_row.detection_count += row.detection_count
                target_row.matched_terms = _merge_matched_terms(
                    target_row.matched_terms, row.matched_terms
                )
                target_row.first_detected_at = min(
                    target_row.first_detected_at, row.first_detected_at
                )
                target_row.last_detected_at = max(
                    target_row.last_detected_at, row.last_detected_at
                )
            await self._session.execute(
                delete(ClaimGroupContradiction).where(
                    ClaimGroupContradiction.id == row.id
                )
            )

    async def _repoint_remembered_keys(
        self, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> None:
        await self._session.execute(
            update(ClaimGroupRememberedKey)
            .where(ClaimGroupRememberedKey.claim_group_id == source_id)
            .values(claim_group_id=target_id)
        )

    async def _remember_absorbed_key(
        self, canonical_text: str, target_id: uuid.UUID
    ) -> None:
        """Record the absorbed canonical text so future claims route to target."""
        existing = (
            await self._session.execute(
                select(ClaimGroupRememberedKey).where(
                    ClaimGroupRememberedKey.canonical_text == canonical_text
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            self._session.add(
                ClaimGroupRememberedKey(
                    canonical_text=canonical_text, claim_group_id=target_id
                )
            )
        else:
            existing.claim_group_id = target_id
