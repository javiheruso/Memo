"""Claim extraction and grouping seams.

The conversation pipeline hands user text to a ``ClaimExtractor`` and persists
whatever candidate claims come back, then routes each claim into a
``ClaimGroup`` through a ``ClaimGrouper``. The deterministic implementations are
the conservative defaults; a real provider (for example an LLM-backed extractor
or a semantic grouper) can be registered in ``app/claims/provider.py`` and
selected via ``MEMO_CLAIM_EXTRACTOR`` / ``MEMO_CLAIM_GROUPER`` without touching
routing, retrieval or the conversation service.
"""
