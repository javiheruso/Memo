"""Memory text assembly shared by retrieval and embedding persistence.

Both the SQLite fallback scorer and the embedding persistence paths must turn a
memory into the same text: a stored embedding and an on-the-fly embedding of the
same memory would otherwise disagree and rank differently. This helper is the
single place that decides what a memory becomes for embedding purposes — its
title plus its narrative.
"""


def memory_search_text(*, title: str | None, narrative: str) -> str:
    """Assemble the text embedded for a memory (title plus narrative)."""
    return " ".join(part for part in (title or "", narrative) if part)
