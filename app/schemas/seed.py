from pydantic import BaseModel


class SeedSummaryRead(BaseModel):
    """Summary of the rows created by an example-data seed run."""

    persons: int
    relationships: int
    facts: int
    memories: int
    media: int
