from pydantic import BaseModel, Field
from typing import List


class RouterOutput(BaseModel):
    collections: List[str] = Field(
        ...,
        description="Names of the collections to search for this query"
    )
    token_allocation: int = Field(
        ...,
        ge=150,
        le=2000,
        description="Number of tokens to allocate for the LLM response"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the routing decision"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the routing decision (0.0 to 1.0)"
    )
