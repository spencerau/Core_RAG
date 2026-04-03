from pydantic import BaseModel, Field
from typing import List, Optional


class RouterOutput(BaseModel):
    collections: List[str] = Field(
        ...,
        description="Names of the collections to search for this query"
    )
    token_allocation: int = Field(
        ...,
        ge=150,
        le=15000,
        description="Number of tokens to allocate for the LLM response"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the routing decision"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the routing decision (0.0 to 1.0)"
    )
