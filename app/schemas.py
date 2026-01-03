from pydantic import BaseModel
from typing import List

class VerifyRequest(BaseModel):
    paragraph: str

class ClaimResult(BaseModel):
    claim: str
    label: str
    confidence: float
    source: str

class CitationResult(BaseModel):
    citation: str
    status: str

class VerifyResponse(BaseModel):
    claims: List[ClaimResult]
    citations: List[CitationResult]
    trust_score: float
