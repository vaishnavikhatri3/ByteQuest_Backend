from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from app.model import ClaimVerifier
from app.claim_extractor import extract_claims
from app.citation_checker import check_citations

app = FastAPI(
    title="ByteQuest Backend",
    description="AI Hallucination & Citation Verification System",
    version="1.0"
)

verifier = ClaimVerifier()

# ------------------------
# HEALTH CHECK (RENDER)
# ------------------------
@app.get("/")
def health():
    return {"status": "Backend running"}

# ------------------------
# REQUEST SCHEMA
# ------------------------
class VerifyRequest(BaseModel):
    paragraph: str

# ------------------------
# RESPONSE SCHEMA
# ------------------------
class ClaimResult(BaseModel):
    claim: str
    label: str
    confidence: float

class VerifyResponse(BaseModel):
    claims: List[ClaimResult]
    citations: List[str]
    trust_score: float

# ------------------------
# MAIN VERIFY ENDPOINT
# ------------------------
@app.post("/verify", response_model=VerifyResponse)
def verify_text(data: VerifyRequest):
    paragraph = data.paragraph.strip()

    if not paragraph:
        raise HTTPException(status_code=400, detail="Paragraph cannot be empty")

    claims = extract_claims(paragraph)
    results = []

    supported_count = 0

    for claim in claims:
        label, confidence = verifier.verify(paragraph, claim)
        if label == "SUPPORTED":
            supported_count += 1

        results.append(
            ClaimResult(
                claim=claim,
                label=label,
                confidence=confidence
            )
        )

    citations = check_citations(paragraph)

    trust_score = round(
        supported_count / len(results), 2
    ) if results else 0.0

    return VerifyResponse(
        claims=results,
        citations=citations,
        trust_score=trust_score
    )
