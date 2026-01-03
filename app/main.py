from fastapi import FastAPI
from app.schemas import VerifyRequest, VerifyResponse, ClaimResult, CitationResult
from app.claim_extractor import extract_claims
from app.search_engine import search_trusted_sources
from app.scraper import scrape_page
from app.model import ClaimVerifier
from app.citation_checker import check_citation
from app.utils import calculate_trust_score

app = FastAPI(title="AI Hallucination & Citation Verification API")

verifier = ClaimVerifier()

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/verify", response_model=VerifyResponse)
def verify_ai_output(data: VerifyRequest):
    claims = extract_claims(data.paragraph)

    claim_results = []
    citation_results = []

    for claim in claims:
        sources = search_trusted_sources(claim)
        document = ""
        source_used = "None"

        if sources:
            source_used = sources[0]
            document = scrape_page(source_used)

        label, confidence = verifier.verify(document, claim)

        claim_results.append(
            ClaimResult(
                claim=claim,
                label=label,
                confidence=confidence,
                source=source_used
            )
        )

        if source_used != "None":
            citation_results.append(
                CitationResult(
                    citation=source_used,
                    status=check_citation(source_used)
                )
            )

    trust_score = calculate_trust_score(claim_results)

    return VerifyResponse(
        claims=claim_results,
        citations=citation_results,
        trust_score=trust_score
    )
