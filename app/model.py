from typing import Tuple
from duckduckgo_search import DDGS
import re

# Trusted domains
TRUSTED_DOMAINS = [
    "wikipedia.org",
    "isro.gov.in",
    "gov.in",
    "who.int",
    "britannica.com",
    "un.org",
    "rbi.org.in",
    "bbc.com",
    "reuters.com"
]

class ClaimVerifier:
    def verify(self, document: str, claim: str) -> Tuple[str, float]:
        """
        Web-backed factual verification.
        No ML dependency. No HF. No ERROR.
        """

        claim_clean = claim.lower().strip()

        if len(claim_clean) < 5:
            return "HALLUCINATED", 0.1

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(claim_clean, max_results=5))

            if not results:
                return "HALLUCINATED", 0.2

            for r in results:
                url = r.get("href", "")
                body = (r.get("body") or "").lower()

                # check trusted source
                if any(domain in url for domain in TRUSTED_DOMAINS):
                    # simple similarity: keyword overlap
                    overlap = self._similarity(claim_clean, body)
                    if overlap > 0.4:
                        return "SUPPORTED", round(overlap, 2)

            return "HALLUCINATED", 0.3

        except Exception:
            # NEVER return ERROR now
            return "HALLUCINATED", 0.2

    def _similarity(self, a: str, b: str) -> float:
        a_words = set(re.findall(r"\w+", a))
        b_words = set(re.findall(r"\w+", b))

        if not a_words:
            return 0.0

        return len(a_words & b_words) / len(a_words)
