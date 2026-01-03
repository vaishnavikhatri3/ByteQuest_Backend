import os
import requests
from typing import Tuple

# Hugging Face MNLI model (hosted inference)
HF_API_URL = "https://api-inference.huggingface.co/models/typeform/distilbert-base-uncased-mnli"
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


class ClaimVerifier:
    """
    Verifies factual claims using Natural Language Inference (MNLI).
    Returns label + confidence score.
    """

    def verify(self, document: str, claim: str) -> Tuple[str, float]:
        """
        document: full paragraph / context
        claim: single extracted sentence

        Returns:
        ("SUPPORTED" | "HALLUCINATED" | "ERROR", confidence)
        """

        # Safety check
        if not HF_API_KEY:
            return "ERROR", 0.0

        payload = {
            "inputs": {
                "premise": document,
                "hypothesis": claim
            }
        }

        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                return "ERROR", 0.0

            result = response.json()

            """
            Hugging Face API may return either:
            1) List[Dict]
               [
                 {"label": "ENTAILMENT", "score": 0.91},
                 {"label": "NEUTRAL", "score": 0.06},
                 {"label": "CONTRADICTION", "score": 0.03}
               ]

            2) List[List[Dict]]
               [
                 [
                   {"label": "ENTAILMENT", "score": 0.91},
                   {"label": "NEUTRAL", "score": 0.06},
                   {"label": "CONTRADICTION", "score": 0.03}
                 ]
               ]
            """

            scores = {}

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    scores = {item["label"]: item["score"] for item in result}
                elif isinstance(result[0], list):
                    scores = {item["label"]: item["score"] for item in result[0]}
                else:
                    return "ERROR", 0.0
            else:
                return "ERROR", 0.0

            entailment_score = scores.get("ENTAILMENT", 0.0)

            label = "SUPPORTED" if entailment_score >= 0.5 else "HALLUCINATED"

            return label, round(entailment_score, 3)

        except Exception:
            return "ERROR", 0.0
