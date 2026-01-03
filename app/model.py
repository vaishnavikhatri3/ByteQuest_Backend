import os
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/typeform/distilbert-base-uncased-mnli"
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

class ClaimVerifier:
    def verify(self, document: str, claim: str):
        """
        Verify a claim against a document using MNLI (Entailment).
        Returns: (label, confidence)
        """

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
            result = response.json()

            if isinstance(result, list):
                scores = {item["label"]: item["score"] for item in result[0]}
                entailment_score = scores.get("ENTAILMENT", 0.0)
            else:
                entailment_score = 0.0

            label = "SUPPORTED" if entailment_score > 0.5 else "HALLUCINATED"
            return label, round(entailment_score, 3)

        except Exception:
            return "ERROR", 0.0
