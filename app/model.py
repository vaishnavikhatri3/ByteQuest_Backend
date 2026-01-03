import os
import requests
from typing import Tuple

HF_API_URL = "https://api-inference.huggingface.co/models/typeform/distilbert-base-uncased-mnli"
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# Hugging Face MNLI label mapping
LABEL_MAP = {
    "LABEL_0": "CONTRADICTION",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "ENTAILMENT"
}

class ClaimVerifier:
    def verify(self, document: str, claim: str) -> Tuple[str, float]:
        if not HF_API_KEY:
            return "ERROR", 0.0

        # ✅ CORRECT MNLI INPUT FORMAT
        payload = {
            "inputs": [
                document,
                claim
            ]
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

            # Expected: List[Dict]
            if not isinstance(result, list):
                return "ERROR", 0.0

            scores = {}
            for item in result:
                label = LABEL_MAP.get(item.get("label"))
                if label:
                    scores[label] = item.get("score", 0.0)

            entailment_score = scores.get("ENTAILMENT", 0.0)

            label = "SUPPORTED" if entailment_score >= 0.5 else "HALLUCINATED"
            return label, round(entailment_score, 3)

        except Exception:
            return "ERROR", 0.0
