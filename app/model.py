import requests
import os

HF_API_URL = "https://api-inference.huggingface.co/models/typeform/distilbert-base-uncased-mnli"
HF_API_KEY = os.getenv("HF_API_KEY")

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

class ClaimVerifier:
    def verify(self, document: str, claim: str):
        payload = {
            "inputs": {
                "premise": document,
                "hypothesis": claim
            }
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, list):
            scores = {item["label"]: item["score"] for item in result[0]}
            entailment_score = scores.get("ENTAILMENT", 0.0)
        else:
            entailment_score = 0.0

        label = "SUPPORTED" if entailment_score > 0.5 else "HALLUCINATED"
        return label, round(entailment_score, 3)
