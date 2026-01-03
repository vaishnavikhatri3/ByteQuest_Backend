import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ClaimVerifier:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-mnli"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

    def verify(self, document: str, claim: str):
        inputs = self.tokenizer(
            document,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        entailment_score = probs[0][2].item()
        label = "SUPPORTED" if entailment_score > 0.5 else "HALLUCINATED"

        return label, round(entailment_score, 3)
