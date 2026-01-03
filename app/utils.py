def calculate_trust_score(claim_results):
    if not claim_results:
        return 0.0
    supported = sum(1 for c in claim_results if c.label == "SUPPORTED")
    return round(supported / len(claim_results), 2)
