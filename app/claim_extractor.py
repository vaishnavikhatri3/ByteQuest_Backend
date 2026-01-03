import re
from typing import List

def extract_claims(text: str) -> List[str]:
    """
    Lightweight sentence splitter without NLTK.
    Safe for Render & production.
    """
    if not text:
        return []

    # Split on . ? ! followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Remove very short/noise sentences
    sentences = [s for s in sentences if len(s.strip()) > 5]

    return sentences if sentences else [text]
