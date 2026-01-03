import nltk

def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

def extract_claims(text: str):
    ensure_punkt()
    return nltk.sent_tokenize(text)
