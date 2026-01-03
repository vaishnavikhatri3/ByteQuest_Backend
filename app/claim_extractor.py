import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def extract_claims(paragraph: str):
    return nltk.sent_tokenize(paragraph)
