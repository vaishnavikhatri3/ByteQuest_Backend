import nltk
nltk.download("punkt")

def extract_claims(paragraph: str):
    sentences = nltk.sent_tokenize(paragraph)
    return [s for s in sentences if len(s.split()) > 4]
