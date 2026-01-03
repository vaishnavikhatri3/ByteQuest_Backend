import re
from typing import List

# Simple trusted sources list (can be expanded later)
TRUSTED_SOURCES = [
    "wikipedia.org",
    "britannica.com",
    "who.int",
    "un.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "gov.in",
    "edu",
    "reuters.com",
    "bbc.com"
]

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text using regex
    """
    url_pattern = r"https?://[^\s\)]+"
    return re.findall(url_pattern, text)


def check_citations(paragraph: str) -> List[str]:
    """
    Check whether citations/URLs in the paragraph
    belong to trusted sources.
    
    Returns a list of verdict strings.
    """
    urls = extract_urls(paragraph)
    results = []

    if not urls:
        return []

    for url in urls:
        trusted = any(source in url for source in TRUSTED_SOURCES)
        if trusted:
            results.append(f"VALID: {url}")
        else:
            results.append(f"UNVERIFIED: {url}")

    return results
