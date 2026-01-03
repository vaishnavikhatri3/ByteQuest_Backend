from duckduckgo_search import DDGS
from app.trusted_sources import TRUSTED_DOMAINS

def search_trusted_sources(query: str, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href", "")
            if any(domain in url for domain in TRUSTED_DOMAINS):
                results.append(url)
    return results
