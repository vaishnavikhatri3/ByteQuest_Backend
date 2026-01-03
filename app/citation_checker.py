import requests

def check_citation(citation: str):
    try:
        res = requests.head(citation, timeout=5, allow_redirects=True)
        return "VALID" if res.status_code == 200 else "BROKEN"
    except Exception:
        return "FAKE"
