# save as test_since_filter.py
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY", "")
EMAIL   = os.getenv("OPENALEX_EMAIL", "")

params = {
    "filter":   "authorships.institutions.ror:https://ror.org/040cnym54,from_updated_date:2026-05-20",
    "select":   "doi,ids,title,publication_year,publication_date,type,open_access,authorships,cited_by_count,biblio,primary_location,concepts,topics,keywords,mesh,awards,referenced_works,abstract_inverted_index",
    "sort":     "publication_date:desc",
    "per-page": 5,
    "cursor":   "*",
    "api_key":  API_KEY,
}

headers = {
    "User-Agent": f"PenningtonKG/1.0 (mailto:{EMAIL})",
    "Accept": "application/json",
}

r = requests.get(
    "https://api.openalex.org/works",
    params=params,
    headers=headers,
    timeout=30,
)

print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    meta = data.get("meta", {})
    print(f"Total matching: {meta.get('count', '?'):,}")
    print(f"Results: {len(data.get('results', []))}")
    print("SUCCESS — since filter works fine")
elif r.status_code == 429:
    print(f"429 received")
    print(f"Response: {r.text[:300]}")
    print(f"\nHeaders:")
    for k, v in r.headers.items():
        if "ratelimit" in k.lower():
            print(f"  {k}: {v}")