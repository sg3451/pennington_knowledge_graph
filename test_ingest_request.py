# save as test_ingest_request.py
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY", "")
EMAIL   = os.getenv("OPENALEX_EMAIL", "")

# Test the exact same params that 01_ingest.py uses
params = {
    "filter":   "authorships.institutions.ror:https://ror.org/040cnym54",
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

print("Sending request...")
print(f"URL: https://api.openalex.org/works")
print(f"Filter: {params['filter']}")

r = requests.get(
    "https://api.openalex.org/works",
    params=params,
    headers=headers,
    timeout=30,
)

print(f"\nStatus: {r.status_code}")
print(f"Final URL: {r.url[:120]}...")

for k, v in r.headers.items():
    if "ratelimit" in k.lower():
        print(f"  {k}: {v}")

if r.status_code == 200:
    data = r.json()
    meta = data.get("meta", {})
    print(f"\nTotal matching: {meta.get('count', '?'):,}")
    print(f"Results on this page: {len(data.get('results', []))}")
    print(f"Next cursor: {meta.get('next_cursor', 'none')[:30]}...")
elif r.status_code == 429:
    print(f"\nResponse body: {r.text[:300]}")