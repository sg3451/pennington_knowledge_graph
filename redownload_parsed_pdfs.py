# save as redownload_parsed_pdfs.py
#download PDFs that were deleted by earlier GROBID code
import requests
import json
from pathlib import Path
from config import RAW_DIR, DATA_DIR

PDFS_OA_DIR = DATA_DIR / "pdfs_oa"
PDFS_OA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
}

# Load manifest to find successfully parsed papers
from config import PARSED_DIR
manifest_path = PARSED_DIR / "parse_manifest.json"
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

# Get OA-downloaded papers (not manual)
oa_papers = {
    pid: entry for pid, entry in manifest.items()
    if entry.get("status") == "success"
    and entry.get("source") == "oa_download"
}
print(f"Papers to re-download: {len(oa_papers)}")

# Load URL lookup from filtered JSONL
candidates = sorted(RAW_DIR.glob("works_*_filtered.jsonl"), reverse=True)
url_lookup = {}
with open(candidates[0], encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pid = None
        if r.get("pmid"):
            pid = f"pmid_{r['pmid']}"
        elif r.get("doi"):
            pid = "doi_" + r["doi"].replace("/", "_").replace(".", "_")
        if pid and r.get("pdf_url"):
            url_lookup[pid] = r["pdf_url"]

downloaded = 0
failed = 0
for pid, entry in oa_papers.items():
    dest = PDFS_OA_DIR / f"{pid}.pdf"
    if dest.exists():
        continue
    url = url_lookup.get(pid)
    if not url:
        continue
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", ""):
            dest.write_bytes(r.content)
            downloaded += 1
            print(f"  ✓ {pid}")
        else:
            failed += 1
    except Exception:
        failed += 1

print(f"\nDownloaded: {downloaded} | Failed: {failed}")
print(f"PDFs saved to: {PDFS_OA_DIR}")