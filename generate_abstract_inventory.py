# save as generate_abstract_inventory.py
# list of PMIDs for which a PDF is available or not
import json
import csv
from pathlib import Path
from config import RAW_DIR, PARSED_DIR

candidates = sorted(RAW_DIR.glob("works_*_filtered.jsonl"), reverse=True)
input_path = candidates[0]

# Load parse manifest to check what has been parsed
manifest_path = PARSED_DIR / "parse_manifest.json"
if manifest_path.exists():
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
else:
    manifest = {}

# Build paper_id from record (same logic as 02_parse_grobid.py)
def get_paper_id(record):
    if record.get("pmid"):
        return f"pmid_{record['pmid']}"
    if record.get("doi"):
        return "doi_" + record["doi"].replace("/", "_").replace(".", "_")
    return record["openalex_id"].replace("https://openalex.org/", "")

rows = []
with open(input_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)

        paper_id = get_paper_id(r)
        manifest_entry = manifest.get(paper_id, {})
        parse_status = manifest_entry.get("status", "not_attempted")

        # Simplify status for readability
        if parse_status == "success":
            pdf_status = "yes"
        elif parse_status in ("download_failed", "grobid_failed",
                               "skipped_non_pdf_url"):
            pdf_status = "no"
        else:
            pdf_status = "not_attempted"

        # Check if PDF file actually exists in parsed dir
        tei_exists = (PARSED_DIR / f"{paper_id}.tei.xml").exists()
        if tei_exists:
            pdf_status = "yes"

        rows.append({
            "pmid": r.get("pmid", ""),
            "doi": r.get("doi", ""),
            "openalex_id": r.get("openalex_id", ""),
            "title": r.get("title", "")[:200],
            "year": r.get("publication_year", ""),
            "journal": r.get("journal_name", "")[:80],
            "work_type": r.get("work_type", ""),
            "cited_by_count": r.get("cited_by_count", 0),
            "has_oa_pdf_url": "yes" if r.get("pdf_url") else "no",
            "full_text_parsed": pdf_status,
            "paper_id": paper_id,
        })

# Sort by cited_by_count descending so high-impact papers are at top
rows.sort(key=lambda x: x["cited_by_count"], reverse=True)

# Write CSV
output_path = RAW_DIR / "abstract_inventory.csv"
with open(output_path, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Summary stats
total = len(rows)
has_pmid = sum(1 for r in rows if r["pmid"])
has_doi = sum(1 for r in rows if r["doi"])
full_text_yes = sum(1 for r in rows if r["full_text_parsed"] == "yes")
has_oa_url = sum(1 for r in rows if r["has_oa_pdf_url"] == "yes")

print(f"\nAbstract inventory written to: {output_path}")
print(f"\nSummary:")
print(f"  Total records          : {total:,}")
print(f"  Has PMID               : {has_pmid:,}")
print(f"  Has DOI                : {has_doi:,}")
print(f"  Has OA PDF URL         : {has_oa_url:,}")
print(f"  Full text parsed       : {full_text_yes:,}")
print(f"  Full text missing      : {total - full_text_yes:,}")
print(f"\nCSV columns: pmid, doi, openalex_id, title, year, journal,")
print(f"             work_type, cited_by_count, has_oa_pdf_url, full_text_parsed")