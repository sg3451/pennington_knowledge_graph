"""
01_ingest.py — Pull the full Pennington Biomedical publication corpus from OpenAlex.

Pipeline stage: INGEST
Input:  OpenAlex API (filtered by Pennington ROR ID)
Output: data/raw/works_<YYYYMMDD>.jsonl  — one JSON record per line
        data/raw/ingest_manifest.json    — run metadata (count, date, cursor state)

Design principles followed:
- Idempotent: re-running on the same date overwrites the same output file safely.
- Incremental: pass --since YYYY-MM-DD to pull only new/updated works.
- No raw text sent externally: only metadata and abstracts (publicly indexed) fetched.
- Cursor pagination used throughout (safer than offset for large result sets).

Usage:
    # Full corpus pull (first run)
    python 01_ingest.py

    # Incremental update (weekly cron)
    python 01_ingest.py --since 2025-01-01
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from itertools import chain
from pathlib import Path

import pyalex
from pyalex import Works

from config import (
    DATA_DIR,
    INGEST_MAX_RESULTS,
    INGEST_PAGE_SIZE,
    OPENALEX_API_KEY,
    OPENALEX_EMAIL,
    OPENALEX_SELECT_FIELDS,
    PENNINGTON_ROR_ID,
    RAW_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAlex client configuration
# ---------------------------------------------------------------------------

def configure_pyalex() -> None:
    """
    Set pyalex credentials and retry behaviour.

    Uses the polite pool (faster, more consistent rate limits) when an
    email is configured. API key is required as of February 2026.
    """
    if not OPENALEX_API_KEY:
        log.warning(
            "OPENALEX_API_KEY not set. You are limited to 100 credits/day. "
            "Get a free key at https://openalex.org/accounts/signup"
        )
    else:
        pyalex.config.api_key = OPENALEX_API_KEY

    if OPENALEX_EMAIL:
        pyalex.config.email = OPENALEX_EMAIL
    else:
        log.warning(
            "OPENALEX_EMAIL not set. Set it in .env for polite pool access."
        )

    # Retry on transient errors with exponential backoff
    pyalex.config.max_retries = 10
    pyalex.config.retry_backoff_factor = 2.0
    pyalex.config.retry_http_codes = [429, 500, 503]


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_query(since_date: str | None = None) -> Works:
    """
    Build a pyalex Works query filtered to Pennington Biomedical publications.

    Args:
        since_date: ISO date string (YYYY-MM-DD). If provided, filters to works
                    with from_updated_date >= since_date for incremental updates.

    Returns:
        A pyalex Works query object ready for pagination.
    """
    query = (
        Works()
        .filter(authorships={"institutions": {"ror": PENNINGTON_ROR_ID}})
        .select(OPENALEX_SELECT_FIELDS)
        .sort(publication_date="desc")
    )

    if since_date:
        log.info(f"Incremental mode: fetching works updated since {since_date}")
        query = query.filter(from_updated_date=since_date)
    else:
        log.info("Full corpus mode: fetching all Pennington works")

    return query


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------

def normalize_work(work: dict) -> dict:
    """
    Flatten and normalize a raw OpenAlex Work record for downstream processing.

    Extracts the fields most relevant to the KG schema and resolves the
    abstract from the inverted index to plaintext (pyalex handles this
    automatically when accessing work['abstract']).

    Args:
        work: Raw dict returned by pyalex for a single Work.

    Returns:
        Normalized dict with consistent keys and resolved identifiers.
    """
    # External IDs
    doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()

    # PMID is nested inside the ids dict which comes back automatically
    # when not using select — but since we use select, extract from doi field
    # OpenAlex returns pmid as a separate field when explicitly requested
    ids = work.get("ids", {})
    pmid = str(ids.get("pmid") or "").replace(
    "https://pubmed.ncbi.nlm.nih.gov/", "")
    pmcid = str(ids.get("pmcid") or "").replace(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/", ""
)
    
    # Open access PDF URL if available
    oa = work.get("open_access", {})
    pdf_url = oa.get("oa_url") if oa.get("is_oa") else None

    # Author + affiliation list
    authorships = []
    for a in work.get("authorships", []):
        author = a.get("author", {})
        affiliations = [
            {
                "ror_id": inst.get("ror", ""),
                "institution_name": inst.get("display_name", ""),
                "openalex_id": inst.get("id", ""),
            }
            for inst in a.get("institutions", [])
        ]
        authorships.append({
            "author_name": author.get("display_name", ""),
            "openalex_author_id": author.get("id", ""),
            "orcid": (author.get("orcid") or "").replace("https://orcid.org/", ""),
            "author_position": a.get("author_position", ""),
            "is_corresponding": a.get("is_corresponding", False),
            "affiliations": affiliations,
        })

    # MeSH terms (directly from PubMed via OpenAlex)
    mesh = [
        {
            "descriptor_ui": m.get("descriptor_ui", ""),
            "descriptor_name": m.get("descriptor_name", ""),
            "qualifier_ui": m.get("qualifier_ui", ""),
            "qualifier_name": m.get("qualifier_name", ""),
            "is_major_topic": m.get("is_major_topic", False),
        }
        for m in work.get("mesh", [])
    ]

    # OpenAlex concept tags (legacy) + newer topic tags
    concepts = [
        {
            "openalex_id": c.get("id", ""),
            "display_name": c.get("display_name", ""),
            "wikidata_id": c.get("wikidata", ""),
            "score": c.get("score", 0.0),
            "level": c.get("level", 0),
        }
        for c in work.get("concepts", [])
    ]

    topics = [
        {
            "openalex_id": t.get("id", ""),
            "display_name": t.get("display_name", ""),
            "score": t.get("score", 0.0),
            "field": t.get("field", {}).get("display_name", ""),
            "domain": t.get("domain", {}).get("display_name", ""),
        }
        for t in work.get("topics", [])
    ]

    # Awards / funding
    awards = [
        {
            "funder_openalex_id": g.get("funder", ""),
            "funder_display_name": g.get("funder_display_name", ""),
            "award_id": g.get("award_id", ""),
        }
        for g in work.get("awards", [])
    ]

    # Outgoing citations (OpenAlex IDs of referenced works)
    referenced_works = work.get("referenced_works", [])

    # Journal / source info
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    return {
        # Core identifiers
        "openalex_id": work.get("id", ""),
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        # Bibliographic metadata
        "title": work.get("title", ""),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date", ""),
        "work_type": work.get("type", ""),
        "cited_by_count": work.get("cited_by_count", 0),
        # Abstract (plaintext; pyalex resolves inverted index automatically)
        "abstract": work.get("abstract", ""),
        # Source / journal
        "journal_name": source.get("display_name", ""),
        "journal_issn": source.get("issn_l", ""),
        "journal_openalex_id": source.get("id", ""),
        # Open access
        "is_oa": oa.get("is_oa", False),
        "oa_status": oa.get("oa_status", ""),
        "pdf_url": pdf_url,
        # Structured metadata for KG loading
        "authorships": authorships,
        "mesh": mesh,
        "concepts": concepts,
        "topics": topics,
        "awards": awards,
        "referenced_works": referenced_works,
        # Provenance
        "_ingested_at": datetime.utcnow().isoformat(),
        "_source": "openalex",
    }


# ---------------------------------------------------------------------------
# Main ingest loop
# ---------------------------------------------------------------------------

def run_ingest(since_date: str | None = None) -> Path:
    """
    Pull all Pennington works from OpenAlex and write to a JSONL file.

    Args:
        since_date: Optional ISO date string for incremental updates.

    Returns:
        Path to the written JSONL output file.
    """
    configure_pyalex()
    query = build_query(since_date)

    # Output file named by today's date for traceability
    today = date.today().strftime("%Y%m%d")
    out_path = RAW_DIR / f"works_{today}.jsonl"
    manifest_path = RAW_DIR / "ingest_manifest.json"

    log.info(f"Writing output to: {out_path}")

    count = 0
    error_count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        pager = query.paginate(
            method="cursor",
            per_page=INGEST_PAGE_SIZE,
            n_max=INGEST_MAX_RESULTS,
        )
        import time
        for page in pager:
            for work in page:
                try:
                    normalized = normalize_work(work)
                    f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    error_count += 1
                    log.warning(
                        f"Failed to normalize work {work.get('id', 'unknown')}: {e}"
                    )

            log.info(f"  Progress: {count} works written ({error_count} errors)...")
            time.sleep(1)  # 1 second between pages to avoid rate limiting

    log.info(f"Ingest complete: {count} works written to {out_path}")
    if error_count:
        log.warning(f"{error_count} works failed normalization — check logs above.")

    # Write manifest for incremental update tracking
    manifest = {
        "last_run": datetime.utcnow().isoformat(),
        "since_date": since_date,
        "output_file": str(out_path.name),
        "works_written": count,
        "errors": error_count,
        "ror_id": PENNINGTON_ROR_ID,
    }
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    log.info(f"Manifest written to {manifest_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest Pennington Biomedical publications from OpenAlex."
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Only fetch works updated on or after this date. "
            "Omit for a full corpus pull."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = run_ingest(since_date=args.since)
    print(f"\nDone. Output: {output_file}")
