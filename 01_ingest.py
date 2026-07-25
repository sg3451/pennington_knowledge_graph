"""
01_ingest.py — Pull Pennington Biomedical publications from OpenAlex.

Pipeline stage: INGEST
Input:  OpenAlex API (filtered by Pennington ROR ID)
Output: data/raw/works_<YYYYMMDD>.jsonl  — one JSON record per line
        data/raw/ingest_manifest.json    — run metadata

This version bypasses pyalex and calls the OpenAlex REST API directly
using requests, giving full control over rate limiting, retries, and
API key passing (as a URL parameter, which is required as of Feb 2026).

Usage:
    python 01_ingest.py                        # full corpus pull
    python 01_ingest.py --since 2026-05-01     # incremental update
"""

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

from config import (
    RAW_DIR,
    OPENALEX_API_KEY,
    OPENALEX_EMAIL,
    OPENALEX_SELECT_FIELDS,
    PENNINGTON_ROR_ID,
    INGEST_PAGE_SIZE,
    INGEST_MAX_RESULTS,
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
# Constants
# ---------------------------------------------------------------------------
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Delay between page requests
PAGE_DELAY = 1.0       # seconds between pages — stays well under rate limit
RETRY_DELAY = 30.0     # seconds to wait after a 429
MAX_RETRIES = 5        # max retries per page on 429/5xx


# ---------------------------------------------------------------------------
# API request helper
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    """Build a requests session with appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            f"PenningtonKG/1.0 (mailto:{OPENALEX_EMAIL})"
            if OPENALEX_EMAIL
            else "PenningtonKG/1.0"
        ),
        "Accept": "application/json",
    })
    return session


def get_page(session: requests.Session, params: dict) -> dict | None:
    """
    Fetch a single page from the OpenAlex API with retry logic.

    Passes the API key as a URL parameter (required since Feb 2026).
    Handles 429 (rate limit) with exponential backoff.

    Args:
        session: Requests session.
        params: Query parameters dict.

    Returns:
        Parsed JSON response dict, or None on permanent failure.
    """
    # Pass API key as URL parameter — more reliable than pyalex config
    if OPENALEX_API_KEY:
        params = {**params, "api_key": OPENALEX_API_KEY}

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(OPENALEX_WORKS_URL, params=params, timeout=30)

            if r.status_code == 200:
                return r.json()

            elif r.status_code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                log.warning(
                    f"Rate limited (429). Waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
                continue

            elif r.status_code in (500, 502, 503, 504):
                wait = RETRY_DELAY
                log.warning(
                    f"Server error {r.status_code}. Waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
                continue

            else:
                log.error(f"Unexpected HTTP {r.status_code}: {r.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            log.warning(f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            continue

        except requests.exceptions.RequestException as e:
            log.error(f"Request error: {e}")
            return None

    log.error(f"All {MAX_RETRIES} retries exhausted")
    return None


# ---------------------------------------------------------------------------
# Abstract decoding
# ---------------------------------------------------------------------------

def decode_abstract(inverted_index: dict | None) -> str:
    """
    Convert OpenAlex inverted abstract index to plaintext.

    OpenAlex stores abstracts as {word: [positions]} to avoid
    copyright issues. This reconstructs the original text.

    Args:
        inverted_index: Dict mapping word -> list of integer positions.

    Returns:
        Plaintext abstract string, or empty string if not available.
    """
    if not inverted_index:
        return ""
    try:
        pos_word = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                pos_word[pos] = word
        return " ".join(pos_word[i] for i in sorted(pos_word))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------

def normalize_work(work: dict) -> dict:
    """
    Flatten and normalize a raw OpenAlex Work record.

    Args:
        work: Raw dict from OpenAlex API.

    Returns:
        Normalized dict ready for downstream pipeline stages.
    """
    ids = work.get("ids") or {}
    doi  = (work.get("doi") or "").replace("https://doi.org/", "").lower()
    pmid = str(ids.get("pmid") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", "")
    pmcid = str(ids.get("pmcid") or "").replace(
        "https://www.ncbi.nlm.nih.gov/pmc/articles/", ""
    )

    oa = work.get("open_access") or {}
    pdf_url = oa.get("oa_url") if oa.get("is_oa") else None

    # Authorships + affiliations
    authorships = []
    for a in (work.get("authorships") or []):
        author = a.get("author") or {}
        affiliations = [
            {
                "ror_id": inst.get("ror", ""),
                "institution_name": inst.get("display_name", ""),
                "openalex_id": inst.get("id", ""),
            }
            for inst in (a.get("institutions") or [])
        ]
        authorships.append({
            "author_name": author.get("display_name", ""),
            "openalex_author_id": author.get("id", ""),
            "orcid": (author.get("orcid") or "").replace("https://orcid.org/", ""),
            "author_position": a.get("author_position", ""),
            "is_corresponding": a.get("is_corresponding", False),
            "affiliations": affiliations,
        })

    # MeSH terms
    mesh_terms = [
        {
            "descriptor_ui":   m.get("descriptor_ui", ""),
            "descriptor_name": m.get("descriptor_name", ""),
            "qualifier_ui":    m.get("qualifier_ui", ""),
            "qualifier_name":  m.get("qualifier_name", ""),
            "is_major_topic":  m.get("is_major_topic", False),
        }
        for m in (work.get("mesh") or [])
    ]

    # Topics (preferred over deprecated concepts)
    topics = [
        {
            "openalex_id":  t.get("id", ""),
            "display_name": t.get("display_name", ""),
            "score":        t.get("score", 0.0),
            "field":        (t.get("field") or {}).get("display_name", ""),
            "domain":       (t.get("domain") or {}).get("display_name", ""),
        }
        for t in (work.get("topics") or [])
    ]

    # Grants / funding
    grants = [
        {
            "funder_openalex_id":  g.get("funder", ""),
            "funder_display_name": g.get("funder_display_name", ""),
            "award_id":            g.get("award_id", ""),
        }
        for g in (work.get("awards") or [])
    ]

    # Journal / source
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    # Abstract — decode from inverted index
    abstract = decode_abstract(work.get("abstract_inverted_index"))

    return {
        # Core identifiers
        "openalex_id":      work.get("id", ""),
        "doi":              doi,
        "pmid":             pmid,
        "pmcid":            pmcid,
        # Bibliographic metadata
        "title":            work.get("title", ""),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date", ""),
        "work_type":        work.get("type", ""),
        "cited_by_count":   work.get("cited_by_count", 0),
        # Abstract
        "abstract":         abstract,
        # Journal
        "journal_name":         source.get("display_name", ""),
        "journal_issn":         source.get("issn_l", ""),
        "journal_openalex_id":  source.get("id", ""),
        # Open access
        "is_oa":     oa.get("is_oa", False),
        "oa_status": oa.get("oa_status", ""),
        "pdf_url":   pdf_url,
        # Structured metadata
        "authorships":      authorships,
        "mesh":             mesh_terms,
        "concepts":         [],   # deprecated by OpenAlex; topics preferred
        "topics":           topics,
        "awards":           grants,
        "referenced_works": work.get("referenced_works") or [],
        # Provenance
        "_ingested_at": datetime.utcnow().isoformat(),
        "_source":      "openalex",
    }


# ---------------------------------------------------------------------------
# Main ingest loop
# ---------------------------------------------------------------------------

def run_ingest(since_date: str | None = None) -> Path:
    """
    Pull all Pennington works from OpenAlex and write to JSONL.

    Uses cursor-based pagination directly via requests.

    Args:
        since_date: Optional ISO date string for incremental updates.

    Returns:
        Path to the written JSONL output file.
    """
    if not OPENALEX_API_KEY:
        log.warning(
            "OPENALEX_API_KEY not set in .env. "
            "API key required since Feb 2026. "
            "Get a free key at https://openalex.org/accounts/signup"
        )
    else:
        log.info(f"Using OpenAlex API key: {OPENALEX_API_KEY[:6]}...")

    if OPENALEX_EMAIL:
        log.info(f"Polite pool email: {OPENALEX_EMAIL}")

    # Build filter string
    filters = [f"authorships.institutions.ror:{PENNINGTON_ROR_ID}"]
    if since_date:
        filters.append(f"from_updated_date:{since_date}")
        log.info(f"Incremental mode: fetching works updated since {since_date}")
    else:
        log.info("Full corpus mode: fetching all Pennington works")

    filter_str = ",".join(filters)

    # Output paths
    today = date.today().strftime("%Y%m%d")
    out_path = RAW_DIR / f"works_{today}.jsonl"
    manifest_path = RAW_DIR / "ingest_manifest.json"
    log.info(f"Writing output to: {out_path}")

    session = build_session()
    count = 0
    error_count = 0
    cursor = "*"

    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            params = {
                "filter":   filter_str,
                "select":   OPENALEX_SELECT_FIELDS,
                "sort":     "publication_date:desc",
                "per-page": INGEST_PAGE_SIZE,
                "cursor":   cursor,
            }

            data = get_page(session, params)
            if data is None:
                log.error("Failed to fetch page — stopping ingest")
                break

            results = data.get("results", [])
            if not results:
                log.info("No more results — ingest complete")
                break

            for work in results:
                try:
                    normalized = normalize_work(work)
                    f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    error_count += 1
                    log.warning(
                        f"Failed to normalize {work.get('id', 'unknown')}: {e}"
                    )

            # Pagination metadata
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")

            log.info(
                f"  Progress: {count:,} works written "
                f"| total matching: {meta.get('count', '?'):,} "
                f"| errors: {error_count}"
            )

            if not next_cursor:
                log.info("No next cursor — reached end of results")
                break

            if INGEST_MAX_RESULTS and count >= INGEST_MAX_RESULTS:
                log.info(f"Reached max results limit ({INGEST_MAX_RESULTS})")
                break

            cursor = next_cursor
            time.sleep(PAGE_DELAY)

    log.info(f"Ingest complete: {count:,} works written to {out_path}")
    if error_count:
        log.warning(f"{error_count} works failed normalization")

    # Write manifest
    manifest = {
        "last_run":     datetime.utcnow().isoformat(),
        "since_date":   since_date,
        "output_file":  out_path.name,
        "works_written": count,
        "errors":       error_count,
        "ror_id":       PENNINGTON_ROR_ID,
    }
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    log.info(f"Manifest written to {manifest_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Pennington Biomedical publications from OpenAlex."
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only fetch works updated on or after this date.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = run_ingest(since_date=args.since)
    print(f"\nDone. Output: {output_file}")
