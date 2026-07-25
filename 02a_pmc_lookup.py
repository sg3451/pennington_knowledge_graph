"""
02a_pmc_lookup.py — Batch PMID to PMCID lookup using NCBI E-utilities.

Pipeline stage: PMC LOOKUP (runs after 01b_filter.py, before 02b_pmc_download.py)
Input:  data/raw/works_<YYYYMMDD>_filtered.jsonl  — filtered corpus
Output: data/raw/pmc_lookup.json                  — PMID -> PMCID mapping table
        data/raw/pmc_lookup_report.json           — coverage statistics

What this script does:
    1. Extracts all PMIDs from the filtered corpus
    2. Queries NCBI E-utilities elink endpoint individually per PMID
       (individual queries are more reliable than batch mode for this endpoint)
    3. Saves a lookup table mapping PMID -> PMCID for papers with PMC versions
    4. Reports coverage statistics

Design principles:
    - Idempotent: safe to re-run; merges new results with existing lookup table
    - Resumable: saves progress every 500 PMIDs in case of interruption
    - Rate-limited: respects NCBI's limits (3 req/sec without key, 10 with key)

NCBI API key (optional but recommended for 3x faster processing):
    Add NCBI_API_KEY=your_key to your .env file
    Get a free key at: https://www.ncbi.nlm.nih.gov/account/

Usage:
    python 02a_pmc_lookup.py

    # Use a specific filtered file
    python 02a_pmc_lookup.py --input data/raw/works_20260429_filtered.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm
from dotenv import load_dotenv

from config import RAW_DIR

load_dotenv()

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
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# NCBI rate limits:
# Without API key: 3 requests/second -> delay 0.34s
# With API key:    10 requests/second -> delay 0.11s
RATE_LIMIT_DELAY = 0.11 if NCBI_API_KEY else 0.34

NCBI_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

LOOKUP_PATH = RAW_DIR / "pmc_lookup.json"
REPORT_PATH = RAW_DIR / "pmc_lookup_report.json"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def find_latest_filtered_file() -> Path:
    """Find the most recently created filtered JSONL file."""
    candidates = sorted(
        RAW_DIR.glob("works_*_filtered.jsonl"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No filtered works_*_filtered.jsonl files found in {RAW_DIR}. "
            "Run 01b_filter.py first."
        )
    return candidates[0]


def extract_pmids(input_path: Path) -> list[str]:
    """
    Extract all PMIDs from the filtered corpus.

    Args:
        input_path: Path to filtered JSONL file.

    Returns:
        List of PMID strings for records that have one.
    """
    pmids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pmid = record.get("pmid", "").strip()
                if pmid:
                    pmids.append(pmid)
            except json.JSONDecodeError:
                continue

    log.info(f"Found {len(pmids)} records with PMIDs")
    return pmids


# ---------------------------------------------------------------------------
# NCBI E-utilities lookup — individual queries for reliability
# ---------------------------------------------------------------------------

def lookup_pmcid_single(pmid: str) -> str | None:
    """
    Query NCBI elink for a single PMID to find its PMCID.

    Individual queries are significantly more reliable than batch mode
    for the elink endpoint — batch mode silently drops results.

    Args:
        pmid: Single PMID string.

    Returns:
        PMCID string (e.g. 'PMC1234567') if found, None otherwise.
    """
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    try:
        r = requests.get(
            NCBI_ELINK_URL,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        linksets = data.get("linksets", [])
        for linkset in linksets:
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("dbto") == "pmc":
                    links = linksetdb.get("links", [])
                    if links:
                        return f"PMC{links[0]}"
        return None

    except (requests.exceptions.RequestException,
            json.JSONDecodeError, KeyError, IndexError) as e:
        log.debug(f"NCBI lookup error for PMID {pmid}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main lookup loop
# ---------------------------------------------------------------------------

def run_lookup(input_path: Path) -> dict:
    """
    Run individual PMID -> PMCID lookups for all papers in the filtered corpus.

    Args:
        input_path: Path to filtered JSONL file.

    Returns:
        Report statistics dict.
    """
    # Load existing lookup table if present (enables resuming interrupted runs)
    if LOOKUP_PATH.exists():
        with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
            lookup_table = json.load(f)
        log.info(f"Loaded existing lookup table: {len(lookup_table)} entries")
    else:
        lookup_table = {}

    # Extract PMIDs from corpus
    all_pmids = extract_pmids(input_path)

    # Skip PMIDs already checked in a previous run
    pmids_to_check = [p for p in all_pmids if p not in lookup_table]
    already_cached = len(all_pmids) - len(pmids_to_check)

    log.info(
        f"PMIDs to check: {len(pmids_to_check):,} "
        f"({already_cached:,} already cached from previous run)"
    )

    if NCBI_API_KEY:
        estimated_min = len(pmids_to_check) // 10 // 60 + 1
        log.info(
            f"NCBI API key found — rate limit: 10 req/sec "
            f"(estimated time: ~{estimated_min} min)"
        )
    else:
        estimated_min = len(pmids_to_check) // 3 // 60 + 1
        log.info(
            f"No NCBI API key — rate limit: 3 req/sec "
            f"(estimated time: ~{estimated_min} min). "
            "Add NCBI_API_KEY to .env for 3x faster processing."
        )

    if not pmids_to_check:
        log.info("All PMIDs already checked — nothing to do.")
    else:
        found_count = sum(1 for v in lookup_table.values() if v is not None)

        for i, pmid in enumerate(tqdm(
            pmids_to_check,
            desc="NCBI PMC lookup",
            unit="pmid",
        )):
            result = lookup_pmcid_single(pmid)
            lookup_table[pmid] = result

            if result:
                found_count += 1

            # Save progress every 500 PMIDs
            if (i + 1) % 500 == 0:
                with open(LOOKUP_PATH, "w", encoding="utf-8") as f:
                    json.dump(lookup_table, f, indent=2)
                log.info(
                    f"Progress saved: {i + 1:,} checked this run | "
                    f"{found_count:,} PMC versions found so far"
                )

            time.sleep(RATE_LIMIT_DELAY)

    # Final save
    with open(LOOKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup_table, f, indent=2)
    log.info("Lookup table saved.")

    # Build report
    has_pmcid = {k: v for k, v in lookup_table.items() if v is not None}
    no_pmcid = {k: v for k, v in lookup_table.items() if v is None}

    report = {
        "run_at": datetime.utcnow().isoformat(),
        "input_file": str(input_path.name),
        "total_pmids_in_corpus": len(all_pmids),
        "total_pmids_checked": len(lookup_table),
        "has_pmc_version": len(has_pmcid),
        "no_pmc_version": len(no_pmcid),
        "pmc_coverage_pct": round(
            100 * len(has_pmcid) / len(lookup_table), 2
        ) if lookup_table else 0,
        "sample_pmcids": list(has_pmcid.items())[:10],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Individual PMID -> PMCID lookup via NCBI E-utilities."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to filtered JSONL file. Defaults to most recent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            log.error(f"Input file not found: {input_path}")
            sys.exit(1)
    else:
        try:
            input_path = find_latest_filtered_file()
            log.info(f"Auto-detected input: {input_path.name}")
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(1)

    report = run_lookup(input_path)

    print("\n" + "=" * 60)
    print("PMC LOOKUP REPORT")
    print("=" * 60)
    print(f"Total PMIDs in corpus    : {report['total_pmids_in_corpus']:,}")
    print(f"Total PMIDs checked      : {report['total_pmids_checked']:,}")
    print(f"Has PMC version          : {report['has_pmc_version']:,}")
    print(f"No PMC version           : {report['no_pmc_version']:,}")
    print(f"PMC coverage             : {report['pmc_coverage_pct']}%")
    print(f"\nSample PMCIDs found:")
    for pmid, pmcid in report["sample_pmcids"]:
        print(f"  PMID {pmid} -> {pmcid}")
    print(f"\nLookup table saved to    : {LOOKUP_PATH}")
    print(f"Full report saved to     : {REPORT_PATH}")
    print("=" * 60)
