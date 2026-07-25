"""
02b_pmc_download.py — Download full-text PDFs from PubMed Central.

Pipeline stage: PMC DOWNLOAD (runs after 02a_pmc_lookup.py)
Input:  data/raw/pmc_lookup.json                  — PMID -> PMCID mapping
        data/raw/works_<YYYYMMDD>_filtered.jsonl  — filtered corpus (for metadata)
Output: data/pdfs_manual/<paper_id>.pdf           — downloaded PDFs
        data/raw/pmc_download_report.json         — download statistics

What this script does:
    1. Reads the PMC lookup table from 02a_pmc_lookup.py
    2. For each PMID with a PMCID, downloads the PDF from PMC
    3. Saves PDFs to data/pdfs_manual/ using the same naming convention
       as the manual drop folder so 02_parse_grobid.py picks them up

Design principles:
    - Idempotent: skips papers already downloaded
    - Rate-limited: respects NCBI's download guidelines
    - Only downloads papers NOT already covered by OA PDF URLs
      (avoids redundant downloads)

Usage:
    python 02b_pmc_download.py

    # Limit to first N downloads (for testing)
    python 02b_pmc_download.py --limit 20
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

from config import DATA_DIR, RAW_DIR

load_dotenv_available = True
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv_available = False

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
PDFS_MANUAL_DIR = DATA_DIR / "pdfs_manual"
PDFS_MANUAL_DIR.mkdir(parents=True, exist_ok=True)

LOOKUP_PATH = RAW_DIR / "pmc_lookup.json"
REPORT_PATH = RAW_DIR / "pmc_download_report.json"

# PMC PDF URL pattern
PMC_PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

# Polite download headers
HEADERS = {
    "User-Agent": (
        "PenningtonKGPipeline/1.0 (research use; "
        "contact: research@pennington.org)"
    )
}

DOWNLOAD_TIMEOUT = 60   # PMC PDFs can be slow
RATE_LIMIT_DELAY = 0.5  # seconds between downloads — be polite to PMC


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_latest_filtered_file() -> Path:
    """Find the most recently created filtered JSONL file."""
    candidates = sorted(
        RAW_DIR.glob("works_*_filtered.jsonl"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No filtered works_*_filtered.jsonl files found in {RAW_DIR}."
        )
    return candidates[0]


def load_lookup_table() -> dict[str, str | None]:
    """Load the PMID -> PMCID lookup table."""
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"PMC lookup table not found at {LOOKUP_PATH}. "
            "Run 02a_pmc_lookup.py first."
        )
    with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pmid_to_record(input_path: Path) -> dict[str, dict]:
    """
    Build a mapping from PMID to full record for enriching download metadata.

    Args:
        input_path: Path to filtered JSONL file.

    Returns:
        Dict mapping PMID -> record dict.
    """
    pmid_map = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pmid = record.get("pmid", "").strip()
                if pmid:
                    pmid_map[pmid] = record
            except json.JSONDecodeError:
                continue
    return pmid_map


def get_paper_id(record: dict) -> str:
    """Get stable paper ID for filename, preferring PMID."""
    if record.get("pmid"):
        return f"pmid_{record['pmid']}"
    if record.get("doi"):
        return "doi_" + record["doi"].replace("/", "_").replace(".", "_")
    return record["openalex_id"].replace("https://openalex.org/", "")


def already_has_oa_pdf(record: dict) -> bool:
    """Check if a paper already has an OA PDF URL (avoid redundant download)."""
    return bool(record.get("pdf_url"))


def download_pmc_pdf(pmcid: str, dest_path: Path) -> bool:
    """
    Download a PDF from PubMed Central.

    PMC serves PDFs at a predictable URL pattern. Some articles may only
    have HTML full text — in that case the download will fail gracefully.

    Args:
        pmcid: PMC ID string (e.g. 'PMC1234567').
        dest_path: Local path to save the PDF.

    Returns:
        True if download succeeded and file is a valid PDF.
    """
    url = PMC_PDF_URL.format(pmcid=pmcid)

    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )

        if r.status_code != 200:
            log.debug(f"PMC download failed: HTTP {r.status_code} for {pmcid}")
            return False

        # Write file
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify it's a valid PDF
        with open(dest_path, "rb") as f:
            header = f.read(4)

        if header != b"%PDF":
            log.debug(f"{pmcid}: downloaded file is not a PDF (may be HTML only)")
            dest_path.unlink(missing_ok=True)
            return False

        return True

    except requests.exceptions.RequestException as e:
        log.debug(f"PMC download error for {pmcid}: {e}")
        dest_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def run_download(
    input_path: Path,
    limit: int | None = None,
) -> dict:
    """
    Download PMC PDFs for all papers in the lookup table with PMCIDs.

    Args:
        input_path: Path to filtered JSONL file.
        limit: Optional max number of downloads (for testing).

    Returns:
        Report statistics dict.
    """
    lookup_table = load_lookup_table()
    pmid_to_record = build_pmid_to_record(input_path)

    # Build download queue: PMIDs with PMCIDs that don't already have OA URLs
    download_queue = []
    skipped_has_oa = 0

    for pmid, pmcid in lookup_table.items():
        if pmcid is None:
            continue  # No PMC version

        record = pmid_to_record.get(pmid)
        if not record:
            continue  # PMID not in current corpus

        # Skip if already has an OA PDF URL — don't duplicate effort
        if already_has_oa_pdf(record):
            skipped_has_oa += 1
            continue

        paper_id = get_paper_id(record)
        dest_path = PDFS_MANUAL_DIR / f"{paper_id}.pdf"

        download_queue.append({
            "pmid": pmid,
            "pmcid": pmcid,
            "paper_id": paper_id,
            "dest_path": dest_path,
            "record": record,
        })

    log.info(f"Download queue: {len(download_queue)} papers")
    log.info(f"Skipped (already have OA URL): {skipped_has_oa}")

    if limit:
        download_queue = download_queue[:limit]
        log.info(f"Limiting to first {limit} downloads")

    stats = {
        "total_queued": len(download_queue),
        "downloaded": 0,
        "skipped_exists": 0,
        "failed": 0,
        "skipped_has_oa": skipped_has_oa,
    }

    for item in tqdm(download_queue, desc="Downloading from PMC", unit="paper"):
        dest_path = item["dest_path"]

        # Skip if already downloaded
        if dest_path.exists():
            stats["skipped_exists"] += 1
            continue

        success = download_pmc_pdf(item["pmcid"], dest_path)

        if success:
            stats["downloaded"] += 1
            log.debug(
                f"Downloaded {item['pmcid']} → {dest_path.name}"
            )
        else:
            stats["failed"] += 1

        time.sleep(RATE_LIMIT_DELAY)

        # Log progress every 100 downloads
        if (stats["downloaded"] + stats["failed"]) % 100 == 0:
            log.info(
                f"Progress: {stats['downloaded']} downloaded | "
                f"{stats['failed']} failed"
            )

    # Save report
    report = {
        "run_at": datetime.utcnow().isoformat(),
        "input_file": str(input_path.name),
        "total_queued": stats["total_queued"],
        "downloaded": stats["downloaded"],
        "skipped_already_exists": stats["skipped_exists"],
        "skipped_has_oa_url": stats["skipped_has_oa"],
        "failed": stats["failed"],
        "success_rate_pct": round(
            100 * stats["downloaded"] / stats["total_queued"], 2
        ) if stats["total_queued"] > 0 else 0,
        "pdfs_location": str(PDFS_MANUAL_DIR),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PMC PDFs using PMCID lookup table."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to filtered JSONL file. Defaults to most recent.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only download first N PDFs (useful for testing).",
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

    report = run_download(input_path, limit=args.limit)

    print("\n" + "=" * 60)
    print("PMC DOWNLOAD REPORT")
    print("=" * 60)
    print(f"Total queued             : {report['total_queued']:,}")
    print(f"Successfully downloaded  : {report['downloaded']:,}")
    print(f"Skipped (already exists) : {report['skipped_already_exists']:,}")
    print(f"Skipped (has OA URL)     : {report['skipped_has_oa_url']:,}")
    print(f"Failed                   : {report['failed']:,}")
    print(f"Success rate             : {report['success_rate_pct']}%")
    print(f"\nPDFs saved to           : {report['pdfs_location']}")
    print("=" * 60)
