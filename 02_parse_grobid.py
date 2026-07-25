"""
02_parse_grobid.py — Download open-access PDFs and parse with GROBID.

Pipeline stage: PARSE (runs after 01b_filter.py)
Input:  data/raw/works_<YYYYMMDD>_filtered.jsonl  — filtered corpus
Output: data/parsed/<paper_id>.tei.xml            — one TEI XML file per paper
        data/parsed/<paper_id>.sections.json      — structured section extract
        data/parsed/parse_manifest.json           — run metadata and status

URL handling (v4):
    Skipped URL categories (confirmed non-downloadable):
        - DOI redirects (doi.org, dx.doi.org)
        - GitHub repositories
        - PMC HTML article pages (pmc.ncbi.nlm.nih.gov/articles/,
          www.ncbi.nlm.nih.gov/pmc/articles/) — serve HTML not PDF
        - LSU institutional repositories (repository.lsu.edu,
          digitalcommons.lsu.edu) — require institutional login
        - Handle.net redirects
        - Other institutional repo patterns (pure., ir., researchprofiles.)

    Direct PDF URLs attempted (confirmed working):
        - Nature, Springer, PLOS, Frontiers, MDPI
        - Diabetes Journals, Cambridge, Cell, Wiley*, OUP*
        (* partial success — some require institutional access)

    Also checks data/pdfs_manual/ for manually downloaded PDFs.

Design principles:
    - Idempotent: skips papers already successfully parsed
    - Resumable: parse_manifest.json tracks status per paper
    - Rate-limited: respects GROBID thread pool with backoff on 503
    - Privacy-safe: downloaded PDFs deleted immediately after parsing

Usage:
    python 02_parse_grobid.py
    python 02_parse_grobid.py --force    # reprocess everything
    python 02_parse_grobid.py --limit 50 # test with 50 papers
"""

import argparse
import json
import logging
import sys
import time
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import (
    GROBID_URL,
    PARSED_DIR,
    RAW_DIR,
    DATA_DIR,
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
GROBID_ENDPOINT = f"{GROBID_URL}/api/processFulltextDocument"
GROBID_ALIVE_ENDPOINT = f"{GROBID_URL}/api/isalive"

PDFS_MANUAL_DIR = DATA_DIR / "pdfs_manual"
PDFS_MANUAL_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_TIMEOUT = 30
GROBID_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF = 5

# Browser-like headers — reduces 403 rejections from publishers
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
}

# URL prefixes confirmed to never serve direct downloadable PDFs.
# These are skipped immediately without any network request.
NON_PDF_URL_PATTERNS = [
    # DOI redirects — land on publisher HTML pages
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    # GitHub — code repositories, not papers
    "https://github.com/",
    "http://github.com/",
    # PMC HTML article pages — serve text/html even at /pdf/ endpoint
    "https://pmc.ncbi.nlm.nih.gov/articles/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/",
    # LSU institutional repositories — require institutional login
    "https://repository.lsu.edu/",
    "https://digitalcommons.lsu.edu/",
    # Other institutional repository patterns
    "https://researchprofiles.",
    "https://pure.",
    "https://hdl.handle.net/",
    "https://ir.",
]


# ---------------------------------------------------------------------------
# GROBID health check
# ---------------------------------------------------------------------------

def check_grobid_alive() -> bool:
    """Verify GROBID is running before starting batch processing."""
    try:
        r = requests.get(GROBID_ALIVE_ENDPOINT, timeout=10)
        if r.status_code == 200 and r.text.strip().lower() == "true":
            log.info(f"GROBID is alive at {GROBID_URL}")
            return True
        else:
            log.error(f"GROBID not ready: {r.status_code} {r.text}")
            return False
    except requests.exceptions.ConnectionError:
        log.error(
            f"Cannot connect to GROBID at {GROBID_URL}. "
            "Is the Docker container running? "
            "Run: docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1"
        )
        return False


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


def load_oa_records(input_path: Path) -> list[dict]:
    """Load records that have either an OA PDF URL or a manual PDF available."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                paper_id = get_paper_id(record)
                has_oa_url = bool(record.get("pdf_url"))
                has_manual = (PDFS_MANUAL_DIR / f"{paper_id}.pdf").exists()
                if has_oa_url or has_manual:
                    records.append(record)
            except json.JSONDecodeError:
                continue

    oa_count = sum(1 for r in records if r.get("pdf_url"))
    manual_count = sum(
        1 for r in records
        if not r.get("pdf_url") and
        (PDFS_MANUAL_DIR / f"{get_paper_id(r)}.pdf").exists()
    )
    log.info(f"Records with OA PDF URL : {oa_count:,}")
    log.info(f"Records with manual PDF : {manual_count:,}")
    log.info(f"Total to process        : {len(records):,}")
    return records


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Load the parse manifest tracking per-paper processing status."""
    manifest_path = PARSED_DIR / "parse_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    """Save the parse manifest to disk."""
    manifest_path = PARSED_DIR / "parse_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_paper_id(record: dict) -> str:
    """Get a stable identifier for a paper safe for use as a filename."""
    if record.get("pmid"):
        return f"pmid_{record['pmid']}"
    if record.get("doi"):
        return "doi_" + record["doi"].replace("/", "_").replace(".", "_")
    return record["openalex_id"].replace("https://openalex.org/", "")


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

def should_skip_url(url: str) -> bool:
    """
    Check if a URL should be skipped entirely.

    Returns True for DOI redirects, GitHub repos, PMC HTML pages,
    institutional repositories, and other confirmed non-PDF sources.

    Args:
        url: URL to check.

    Returns:
        True if URL should be skipped without attempting download.
    """
    return any(url.startswith(pattern) for pattern in NON_PDF_URL_PATTERNS)


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest_path: Path) -> bool:
    """
    Download a PDF from the given URL.

    Skips known non-PDF URL patterns immediately.
    Checks Content-Type header before downloading full body.
    Uses browser-like headers to reduce 403 rejections.

    Args:
        url: URL to download from.
        dest_path: Local path to save the PDF.

    Returns:
        True if download succeeded and file is a valid PDF.
    """
    # Skip confirmed non-PDF URL patterns
    if should_skip_url(url):
        log.debug(f"Skipping non-PDF URL: {url}")
        return False

    try:
        r = requests.get(
            url,
            headers=DOWNLOAD_HEADERS,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
        )

        if r.status_code != 200:
            log.debug(f"HTTP {r.status_code} for {url}")
            return False

        # Check Content-Type before downloading full body
        content_type = r.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            log.debug(f"Non-PDF content-type ({content_type[:50]}) for {url}")
            return False

        # Download file
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify PDF magic bytes
        with open(dest_path, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            log.debug(f"Downloaded file is not a valid PDF: {url}")
            dest_path.unlink(missing_ok=True)
            return False

        return True

    except requests.exceptions.RequestException as e:
        log.debug(f"Download error for {url}: {e}")
        dest_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# GROBID processing
# ---------------------------------------------------------------------------

def process_with_grobid(pdf_path: Path) -> str | None:
    """
    Send a PDF to GROBID's processFulltextDocument endpoint.

    Handles 503 (server busy) with exponential backoff and retries.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        TEI XML string if successful, None otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            with open(pdf_path, "rb") as pdf_file:
                r = requests.post(
                    GROBID_ENDPOINT,
                    files={"input": (pdf_path.name, pdf_file, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "0",
                        "includeRawCitations": "0",
                        "segmentSentences": "1",
                        "generateIDs": "1",
                    },
                    timeout=GROBID_TIMEOUT,
                )

            if r.status_code == 200:
                return r.text
            elif r.status_code == 503:
                wait_time = RETRY_BACKOFF * (attempt + 1)
                log.debug(f"GROBID busy, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                log.debug(f"GROBID HTTP {r.status_code} for {pdf_path.name}")
                return None

        except requests.exceptions.Timeout:
            log.debug(f"GROBID timeout (attempt {attempt + 1})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF)
            continue
        except requests.exceptions.RequestException as e:
            log.debug(f"GROBID request error: {e}")
            return None

    return None


# ---------------------------------------------------------------------------
# TEI XML section extraction
# ---------------------------------------------------------------------------

def extract_sections_from_tei(tei_xml: str) -> dict:
    """
    Parse GROBID's TEI XML and extract structured IMRaD sections.

    Args:
        tei_xml: TEI XML string returned by GROBID.

    Returns:
        Dict with section names as keys and text content as values.
    """
    soup = BeautifulSoup(tei_xml, "xml")

    sections = {
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "acknowledgements": "",
        "references": [],
    }

    # Abstract
    abstract_tag = soup.find("abstract")
    if abstract_tag:
        sections["abstract"] = abstract_tag.get_text(separator=" ").strip()

    # Body sections — matched by heading text
    SECTION_MAP = {
        "introduction": ["introduction", "background"],
        "methods": [
            "method", "methods", "materials and methods",
            "patients and methods", "study design", "experimental",
        ],
        "results": ["result", "results", "findings"],
        "discussion": ["discussion"],
        "conclusion": ["conclusion", "conclusions", "summary"],
        "acknowledgements": [
            "acknowledgement", "acknowledgements",
            "acknowledgment", "acknowledgments",
        ],
    }

    body = soup.find("body")
    if body:
        for div in body.find_all("div"):
            head = div.find("head")
            if not head:
                continue
            head_text = head.get_text().strip().lower()
            for section_key, keywords in SECTION_MAP.items():
                if any(kw in head_text for kw in keywords):
                    paragraphs = [
                        p.get_text(separator=" ").strip()
                        for p in div.find_all("p")
                    ]
                    text = " ".join(paragraphs)
                    if text and not sections[section_key]:
                        sections[section_key] = text
                    break

    # References
    ref_list = soup.find("listBibl")
    if ref_list:
        for bibl in ref_list.find_all("biblStruct"):
            title_tag = (
                bibl.find("title", {"level": "a"}) or bibl.find("title")
            )
            authors = [
                p.get_text(separator=" ").strip()
                for p in bibl.find_all("persName")
            ]
            date_tag = bibl.find("date")
            sections["references"].append({
                "title": title_tag.get_text().strip() if title_tag else "",
                "authors": authors[:5],
                "year": date_tag.get("when", "")[:4] if date_tag else "",
            })

    return sections


# ---------------------------------------------------------------------------
# Main parse loop
# ---------------------------------------------------------------------------

def run_parse(
    input_path: Path,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """
    Download and parse all available PDFs in the filtered corpus.

    Args:
        input_path: Path to the filtered JSONL corpus.
        limit: Optional max number of papers to process (for testing).
        force: If True, reprocess papers already in the manifest.

    Returns:
        Summary statistics dict.
    """
    if not check_grobid_alive():
        sys.exit(1)

    records = load_oa_records(input_path)
    if limit:
        records = records[:limit]
        log.info(f"Limiting to first {limit} records")

    manifest = load_manifest()

    stats = {
        "total_available": len(records),
        "parsed": 0,
        "skipped": 0,
        "skipped_non_pdf_url": 0,
        "download_failed": 0,
        "grobid_failed": 0,
        "errors": 0,
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="pennington_pdfs_"))
    log.info(f"Temp dir for downloads: {temp_dir}")

    try:
        for record in tqdm(records, desc="Parsing papers", unit="paper"):
            paper_id = get_paper_id(record)
            tei_path = PARSED_DIR / f"{paper_id}.tei.xml"
            json_path = PARSED_DIR / f"{paper_id}.sections.json"

            # Skip successfully parsed papers unless --force
            if not force and paper_id in manifest:
                if manifest[paper_id].get("status") == "success":
                    stats["skipped"] += 1
                    continue

            # Determine PDF source
            # Priority: manual PDF first, then OA URL
            manual_pdf = PDFS_MANUAL_DIR / f"{paper_id}.pdf"
            pdf_url = record.get("pdf_url", "")

            if manual_pdf.exists():
                pdf_path = manual_pdf
                source = "manual"
                download_ok = True

            elif pdf_url:
                # Skip non-PDF URLs immediately
                if should_skip_url(pdf_url):
                    stats["skipped_non_pdf_url"] += 1
                    manifest[paper_id] = {
                        "status": "skipped_non_pdf_url",
                        "url": pdf_url,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    continue

                pdf_path = temp_dir / f"{paper_id}.pdf"
                download_ok = download_pdf(pdf_url, pdf_path)
                source = "oa_download"

                if not download_ok:
                    stats["download_failed"] += 1
                    manifest[paper_id] = {
                        "status": "download_failed",
                        "url": pdf_url,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    continue
            else:
                continue

            # Process with GROBID
            tei_xml = process_with_grobid(pdf_path)

            # Clean up downloaded PDF immediately (not manual PDFs)
            if source == "oa_download":
                pdf_path.unlink(missing_ok=True)

            if not tei_xml:
                stats["grobid_failed"] += 1
                manifest[paper_id] = {
                    "status": "grobid_failed",
                    "url": pdf_url if source == "oa_download" else "manual",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                continue

            # Save TEI XML
            tei_path.write_text(tei_xml, encoding="utf-8")

            # Extract and save structured sections
            try:
                sections = extract_sections_from_tei(tei_xml)
                output = {
                    "paper_id": paper_id,
                    "openalex_id": record.get("openalex_id", ""),
                    "doi": record.get("doi", ""),
                    "pmid": record.get("pmid", ""),
                    "title": record.get("title", ""),
                    "publication_year": record.get("publication_year"),
                    "journal_name": record.get("journal_name", ""),
                    "sections": sections,
                    "_parsed_at": datetime.utcnow().isoformat(),
                    "_source": source,
                }
                json_path.write_text(
                    json.dumps(output, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                log.warning(f"Section extraction failed for {paper_id}: {e}")
                stats["errors"] += 1

            stats["parsed"] += 1
            manifest[paper_id] = {
                "status": "success",
                "source": source,
                "tei_path": str(tei_path.name),
                "json_path": str(json_path.name),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Save manifest every 50 papers
            if stats["parsed"] % 50 == 0:
                save_manifest(manifest)
                log.info(
                    f"Progress: {stats['parsed']} parsed | "
                    f"{stats['download_failed']} download failures | "
                    f"{stats['grobid_failed']} GROBID failures | "
                    f"{stats['skipped_non_pdf_url']} non-PDF URLs skipped"
                )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        save_manifest(manifest)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OA PDFs and parse with GROBID."
    )
    parser.add_argument(
        "--input", type=str, default=None, metavar="PATH",
        help="Path to filtered JSONL file. Defaults to most recent.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only process first N papers (for testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess papers already in the parse manifest.",
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

    stats = run_parse(input_path, limit=args.limit, force=args.force)

    print("\n" + "=" * 60)
    print("PARSE REPORT SUMMARY")
    print("=" * 60)
    print(f"Total available PDFs   : {stats['total_available']:,}")
    print(f"Successfully parsed    : {stats['parsed']:,}")
    print(f"Skipped (already done) : {stats['skipped']:,}")
    print(f"Non-PDF URLs skipped   : {stats['skipped_non_pdf_url']:,}")
    print(f"Download failures      : {stats['download_failed']:,}")
    print(f"GROBID failures        : {stats['grobid_failed']:,}")
    print(f"Section extract errors : {stats['errors']:,}")
    print(f"\nTEI XML files in : {PARSED_DIR}")
    print("=" * 60)
