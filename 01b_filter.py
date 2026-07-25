"""
01b_filter.py — Post-ingest quality filter for the Pennington Biomedical corpus.

Pipeline stage: FILTER (runs after 01_ingest.py)
Input:  data/raw/works_<YYYYMMDD>.jsonl       — raw OpenAlex records
Output: data/raw/works_<YYYYMMDD>_filtered.jsonl  — cleaned corpus
        data/raw/filter_report_<YYYYMMDD>.json    — detailed rejection statistics

Problem this solves:
    OpenAlex occasionally misassigns Pennington's ROR ID (040cnym54) to authors
    at other institutions, causing non-Pennington papers to appear in the corpus.
    This script removes those false positives using a multi-signal approach:
        1. Publication year must be >= 1988 (Pennington founded 1990; 2yr margin)
        2. Institution name must contain a Pennington-related term (primary filter)
        3. Work type must be a recognized scholarly output (secondary filter)
        4. Optional: domain keyword filter to flag non-biomedical outliers

Design principles:
    - Idempotent: safe to re-run; always reads from raw, writes to filtered.
    - Conservative: when in doubt, KEEP the record (false negatives are worse
      than false positives at this stage — downstream NER will further refine).
    - Transparent: every rejected record is logged with its rejection reason.

Usage:
    # Filter the most recent ingest file
    python 01b_filter.py

    # Filter a specific ingest file
    python 01b_filter.py --input data/raw/works_20260429.jsonl
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import RAW_DIR

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
# Filter configuration
# ---------------------------------------------------------------------------

# Pennington Biomedical Research Center was established in 1990.
# We use 1988 as a conservative lower bound (2-year safety margin).
MIN_PUBLICATION_YEAR = 1988

# Institution name terms that confirm a Pennington affiliation.
# Case-insensitive. A record passes if ANY author has ANY affiliation
# whose institution_name contains ANY of these terms.
PENNINGTON_NAME_TERMS = [
    "pennington",
    "pennington biomedical",
    "pbrc",
]

# Work types to keep. OpenAlex type vocabulary:
# article, review, preprint, book-chapter, dissertation, report, paratext
# We exclude 'paratext' (journal front matter) and other non-scholarly types.
ACCEPTED_WORK_TYPES = {
    "article",
    "review",
    "preprint",
    "book-chapter",
    "book",
    "dissertation",
    "report",
    "dataset",
    "other",  # keep 'other' conservatively
}

# Biomedical domain keywords — used ONLY for flagging, not hard rejection.
# Records missing these are tagged as 'low_confidence' but still kept.
BIOMEDICAL_KEYWORDS = [
    "obesity", "metabolism", "metabolic", "nutrition", "nutritional",
    "diabetes", "diabetic", "insulin", "glucose", "glycemic",
    "adipose", "adiposity", "adipocyte", "fat", "lipid", "lipolysis",
    "energy expenditure", "energy intake", "caloric", "calorie",
    "cardiovascular", "cardiometabolic", "hypertension", "blood pressure",
    "physical activity", "exercise", "sedentary", "fitness",
    "weight loss", "weight gain", "body weight", "body mass", "bmi",
    "gut microbiome", "microbiota", "inflammation", "inflammatory",
    "hormone", "leptin", "ghrelin", "adiponectin", "cortisol",
    "clinical trial", "randomized", "cohort", "epidemiology",
    "gene", "genomic", "epigenetic", "proteomics", "metabolomics",
    "mouse", "rat", "rodent", "animal model",
    "bariatric", "gastric bypass", "caloric restriction",
    "liver", "hepatic", "pancreas", "pancreatic", "thyroid",
    "muscle", "skeletal muscle", "myocyte",
]


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def has_valid_year(record: dict) -> bool:
    """
    Check whether the publication year is plausible for a Pennington paper.

    Pennington Biomedical was established in 1990. We reject anything
    published before 1988 (2-year safety margin) as a false positive.
    Records with no year are kept conservatively.

    Args:
        record: Normalized work record.

    Returns:
        True if year is valid or unknown.
    """
    year = record.get("publication_year")
    if year is None:
        return True  # keep if year unknown — conservative
    return int(year) >= MIN_PUBLICATION_YEAR


def has_pennington_affiliation(record: dict) -> bool:
    """
    Check whether any author in this record has a confirmed Pennington
    affiliation based on institution name or ROR ID.

    Args:
        record: Normalized work record from 01_ingest.py

    Returns:
        True if at least one Pennington-affiliated author is found.
    """
    for authorship in record.get("authorships", []):
        for affiliation in authorship.get("affiliations", []):
            inst_name = (affiliation.get("institution_name") or "").lower()
            ror_id = (affiliation.get("ror_id") or "")

            if any(term in inst_name for term in PENNINGTON_NAME_TERMS):
                return True

            if "040cnym54" in ror_id:
                return True

    return False


def has_accepted_work_type(record: dict) -> bool:
    """
    Check whether the work type is a recognized scholarly output type.

    Args:
        record: Normalized work record.

    Returns:
        True if work type is in the accepted set.
    """
    work_type = (record.get("work_type") or "").lower()
    return work_type in ACCEPTED_WORK_TYPES


def has_biomedical_signal(record: dict) -> bool:
    """
    Check whether the record has any biomedical content signal.
    Used for confidence flagging only — does NOT drive hard rejection.

    Args:
        record: Normalized work record.

    Returns:
        True if any biomedical keyword is found in the record's text fields.
    """
    text_signals = " ".join(filter(None, [
        record.get("title", ""),
        record.get("abstract", ""),
        " ".join(m.get("descriptor_name", "") for m in record.get("mesh", [])),
        " ".join(c.get("display_name", "") for c in record.get("concepts", [])),
        " ".join(t.get("display_name", "") for t in record.get("topics", [])),
    ])).lower()

    return any(kw in text_signals for kw in BIOMEDICAL_KEYWORDS)


def classify_record(record: dict) -> tuple[bool, str, str]:
    """
    Classify a record as keep or reject, with a reason code and description.

    Rejection hierarchy (first match wins):
        1. Publication year predates Pennington → REJECT
        2. No Pennington affiliation by name/ROR → REJECT (false positive)
        3. Unacceptable work type → REJECT (non-scholarly)

    Keep with flag:
        4. No biomedical signal → KEEP but flagged as 'low_confidence'
        5. Everything passes → KEEP as 'confirmed'

    Args:
        record: Normalized work record.

    Returns:
        Tuple of (keep: bool, status_code: str, description: str)
    """
    # Year check — reject pre-Pennington papers
    if not has_valid_year(record):
        year = record.get("publication_year", "unknown")
        return (
            False,
            "invalid_year",
            f"Publication year {year} predates Pennington Biomedical (est. 1990)",
        )

    # Affiliation check — reject ROR ID false positives
    if not has_pennington_affiliation(record):
        return (
            False,
            "no_pennington_affiliation",
            "No author affiliation name or ROR ID matches Pennington",
        )

    # Work type check — reject non-scholarly outputs
    if not has_accepted_work_type(record):
        return (
            False,
            "unacceptable_work_type",
            f"Work type '{record.get('work_type')}' is not a scholarly output",
        )

    # Biomedical signal — flag but keep
    if not has_biomedical_signal(record):
        return (
            True,
            "low_confidence",
            "Pennington affiliation confirmed but no biomedical keyword signal found",
        )

    return (
        True,
        "confirmed",
        "Pennington affiliation confirmed with biomedical signal",
    )


# ---------------------------------------------------------------------------
# Main filter loop
# ---------------------------------------------------------------------------

def find_latest_ingest_file() -> Path:
    """
    Find the most recently created raw ingest JSONL file that has not
    already been filtered.

    Returns:
        Path to the latest unfiltered JSONL file.

    Raises:
        FileNotFoundError: If no suitable input file is found.
    """
    candidates = sorted(
        [f for f in RAW_DIR.glob("works_*.jsonl")
         if "_filtered" not in f.name],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No unfiltered works_*.jsonl files found in {RAW_DIR}. "
            "Run 01_ingest.py first."
        )
    return candidates[0]


def run_filter(input_path: Path) -> tuple[Path, dict]:
    """
    Apply all filters to the input JSONL file and write the filtered output.

    Args:
        input_path: Path to the raw ingest JSONL file.

    Returns:
        Tuple of (output_path, report_dict)
    """
    output_path = input_path.parent / input_path.name.replace(
        ".jsonl", "_filtered.jsonl"
    )
    today = datetime.utcnow().strftime("%Y%m%d")
    report_path = RAW_DIR / f"filter_report_{today}.json"

    log.info(f"Input:  {input_path}")
    log.info(f"Output: {output_path}")

    total = 0
    kept = 0
    rejected = 0
    status_counts: dict[str, int] = {}
    rejection_samples: dict[str, list] = {}

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed JSON at line {total}: {e}")
                rejected += 1
                continue

            keep, status_code, description = classify_record(record)

            status_counts[status_code] = status_counts.get(status_code, 0) + 1

            if keep:
                record["_filter_status"] = status_code
                record["_filter_description"] = description
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected += 1
                if status_code not in rejection_samples:
                    rejection_samples[status_code] = []
                if len(rejection_samples[status_code]) < 3:
                    rejection_samples[status_code].append({
                        "openalex_id": record.get("openalex_id", ""),
                        "title": record.get("title", "")[:120],
                        "journal": record.get("journal_name", ""),
                        "year": record.get("publication_year"),
                        "reason": description,
                        "affiliations": [
                            aff.get("institution_name", "")
                            for auth in record.get("authorships", [])
                            for aff in auth.get("affiliations", [])
                        ][:5],
                    })

            if total % 1000 == 0:
                log.info(
                    f"  Processed {total} records | "
                    f"kept {kept} | rejected {rejected}"
                )

    report = {
        "run_at": datetime.utcnow().isoformat(),
        "input_file": str(input_path.name),
        "output_file": str(output_path.name),
        "min_publication_year": MIN_PUBLICATION_YEAR,
        "total_input": total,
        "total_kept": kept,
        "total_rejected": rejected,
        "retention_rate_pct": round(100 * kept / total, 2) if total > 0 else 0,
        "status_breakdown": status_counts,
        "rejection_samples": rejection_samples,
    }

    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    return output_path, report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the Pennington OpenAlex ingest for false positives."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the raw ingest JSONL file to filter. "
            "If omitted, the most recent works_*.jsonl in data/raw/ is used."
        ),
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
            input_path = find_latest_ingest_file()
            log.info(f"Auto-detected input file: {input_path.name}")
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(1)

    output_path, report = run_filter(input_path)

    print("\n" + "=" * 60)
    print("FILTER REPORT SUMMARY")
    print("=" * 60)
    print(f"Total input records  : {report['total_input']:,}")
    print(f"Records kept         : {report['total_kept']:,}")
    print(f"Records rejected     : {report['total_rejected']:,}")
    print(f"Retention rate       : {report['retention_rate_pct']}%")
    print(f"Min year applied     : {report['min_publication_year']}")
    print("\nStatus breakdown:")
    for status, count in report["status_breakdown"].items():
        print(f"  {status:<35} {count:,}")
    print(f"\nFiltered output : {output_path}")
    print(f"Full report     : data/raw/filter_report_{datetime.utcnow().strftime('%Y%m%d')}.json")
    print("=" * 60)
