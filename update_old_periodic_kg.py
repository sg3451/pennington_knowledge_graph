"""
update_kg.py — Incremental update script for the Pennington Biomedical KG.

Runs the full pipeline update in sequence:
    01_ingest.py --since <date>   Pull new/updated papers from OpenAlex
    01b_filter.py                 Filter for valid Pennington papers
    03_extract_ner.py             Extract biomedical entities from new abstracts
    04_load_neo4j.py              Upsert new records into Neo4j (no duplicates)

The --since date defaults to 35 days ago (5-day overlap over monthly cadence
ensures no papers are missed at month boundaries).

Usage:
    # Standard monthly update (pulls last 35 days)
    python update_kg.py

    # Custom window — pull last N days
    python update_kg.py 60

    # Specific date
    python update_kg.py --since 2026-04-01

    # Dry run — show what would run without executing
    python update_kg.py --dry-run
"""

import argparse
import subprocess
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

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
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
UPDATE_LOG   = PROJECT_ROOT / "data" / "raw" / "update_log.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_update_log() -> list[dict]:
    """Load the history of past updates."""
    if UPDATE_LOG.exists():
        with open(UPDATE_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_update_log(log_entries: list[dict]) -> None:
    """Save the update history."""
    UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(UPDATE_LOG, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)


def get_last_update_date() -> str | None:
    """Return the since_date from the most recent successful update."""
    entries = load_update_log()
    successful = [e for e in entries if e.get("status") == "success"]
    if successful:
        return successful[-1].get("since_date")
    return None


def run_step(
    cmd: list[str],
    step_name: str,
    dry_run: bool = False,
) -> bool:
    """
    Run a pipeline step and return True if successful.

    Args:
        cmd: Command list to execute.
        step_name: Human-readable step name for logging.
        dry_run: If True, print command without executing.

    Returns:
        True if step completed successfully.
    """
    log.info(f"{'[DRY RUN] ' if dry_run else ''}Running: {' '.join(cmd)}")

    if dry_run:
        return True

    start = datetime.now()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = (datetime.now() - start).seconds

    if result.returncode == 0:
        log.info(f"  ✓ {step_name} completed in {elapsed}s")
        return True
    else:
        log.error(f"  ✗ {step_name} failed (exit code {result.returncode})")
        return False


# ---------------------------------------------------------------------------
# Main update pipeline
# ---------------------------------------------------------------------------

def run_update(
    since_date: str,
    dry_run: bool = False,
) -> bool:
    """
    Run the full incremental update pipeline.

    Args:
        since_date: ISO date string (YYYY-MM-DD) — pull papers updated since this date.
        dry_run: If True, show what would run without executing.

    Returns:
        True if all steps completed successfully.
    """
    start_time = datetime.now()

    log.info("=" * 60)
    log.info("PENNINGTON KG — INCREMENTAL UPDATE")
    log.info("=" * 60)
    log.info(f"Since date    : {since_date}")
    log.info(f"Started at    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Dry run       : {dry_run}")
    log.info(f"Project root  : {PROJECT_ROOT}")
    log.info("=" * 60)

    steps = [
        {
            "name": "Ingest new papers from OpenAlex",
            "cmd": ["python", "01_ingest.py", "--since", since_date],
        },
        {
            "name": "Filter corpus",
            "cmd": ["python", "01b_filter.py"],
        },
        {
            "name": "Extract biomedical entities (NER)",
            "cmd": ["python", "03_extract_ner.py", "--pubtator-only"],
            # Using --pubtator-only for speed on incremental updates.
            # PubTator gives high-precision results for PMID-indexed papers.
            # Run without this flag for full SciSpaCy pass if desired.
        },
        {
            "name": "Load new records into Neo4j",
            "cmd": ["python", "04_load_neo4j.py"],
        },
    ]

    failed_step = None
    for i, step in enumerate(steps):
        log.info(f"\nStep {i+1}/{len(steps)}: {step['name']}")
        success = run_step(step["cmd"], step["name"], dry_run=dry_run)
        if not success:
            failed_step = step["name"]
            break

    elapsed_total = (datetime.now() - start_time).seconds
    status = "success" if not failed_step else "failed"

    # Log the update
    if not dry_run:
        log_entries = load_update_log()
        log_entries.append({
            "run_at":      start_time.isoformat(),
            "since_date":  since_date,
            "status":      status,
            "failed_step": failed_step,
            "elapsed_sec": elapsed_total,
        })
        save_update_log(log_entries)

    log.info("\n" + "=" * 60)
    if status == "success":
        log.info(f"UPDATE COMPLETE in {elapsed_total // 60}m {elapsed_total % 60}s")
        log.info("The Neo4j graph now includes all papers updated since "
                 f"{since_date}.")
        log.info("Refresh the Streamlit dashboard to see new data.")
    else:
        log.error(f"UPDATE FAILED at step: {failed_step}")
        log.error("Fix the error above and re-run. Already-completed steps")
        log.error("will be skipped automatically (idempotent pipeline).")
    log.info("=" * 60)

    return status == "success"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental update for the Pennington Biomedical KG."
    )

    parser.add_argument(
        "days",
        nargs="?",
        type=int,
        default=None,
        help="Pull papers from the last N days (default: 35).",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Pull papers updated on or after this date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing anything.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show the history of past updates and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Show history and exit
    if args.history:
        entries = load_update_log()
        if not entries:
            print("No update history found.")
        else:
            print(f"\n{'Date':<25} {'Since':<14} {'Status':<10} {'Time':<10} Notes")
            print("-" * 75)
            for e in entries:
                run_at   = e.get("run_at", "")[:19]
                since    = e.get("since_date", "")
                status   = e.get("status", "")
                elapsed  = e.get("elapsed_sec", 0)
                failed   = e.get("failed_step", "") or ""
                time_str = f"{elapsed // 60}m {elapsed % 60}s"
                print(f"{run_at:<25} {since:<14} {status:<10} {time_str:<10} {failed}")
        sys.exit(0)

    # Determine since_date
    if args.since:
        # Explicit date provided
        since_date = args.since
        try:
            datetime.strptime(since_date, "%Y-%m-%d")
        except ValueError:
            log.error(f"Invalid date format: {since_date}. Use YYYY-MM-DD.")
            sys.exit(1)

    elif args.days:
        # N days back
        since_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    else:
        # Default: suggest using last update date, fall back to 35 days
        last_update = get_last_update_date()
        if last_update:
            # Use last update date minus 5 days for overlap
            last_dt = datetime.strptime(last_update, "%Y-%m-%d")
            since_date = (last_dt - timedelta(days=5)).strftime("%Y-%m-%d")
            log.info(f"Last update was for papers since {last_update}.")
            log.info(f"Using {since_date} (5-day overlap to prevent gaps).")
        else:
            since_date = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
            log.info("No previous update found. Defaulting to last 35 days.")

    success = run_update(since_date=since_date, dry_run=args.dry_run)
    sys.exit(0 if success else 1)
