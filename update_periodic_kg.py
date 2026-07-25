"""
update_periodic_kg.py — Monthly update script for the Pennington Biomedical KG.

Runs the full pipeline update in sequence:
    01_ingest.py          Pull full corpus from OpenAlex (no date filter —
                          from_updated_date requires a premium plan)
    01b_filter.py         Filter for valid Pennington papers
    03_extract_ner.py     Extract entities from new abstracts (skips existing)
    04_load_neo4j.py      Upsert new/updated records into Neo4j (no duplicates)

Why full corpus pull instead of incremental:
    The OpenAlex `from_updated_date` filter requires a Premium plan as of 2026.
    The free tier (10,000 credits/day) supports a full corpus pull of ~9,800
    Pennington papers in ~50 API pages (~50 credits) — well within limits.
    Neo4j's MERGE ensures existing records are updated, not duplicated.

Usage:
    python update_periodic_kg.py              # standard monthly update
    python update_periodic_kg.py --dry-run    # show what would run
    python update_periodic_kg.py --history    # show past update history
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
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
# Update log management
# ---------------------------------------------------------------------------

def load_update_log() -> list[dict]:
    """Load the history of past updates."""
    if UPDATE_LOG.exists():
        with open(UPDATE_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_update_log(entries: list[dict]) -> None:
    """Save the update history."""
    UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(UPDATE_LOG, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(cmd: list[str], step_name: str, dry_run: bool = False) -> bool:
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

def run_update(dry_run: bool = False) -> bool:
    """
    Run the full monthly update pipeline.

    Pulls the complete Pennington corpus from OpenAlex (free tier compatible),
    filters, extracts entities for new papers, and upserts into Neo4j.

    Args:
        dry_run: If True, show what would run without executing.

    Returns:
        True if all steps completed successfully.
    """
    start_time = datetime.now()

    log.info("=" * 60)
    log.info("PENNINGTON KG — MONTHLY UPDATE")
    log.info("=" * 60)
    log.info(f"Started at    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Dry run       : {dry_run}")
    log.info(f"Project root  : {PROJECT_ROOT}")
    log.info("")
    log.info("Note: Using full corpus pull (from_updated_date requires")
    log.info("      OpenAlex Premium). Neo4j MERGE handles deduplication.")
    log.info("=" * 60)

    steps = [
        {
            "name": "Pull full Pennington corpus from OpenAlex",
            "cmd":  ["python", "01_ingest.py"],
            # No --since flag — full corpus pull works on free tier
            # ~50 API pages = ~50 credits (well within 10,000/day limit)
        },
        {
            "name": "Filter corpus for valid Pennington papers",
            "cmd":  ["python", "01b_filter.py"],
        },
        {
            "name": "Extract biomedical entities (new papers only)",
            "cmd":  ["python", "03_extract_ner.py", "--pubtator-only"],
            # --pubtator-only: fast (~7 min), skips papers already extracted
            # Remove flag for full SciSpaCy pass if desired
        },
        {
            "name": "Upsert new/updated records into Neo4j",
            "cmd":  ["python", "04_load_neo4j.py"],
            # MERGE ensures no duplicates — safe to run on full corpus
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

    # Record this run
    if not dry_run:
        entries = load_update_log()
        entries.append({
            "run_at":      start_time.isoformat(),
            "mode":        "full_corpus",
            "status":      status,
            "failed_step": failed_step,
            "elapsed_sec": elapsed_total,
        })
        save_update_log(entries)

    log.info("\n" + "=" * 60)
    if status == "success":
        log.info(
            f"UPDATE COMPLETE in "
            f"{elapsed_total // 60}m {elapsed_total % 60}s"
        )
        log.info(
            "Neo4j now reflects the latest Pennington publications."
        )
        log.info("Refresh the Streamlit dashboard to see new data.")
    else:
        log.error(f"UPDATE FAILED at step: {failed_step}")
        log.error(
            "Fix the error above and re-run. Already-completed steps "
            "will be skipped automatically (idempotent pipeline)."
        )
    log.info("=" * 60)

    return status == "success"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monthly update for the Pennington Biomedical KG."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing anything.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show history of past updates and exit.",
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
            print(f"\n{'Date':<25} {'Mode':<15} {'Status':<10} {'Time':<12} Notes")
            print("-" * 75)
            for e in entries:
                run_at   = e.get("run_at", "")[:19]
                mode     = e.get("mode", "incremental")
                status   = e.get("status", "")
                elapsed  = e.get("elapsed_sec", 0)
                failed   = e.get("failed_step", "") or ""
                time_str = f"{elapsed // 60}m {elapsed % 60}s"
                print(f"{run_at:<25} {mode:<15} {status:<10} {time_str:<12} {failed}")
        sys.exit(0)

    success = run_update(dry_run=args.dry_run)
    sys.exit(0 if success else 1)
