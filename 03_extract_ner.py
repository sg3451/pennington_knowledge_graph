"""
03_extract_ner.py — Biomedical NER extraction for the Pennington corpus.

Pipeline stage: EXTRACT (runs after 01b_filter.py and 02_parse_grobid.py)
Input:  data/raw/works_<YYYYMMDD>_filtered.jsonl  — filtered corpus (abstracts)
        data/parsed/<paper_id>.sections.json      — full text sections (where available)
Output: data/entities/<paper_id>.entities.json    — extracted entities per paper
        data/entities/ner_manifest.json           — run metadata and status

Two-pass extraction strategy:
    Pass 1 — PubTator 3 API (for papers with PMIDs)
        Pre-computed, high-precision annotations for genes, diseases,
        chemicals, species, variants, cell lines. Normalized to NCBI Gene,
        MeSH, ChEBI, NCBI Taxonomy IDs. Fast — batch of 100 PMIDs per call.

    Pass 2 — SciSpaCy (en_core_sci_lg) with UMLS entity linker
        Runs on abstracts and full text for ALL papers.
        Only keeps entities with confirmed UMLS concept IDs (filters noise).
        Deduplicates by normalized_id so each concept appears once per paper.

    Merge: PubTator annotations take precedence (higher precision);
    SciSpaCy fills gaps for papers not in PubTator or entities not found.

Entity types extracted:
    Gene        — NCBI Gene ID
    Disease     — MeSH Disease ID
    Chemical    — MeSH Chemical / ChEBI ID
    Species     — NCBI Taxonomy ID
    Variant     — dbSNP / ClinVar ID
    CellLine    — Cellosaurus ID
    biomedical  — UMLS-linked entities from SciSpaCy

Design principles:
    - Idempotent: skips papers already in ner_manifest with status=success
    - Resumable: manifest saved every 100 papers
    - Rate-limited: PubTator API batched with delays
    - SciSpaCy: only keeps UMLS-confirmed entities to reduce noise

Usage:
    python 03_extract_ner.py
    python 03_extract_ner.py --limit 50       # test run
    python 03_extract_ner.py --force          # reprocess everything
    python 03_extract_ner.py --pubtator-only  # skip SciSpaCy (faster)
    python 03_extract_ner.py --scispacy-only  # skip PubTator (offline mode)
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

from config import (
    RAW_DIR,
    PARSED_DIR,
    ENTITIES_DIR,
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
PUBTATOR_API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
PUBTATOR_BATCH_SIZE = 100
PUBTATOR_RATE_DELAY = 0.4

PUBTATOR_TYPE_MAP = {
    "Gene":            "gene",
    "Disease":         "disease",
    "Chemical":        "chemical",
    "Species":         "species",
    "Mutation":        "variant",
    "CellLine":        "cell_line",
    "DNAMutation":     "variant",
    "ProteinMutation": "variant",
    "SNP":             "variant",
}

NER_MANIFEST_PATH = ENTITIES_DIR / "ner_manifest.json"


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
            f"No filtered works_*_filtered.jsonl files found in {RAW_DIR}."
        )
    return candidates[0]


def load_corpus(input_path: Path) -> list[dict]:
    """Load all records from the filtered corpus."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    log.info(f"Loaded {len(records):,} records from {input_path.name}")
    return records


def get_paper_id(record: dict) -> str:
    """Get stable paper ID matching other pipeline stages."""
    if record.get("pmid"):
        return f"pmid_{record['pmid']}"
    if record.get("doi"):
        return "doi_" + record["doi"].replace("/", "_").replace(".", "_")
    return record["openalex_id"].replace("https://openalex.org/", "")


def get_text_for_ner(record: dict) -> dict[str, str]:
    """
    Get all available text for a paper from abstract and full text sections.

    Returns:
        Dict mapping section name to text content.
    """
    texts = {}

    title = record.get("title", "").strip()
    if title:
        texts["title"] = title

    abstract = record.get("abstract", "").strip()
    if abstract:
        texts["abstract"] = abstract

    # Add full text sections if available
    paper_id = get_paper_id(record)
    sections_path = PARSED_DIR / f"{paper_id}.sections.json"
    if sections_path.exists():
        try:
            with open(sections_path, "r", encoding="utf-8") as f:
                sections_data = json.load(f)
            for section_name, text in sections_data.get("sections", {}).items():
                if isinstance(text, str) and text.strip():
                    texts[f"fulltext_{section_name}"] = text.strip()
        except (json.JSONDecodeError, KeyError):
            pass

    return texts


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Load the NER manifest."""
    if NER_MANIFEST_PATH.exists():
        with open(NER_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    """Save the NER manifest."""
    with open(NER_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# PubTator 3 extraction
# ---------------------------------------------------------------------------

def fetch_pubtator_batch(pmids: list[str]) -> dict[str, list[dict]]:
    """
    Fetch pre-computed entity annotations from PubTator 3 for a batch of PMIDs.

    Returns:
        Dict mapping PMID to list of entity annotation dicts.
    """
    url = f"{PUBTATOR_API}/publications/export/biocjson"
    params = {"pmids": ",".join(pmids)}

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            log.debug(f"PubTator API {r.status_code} for batch of {len(pmids)}")
            return {}
        data = r.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        log.debug(f"PubTator API error: {e}")
        return {}

    results: dict[str, list[dict]] = {}
    documents = data if isinstance(data, list) else data.get("PubTator3", [])

    for doc in documents:
        pmid = str(doc.get("id", ""))
        if not pmid:
            continue

        entities = []
        # Deduplicate PubTator results by (normalized_id, entity_type)
        seen = set()

        for passage in doc.get("passages", []):
            for annotation in passage.get("annotations", []):
                infons = annotation.get("infons", {})
                entity_type = infons.get("type", "")
                normalized_type = PUBTATOR_TYPE_MAP.get(entity_type)
                if not normalized_type:
                    continue

                identifier = (
                    infons.get("identifier") or
                    infons.get("Identifier") or
                    ""
                )
                if not identifier or identifier in ("None", "-"):
                    continue

                # Deduplicate by (normalized_id, type)
                dedup_key = (str(identifier), normalized_type)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                entities.append({
                    "text": annotation.get("text", ""),
                    "entity_type": normalized_type,
                    "normalized_id": str(identifier),
                    "source": "pubtator3",
                    "passage_type": passage.get("infons", {}).get("type", ""),
                })

        if entities:
            results[pmid] = entities

    return results


def run_pubtator_pass(
    records: list[dict],
    manifest: dict,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Run PubTator 3 extraction for all records with PMIDs.

    Returns:
        Dict mapping paper_id to list of PubTator entity annotations.
    """
    pmid_to_paper = {}
    for record in records:
        pmid = record.get("pmid", "").strip()
        if not pmid:
            continue
        paper_id = get_paper_id(record)
        if not force and manifest.get(paper_id, {}).get("pubtator_done"):
            continue
        pmid_to_paper[pmid] = paper_id

    if not pmid_to_paper:
        log.info("PubTator: all PMIDs already processed, skipping")
        return {}

    log.info(f"PubTator: fetching annotations for {len(pmid_to_paper):,} PMIDs")

    all_results: dict[str, list[dict]] = {}
    pmids = list(pmid_to_paper.keys())
    batches = [
        pmids[i:i + PUBTATOR_BATCH_SIZE]
        for i in range(0, len(pmids), PUBTATOR_BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc="PubTator batches", unit="batch"):
        batch_results = fetch_pubtator_batch(batch)
        for pmid, entities in batch_results.items():
            paper_id = pmid_to_paper.get(pmid)
            if paper_id:
                all_results[paper_id] = entities
        time.sleep(PUBTATOR_RATE_DELAY)

    log.info(
        f"PubTator: found annotations for "
        f"{len(all_results):,} / {len(pmid_to_paper):,} PMIDs"
    )
    return all_results


# ---------------------------------------------------------------------------
# SciSpaCy extraction
# ---------------------------------------------------------------------------

def load_scispacy_pipeline():
    """
    Load the SciSpaCy NER pipeline with UMLS entity linking.

    Returns:
        Loaded spaCy pipeline object, or None if not installed.
    """
    try:
        import spacy
        import scispacy  # noqa: F401
        from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
        from scispacy.linking import EntityLinker  # noqa: F401
    except ImportError as e:
        log.error(
            f"SciSpaCy not installed: {e}\n"
            "Install with:\n"
            "  pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        )
        sys.exit(1)

    log.info("Loading SciSpaCy en_core_sci_lg model (may take 1-2 minutes)...")
    try:
        nlp = spacy.load("en_core_sci_lg")
    except OSError:
        log.error(
            "en_core_sci_lg model not found. Install with:\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        )
        sys.exit(1)

    # Abbreviation detector — resolves "T2DM" -> "type 2 diabetes mellitus"
    nlp.add_pipe("abbreviation_detector")

    # UMLS entity linker — normalized concept IDs, threshold=0.85 for precision
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.85,
            "max_entities_per_mention": 1,
        },
        last=True,
    )

    log.info("SciSpaCy pipeline loaded")
    return nlp


def extract_scispacy_entities(nlp, texts: dict[str, str]) -> list[dict]:
    """
    Extract biomedical entities from text using SciSpaCy.

    Quality filters applied:
    - Only keeps entities with a confirmed UMLS link (filters generic noise)
    - Deduplicates by UMLS concept ID (each concept appears once per paper)
    - Skips single short tokens that are likely generic scientific words

    Args:
        nlp: Loaded SciSpaCy pipeline.
        texts: Dict mapping section name to text content.

    Returns:
        Deduplicated list of UMLS-confirmed entity annotation dicts.
    """
    entities = []
    seen_umls_ids = set()  # deduplicate by UMLS concept ID across all sections

    for section_name, text in texts.items():
        if not text or len(text) < 20:
            continue

        # Truncate very long sections to avoid memory issues
        if len(text) > 100_000:
            text = text[:100_000]

        try:
            doc = nlp(text)
        except Exception as e:
            log.debug(f"SciSpaCy error on section {section_name}: {e}")
            continue

        for ent in doc.ents:
            # Require a confirmed UMLS link — filters out generic noise
            # ("Prevalence", "Models", "Pathways" won't have UMLS IDs)
            if not ent._.kb_ents:
                continue

            best_link = ent._.kb_ents[0]
            umls_id = best_link[0]
            umls_score = best_link[1]

            # Only accept high-confidence links
            if umls_score < 0.85:
                continue

            # Skip single short tokens — likely noise even with UMLS IDs
            # (e.g. "Urban" → C0442529, "Regulation" → C0851285)
            token_count = len(ent.text.split())
            if token_count == 1 and len(ent.text) < 8:
                continue

            # Deduplicate by UMLS concept ID — prevents EPO appearing 7 times
            if umls_id in seen_umls_ids:
                continue
            seen_umls_ids.add(umls_id)

            entities.append({
                "text": ent.text,
                "entity_type": "biomedical",
                "normalized_id": umls_id,
                "umls_score": round(umls_score, 3),
                "source": "scispacy",
                "section": section_name,
                "label": ent.label_,
            })

    return entities


# ---------------------------------------------------------------------------
# Entity merging
# ---------------------------------------------------------------------------

def merge_entities(
    pubtator_entities: list[dict],
    scispacy_entities: list[dict],
) -> list[dict]:
    """
    Merge PubTator and SciSpaCy entity lists.

    PubTator takes precedence — higher precision, normalized IDs.
    SciSpaCy fills in entities not found by PubTator.

    Args:
        pubtator_entities: Entities from PubTator 3.
        scispacy_entities: Entities from SciSpaCy.

    Returns:
        Merged and deduplicated entity list.
    """
    merged = list(pubtator_entities)

    # Track normalized IDs already covered by PubTator
    seen_ids = {
        e["normalized_id"]
        for e in pubtator_entities
        if e.get("normalized_id")
    }

    for entity in scispacy_entities:
        norm_id = entity.get("normalized_id", "")
        # Skip if PubTator already has this concept (avoid duplication)
        if norm_id and norm_id in seen_ids:
            continue
        merged.append(entity)
        if norm_id:
            seen_ids.add(norm_id)

    return merged


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_entity_output(
    paper_id: str,
    record: dict,
    entities: list[dict],
) -> Path:
    """Write entity extraction output for a single paper."""
    type_counts: dict[str, int] = {}
    for e in entities:
        t = e.get("entity_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    output = {
        "paper_id": paper_id,
        "openalex_id": record.get("openalex_id", ""),
        "doi": record.get("doi", ""),
        "pmid": record.get("pmid", ""),
        "title": record.get("title", ""),
        "publication_year": record.get("publication_year"),
        "entities": entities,
        "entity_count": len(entities),
        "entity_type_counts": type_counts,
        "_extracted_at": datetime.utcnow().isoformat(),
    }

    out_path = ENTITIES_DIR / f"{paper_id}.entities.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(
    input_path: Path,
    limit: int | None = None,
    force: bool = False,
    pubtator_only: bool = False,
    scispacy_only: bool = False,
) -> dict:
    """
    Run full NER extraction pipeline on the corpus.

    Args:
        input_path: Path to filtered JSONL corpus.
        limit: Optional max papers to process.
        force: If True, reprocess already-extracted papers.
        pubtator_only: Skip SciSpaCy pass.
        scispacy_only: Skip PubTator pass.

    Returns:
        Summary statistics dict.
    """
    records = load_corpus(input_path)
    if limit:
        records = records[:limit]
        log.info(f"Limiting to first {limit} records")

    manifest = load_manifest()

    stats = {
        "total": len(records),
        "extracted": 0,
        "skipped": 0,
        "pubtator_hits": 0,
        "scispacy_only_count": 0,
        "no_text": 0,
        "errors": 0,
    }

    # ---------------------------------------------------------------------------
    # Pass 1: PubTator (batch API — fast)
    # ---------------------------------------------------------------------------
    pubtator_results: dict[str, list[dict]] = {}
    if not scispacy_only:
        pubtator_results = run_pubtator_pass(records, manifest, force=force)
        stats["pubtator_hits"] = len(pubtator_results)
        log.info(
            f"PubTator pass complete: {stats['pubtator_hits']:,} papers annotated"
        )

    # ---------------------------------------------------------------------------
    # Pass 2: SciSpaCy (per paper — slower)
    # ---------------------------------------------------------------------------
    nlp = None
    if not pubtator_only:
        nlp = load_scispacy_pipeline()

    # ---------------------------------------------------------------------------
    # Per-paper: merge and write output
    # ---------------------------------------------------------------------------
    for record in tqdm(records, desc="Extracting entities", unit="paper"):
        paper_id = get_paper_id(record)

        # Skip if already successfully extracted
        if not force and manifest.get(paper_id, {}).get("status") == "success":
            stats["skipped"] += 1
            continue

        try:
            texts = get_text_for_ner(record)
            if not texts:
                stats["no_text"] += 1
                manifest[paper_id] = {
                    "status": "no_text",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                continue

            # PubTator entities for this paper
            pubtator_entities = pubtator_results.get(paper_id, [])

            # SciSpaCy entities
            scispacy_entities = []
            if nlp is not None:
                scispacy_entities = extract_scispacy_entities(nlp, texts)

            # Merge PubTator + SciSpaCy
            merged = merge_entities(pubtator_entities, scispacy_entities)

            # Write output file
            write_entity_output(paper_id, record, merged)

            stats["extracted"] += 1
            if not pubtator_entities and scispacy_entities:
                stats["scispacy_only_count"] += 1

            manifest[paper_id] = {
                "status": "success",
                "entity_count": len(merged),
                "pubtator_count": len(pubtator_entities),
                "scispacy_count": len(scispacy_entities),
                "has_fulltext": any(k.startswith("fulltext_") for k in texts),
                "pubtator_done": True,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            log.warning(f"Extraction failed for {paper_id}: {e}")
            stats["errors"] += 1
            manifest[paper_id] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Save manifest every 100 papers
        if (stats["extracted"] + stats["errors"]) % 100 == 0:
            save_manifest(manifest)
            log.info(
                f"Progress: {stats['extracted']:,} extracted | "
                f"{stats['errors']:,} errors | "
                f"{stats['skipped']:,} skipped"
            )

    save_manifest(manifest)
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Biomedical NER extraction for the Pennington corpus."
    )
    parser.add_argument(
        "--input", type=str, default=None, metavar="PATH",
        help="Path to filtered JSONL. Defaults to most recent.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only process first N papers (for testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess already-extracted papers.",
    )
    parser.add_argument(
        "--pubtator-only", action="store_true",
        help="Skip SciSpaCy pass (faster, PMIDs only).",
    )
    parser.add_argument(
        "--scispacy-only", action="store_true",
        help="Skip PubTator pass (fully offline).",
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

    stats = run_extraction(
        input_path,
        limit=args.limit,
        force=args.force,
        pubtator_only=getattr(args, "pubtator_only", False),
        scispacy_only=getattr(args, "scispacy_only", False),
    )

    print("\n" + "=" * 60)
    print("NER EXTRACTION REPORT")
    print("=" * 60)
    print(f"Total records          : {stats['total']:,}")
    print(f"Successfully extracted : {stats['extracted']:,}")
    print(f"Skipped (already done) : {stats['skipped']:,}")
    print(f"PubTator hits          : {stats['pubtator_hits']:,}")
    print(f"SciSpaCy only          : {stats['scispacy_only_count']:,}")
    print(f"No text available      : {stats['no_text']:,}")
    print(f"Errors                 : {stats['errors']:,}")
    print(f"\nEntity files in : {ENTITIES_DIR}")
    print("=" * 60)
