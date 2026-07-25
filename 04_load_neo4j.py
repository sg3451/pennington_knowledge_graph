"""
04_load_neo4j.py — Load the Pennington Biomedical knowledge graph into Neo4j.

Pipeline stage: LOAD (runs after 03_extract_ner.py)
Input:  data/raw/works_<YYYYMMDD>_filtered.jsonl  — filtered corpus
        data/entities/<paper_id>.entities.json    — NER extraction output
        data/parsed/<paper_id>.sections.json      — full text (where available)
Output: Neo4j graph database (bolt://localhost:7687)

Usage:
    python 04_load_neo4j.py
    python 04_load_neo4j.py --limit 100   # test with 100 papers
    python 04_load_neo4j.py --enrich      # only papers with full text
    python 04_load_neo4j.py --wipe        # clear database first
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RAW_DIR,
    ENTITIES_DIR,
    PARSED_DIR,
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

BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Stable paper ID
# ---------------------------------------------------------------------------

def get_paper_id(record: dict) -> str:
    """
    Compute a stable unique identifier for a paper.
    Priority: PMID > DOI > OpenAlex ID > title hash.
    """
    if record.get("pmid"):
        return f"pmid_{record['pmid']}"
    if record.get("doi"):
        return "doi_" + record["doi"].replace("/", "_").replace(".", "_")
    oa_id = record.get("openalex_id", "")
    if oa_id:
        return oa_id.replace("https://openalex.org/", "W")
    return f"title_{abs(hash(record.get('title', '')))}"


# ---------------------------------------------------------------------------
# Neo4j connection
# ---------------------------------------------------------------------------

def get_driver():
    """Create and verify Neo4j driver connection."""
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        log.info(f"Connected to Neo4j at {NEO4J_URI}")
        return driver
    except Exception as e:
        log.error(f"Cannot connect to Neo4j at {NEO4J_URI}: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Schema setup — drops old constraints first, then creates new ones
# ---------------------------------------------------------------------------

# Old constraints from previous runs that conflict — drop these first
OLD_CONSTRAINTS_TO_DROP = [
    "DROP CONSTRAINT paper_openalex IF EXISTS",
    "DROP CONSTRAINT concept_openalex IF EXISTS",
    "DROP CONSTRAINT grant_award IF EXISTS",
    "DROP CONSTRAINT journal_openalex IF EXISTS",
    "DROP CONSTRAINT paper_id IF EXISTS",
    "DROP CONSTRAINT concept_id IF EXISTS",
    "DROP CONSTRAINT grant_key IF EXISTS",
]

NEW_CONSTRAINTS = [
    "CREATE CONSTRAINT paper_pid IF NOT EXISTS "
    "FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",

    "CREATE CONSTRAINT author_oa IF NOT EXISTS "
    "FOR (a:Author) REQUIRE a.openalex_author_id IS UNIQUE",

    "CREATE CONSTRAINT inst_ror IF NOT EXISTS "
    "FOR (i:Institution) REQUIRE i.ror_id IS UNIQUE",

    "CREATE CONSTRAINT concept_cid IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",

    "CREATE CONSTRAINT bioent_nid IF NOT EXISTS "
    "FOR (b:BioEntity) REQUIRE b.normalized_id IS UNIQUE",

    "CREATE CONSTRAINT grant_gid IF NOT EXISTS "
    "FOR (g:Grant) REQUIRE g.grant_id IS UNIQUE",

    "CREATE CONSTRAINT journal_jid IF NOT EXISTS "
    "FOR (j:Journal) REQUIRE j.journal_id IS UNIQUE",
]

NEW_INDEXES = [
    "CREATE INDEX paper_pmid_idx IF NOT EXISTS FOR (p:Paper) ON (p.pmid)",
    "CREATE INDEX paper_doi_idx IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
    "CREATE INDEX paper_year_idx IF NOT EXISTS FOR (p:Paper) ON (p.publication_year)",
    "CREATE INDEX bioent_type_idx IF NOT EXISTS FOR (b:BioEntity) ON (b.entity_type)",
    "CREATE INDEX author_name_idx IF NOT EXISTS FOR (a:Author) ON (a.name)",
]


def setup_schema(driver) -> None:
    """Drop conflicting old constraints, then create new schema."""
    log.info("Dropping old constraints...")
    with driver.session() as session:
        for stmt in OLD_CONSTRAINTS_TO_DROP:
            try:
                session.run(stmt)
            except Exception:
                pass

    log.info("Creating new constraints and indexes...")
    with driver.session() as session:
        for stmt in NEW_CONSTRAINTS + NEW_INDEXES:
            try:
                session.run(stmt)
            except Exception as e:
                log.debug(f"Schema statement note: {e}")

    log.info("Schema setup complete")


def wipe_database(driver) -> None:
    """Delete all nodes and relationships (keeps schema)."""
    log.warning("Wiping all nodes and relationships...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    log.info("Database wiped")


# ---------------------------------------------------------------------------
# Corpus loading helpers
# ---------------------------------------------------------------------------

def find_latest_filtered_file() -> Path:
    candidates = sorted(RAW_DIR.glob("works_*_filtered.jsonl"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No filtered JSONL in {RAW_DIR}")
    return candidates[0]


def load_entities(paper_id: str) -> list[dict]:
    path = ENTITIES_DIR / f"{paper_id}.entities.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("entities", [])
    except Exception:
        return []


def load_sections(paper_id: str) -> dict:
    path = PARSED_DIR / f"{paper_id}.sections.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("sections", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Cypher upserts — all keyed on stable IDs
# ---------------------------------------------------------------------------

def upsert_paper(session, record: dict, paper_id: str, sections: dict) -> None:
    """Merge Paper node using paper_id as the unique key."""
    session.run("""
        MERGE (p:Paper {paper_id: $pid})
        ON CREATE SET
            p.openalex_id   = $openalex_id,
            p.doi           = $doi,
            p.pmid          = $pmid,
            p.title         = $title,
            p.publication_year = $year,
            p.work_type     = $work_type,
            p.cited_by_count = $cited_by,
            p.abstract      = $abstract,
            p.is_oa         = $is_oa,
            p.oa_status     = $oa_status,
            p.has_full_text = $has_ft,
            p._loaded_at    = $ts
        ON MATCH SET
            p.cited_by_count = $cited_by,
            p.has_full_text  = $has_ft,
            p._updated_at    = $ts
    """, {
        "pid":        paper_id,
        "openalex_id": record.get("openalex_id", ""),
        "doi":        record.get("doi", ""),
        "pmid":       record.get("pmid", ""),
        "title":      record.get("title", ""),
        "year":       record.get("publication_year"),
        "work_type":  record.get("work_type", ""),
        "cited_by":   record.get("cited_by_count", 0),
        "abstract":   record.get("abstract", ""),
        "is_oa":      record.get("is_oa", False),
        "oa_status":  record.get("oa_status", ""),
        "has_ft":     bool(sections),
        "ts":         datetime.utcnow().isoformat(),
    })


def upsert_journal(session, record: dict, paper_id: str) -> None:
    """Merge Journal node and link Paper to it."""
    jname = record.get("journal_name", "")
    if not jname:
        return
    # Use journal name as key if no OpenAlex ID
    jid = record.get("journal_openalex_id", "") or f"name_{jname[:50]}"
    session.run("""
        MERGE (j:Journal {journal_id: $jid})
        ON CREATE SET j.name = $name, j.issn = $issn, j.openalex_id = $oa_id
        ON MATCH SET  j.name = $name
        WITH j
        MATCH (p:Paper {paper_id: $pid})
        MERGE (p)-[:PUBLISHED_IN]->(j)
    """, {
        "jid":  jid,
        "name": jname,
        "issn": record.get("journal_issn", "") or "",
        "oa_id": record.get("journal_openalex_id", ""),
        "pid":  paper_id,
    })


def upsert_authors(session, record: dict, paper_id: str) -> None:
    """Merge Author + Institution nodes and link to Paper."""
    for auth in record.get("authorships", []):
        aid = auth.get("openalex_author_id", "")
        if not aid:
            continue
        session.run("""
            MERGE (a:Author {openalex_author_id: $aid})
            ON CREATE SET a.name = $name, a.orcid = $orcid
            ON MATCH SET  a.name = $name
            WITH a
            MATCH (p:Paper {paper_id: $pid})
            MERGE (p)-[r:AUTHORED_BY]->(a)
            ON CREATE SET r.position = $pos, r.is_corresponding = $corr
        """, {
            "aid":  aid,
            "name": auth.get("author_name", ""),
            "orcid": auth.get("orcid", ""),
            "pid":  paper_id,
            "pos":  auth.get("author_position", ""),
            "corr": auth.get("is_corresponding", False),
        })

        for aff in auth.get("affiliations", []):
            ror = aff.get("ror_id", "")
            if not ror:
                continue
            session.run("""
                MERGE (i:Institution {ror_id: $ror})
                ON CREATE SET i.name = $name, i.openalex_id = $oa_id
                ON MATCH SET  i.name = $name
                WITH i
                MATCH (a:Author {openalex_author_id: $aid})
                MERGE (a)-[:AFFILIATED_WITH]->(i)
            """, {
                "ror":   ror,
                "name":  aff.get("institution_name", ""),
                "oa_id": aff.get("openalex_id", ""),
                "aid":   aid,
            })


def upsert_concepts(session, record: dict, paper_id: str) -> None:
    """Merge Concept nodes (topics + MeSH) and link to Paper."""
    for topic in record.get("topics", []):
        cid = topic.get("openalex_id", "")
        if not cid:
            continue
        session.run("""
            MERGE (c:Concept {concept_id: $cid})
            ON CREATE SET
                c.display_name = $name,
                c.field        = $field,
                c.domain       = $domain,
                c.concept_type = 'topic'
            ON MATCH SET c.display_name = $name
            WITH c
            MATCH (p:Paper {paper_id: $pid})
            MERGE (p)-[r:TAGGED_WITH]->(c)
            ON CREATE SET r.score = $score
        """, {
            "cid":   cid,
            "name":  topic.get("display_name", ""),
            "field": topic.get("field", ""),
            "domain": topic.get("domain", ""),
            "score": topic.get("score", 0.0),
            "pid":   paper_id,
        })

    for mesh in record.get("mesh", []):
        dui = mesh.get("descriptor_ui", "")
        if not dui:
            continue
        cid = f"MESH:{dui}"
        session.run("""
            MERGE (c:Concept {concept_id: $cid})
            ON CREATE SET
                c.display_name = $name,
                c.concept_type = 'mesh'
            ON MATCH SET c.display_name = $name
            WITH c
            MATCH (p:Paper {paper_id: $pid})
            MERGE (p)-[r:TAGGED_WITH]->(c)
            ON CREATE SET r.is_major_topic = $major
        """, {
            "cid":   cid,
            "name":  mesh.get("descriptor_name", ""),
            "major": mesh.get("is_major_topic", False),
            "pid":   paper_id,
        })


def upsert_bioentities(session, paper_id: str, entities: list[dict]) -> None:
    """Merge BioEntity nodes and link Paper via MENTIONS."""
    if not entities:
        return
    # Deduplicate by normalized_id
    seen: dict[str, dict] = {}
    for e in entities:
        nid = e.get("normalized_id", "")
        if not nid:
            continue
        if nid not in seen:
            seen[nid] = {
                "nid":   nid,
                "etype": e.get("entity_type", "biomedical"),
                "name":  e.get("text", ""),
                "src":   e.get("source", ""),
                "count": 0,
            }
        seen[nid]["count"] += 1

    for nid, ent in seen.items():
        session.run("""
            MERGE (b:BioEntity {normalized_id: $nid})
            ON CREATE SET
                b.entity_type  = $etype,
                b.display_name = $name,
                b.source       = $src
            ON MATCH SET
                b.display_name = CASE
                    WHEN b.display_name = '' THEN $name
                    ELSE b.display_name END
            WITH b
            MATCH (p:Paper {paper_id: $pid})
            MERGE (p)-[r:MENTIONS]->(b)
            ON CREATE SET r.count = $cnt
            ON MATCH SET  r.count = r.count + $cnt
        """, {
            "nid":   nid,
            "etype": ent["etype"],
            "name":  ent["name"],
            "src":   ent["src"],
            "pid":   paper_id,
            "cnt":   ent["count"],
        })


def upsert_grants(session, record: dict, paper_id: str) -> None:
    """Merge Grant nodes and link Paper to them."""
    for award in record.get("awards", []):
        award_id  = award.get("award_id", "")
        funder_id = award.get("funder_openalex_id", "")
        if not award_id and not funder_id:
            continue
        gid = award_id if award_id else funder_id
        session.run("""
            MERGE (g:Grant {grant_id: $gid})
            ON CREATE SET
                g.award_id          = $award_id,
                g.funder_name       = $fname,
                g.funder_openalex_id = $fid
            ON MATCH SET g.funder_name = $fname
            WITH g
            MATCH (p:Paper {paper_id: $pid})
            MERGE (p)-[:FUNDED_BY]->(g)
        """, {
            "gid":      gid,
            "award_id": award_id,
            "fname":    award.get("funder_display_name", ""),
            "fid":      funder_id,
            "pid":      paper_id,
        })


def upsert_citations(session, record: dict,
                     paper_id: str, corpus_ids: set) -> None:
    """Create CITES relationships between corpus papers."""
    for ref_id in record.get("referenced_works", []):
        if ref_id in corpus_ids:
            session.run("""
                MATCH (p:Paper {paper_id: $pid})
                MATCH (r:Paper {openalex_id: $rid})
                MERGE (p)-[:CITES]->(r)
            """, {"pid": paper_id, "rid": ref_id})


# ---------------------------------------------------------------------------
# Main load loop
# ---------------------------------------------------------------------------

def run_load(input_path: Path, driver,
             limit=None, enrich_only=False, wipe=False) -> dict:

    if wipe:
        wipe_database(driver)

    setup_schema(driver)

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

    if enrich_only:
        records = [r for r in records
                   if (PARSED_DIR / f"{get_paper_id(r)}.sections.json").exists()]
        log.info(f"Enrich mode: {len(records):,} papers with full text")

    if limit:
        records = records[:limit]
        log.info(f"Limiting to first {limit} records")

    corpus_oa_ids = {r.get("openalex_id", "") for r in records
                     if r.get("openalex_id")}
    log.info(f"Corpus OpenAlex IDs indexed: {len(corpus_oa_ids):,}")

    stats = {"total": len(records), "loaded": 0,
             "errors": 0, "with_entities": 0, "with_full_text": 0}

    with driver.session() as session:
        for record in tqdm(records, desc="Loading to Neo4j", unit="paper"):
            try:
                pid      = get_paper_id(record)
                entities = load_entities(pid)
                sections = load_sections(pid)

                upsert_paper(session, record, pid, sections)
                upsert_journal(session, record, pid)
                upsert_authors(session, record, pid)
                upsert_concepts(session, record, pid)
                upsert_grants(session, record, pid)
                if entities:
                    upsert_bioentities(session, pid, entities)
                upsert_citations(session, record, pid, corpus_oa_ids)

                stats["loaded"] += 1
                if entities:
                    stats["with_entities"] += 1
                if sections:
                    stats["with_full_text"] += 1

            except Exception as e:
                log.warning(f"Failed {get_paper_id(record)}: {e}")
                stats["errors"] += 1

            if stats["loaded"] % 500 == 0 and stats["loaded"] > 0:
                log.info(f"Progress: {stats['loaded']:,} loaded | "
                         f"{stats['errors']:,} errors")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Load Pennington KG into Neo4j.")
    p.add_argument("--input",  type=str,   default=None)
    p.add_argument("--limit",  type=int,   default=None)
    p.add_argument("--enrich", action="store_true")
    p.add_argument("--wipe",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            log.error(f"Not found: {args.input}")
            sys.exit(1)
    else:
        try:
            input_path = find_latest_filtered_file()
            log.info(f"Auto-detected input: {input_path.name}")
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(1)

    driver = get_driver()
    try:
        stats = run_load(input_path, driver,
                         limit=args.limit,
                         enrich_only=args.enrich,
                         wipe=args.wipe)
    finally:
        driver.close()

    print("\n" + "=" * 60)
    print("NEO4J LOAD REPORT")
    print("=" * 60)
    print(f"Total records      : {stats['total']:,}")
    print(f"Successfully loaded: {stats['loaded']:,}")
    print(f"With entities      : {stats['with_entities']:,}")
    print(f"With full text     : {stats['with_full_text']:,}")
    print(f"Errors             : {stats['errors']:,}")
    print(f"\nGraph accessible at: http://localhost:7474")
    print("=" * 60)
