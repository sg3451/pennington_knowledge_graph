"""
05_query.py — Natural language query interface for the Pennington Biomedical KG.

Pipeline stage: QUERY (runs after 04_load_neo4j.py)

Two-tier query architecture:
    Tier 1 — Pre-built queries (instant, no LLM, always correct)
        A curated library of 20+ high-value research questions mapped
        to optimized Cypher queries. Biologists select from a numbered
        menu. Zero API cost, works offline, fully deterministic.

    Tier 2 — Free-text NL query (uses Claude API as fallback)
        For questions not covered by the menu, Claude translates
        natural language to Cypher, runs it, and summarizes results.
        Requires ANTHROPIC_API_KEY in .env.

Usage:
    python 05_query.py              # interactive menu mode
    python 05_query.py --query "Who publishes most on obesity?"
    python 05_query.py --list       # show all pre-built queries
    python 05_query.py --cypher "MATCH (p:Paper) RETURN count(p)"
"""

import argparse
import json
import os
import sys
import textwrap

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEO4J_URI         = os.getenv("NEO4J_URI",         "bolt://localhost:7687")
NEO4J_USER        = os.getenv("NEO4J_USER",        "neo4j")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD",    "")

# Shared with 05_streamlit.py / nl_query.py so every entry point agrees on
# the API key, the model, and the hard query timeout.
from config import ANTHROPIC_API_KEY, QUERY_TIMEOUT_SECONDS
from nl_query import nl_to_cypher, summarize_results

DIVIDER = "=" * 65


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def get_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to Neo4j: {e}")
        print("Make sure the Docker container is running:  docker start neo4j")
        sys.exit(1)


def run_cypher(driver, query: str, params: dict = None) -> list[dict]:
    """Execute a Cypher query and return results as list of dicts.

    Uses an explicit transaction with `timeout=` so the database itself
    terminates a runaway query after QUERY_TIMEOUT_SECONDS. Note:
    session.run(query, params, timeout=X) does NOT do this - the neo4j
    driver treats an unrecognized kwarg to Session.run() as an extra query
    parameter, not transaction config, so it would silently do nothing.
    """
    with driver.session() as session:
        with session.begin_transaction(timeout=QUERY_TIMEOUT_SECONDS) as tx:
            result = tx.run(query, params or {})
            return [dict(record) for record in result]


# ---------------------------------------------------------------------------
# Safe value helpers
# ---------------------------------------------------------------------------

def sv(val, default="—", maxlen=None):
    """Return a safe string value, truncated if needed."""
    s = str(val) if val is not None else default
    if maxlen and len(s) > maxlen:
        s = s[:maxlen]
    return s


def si(val, default=0):
    """Return a safe integer value."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Pre-built query library
# ---------------------------------------------------------------------------

PREBUILT_QUERIES = [

    # ── RESEARCH THEMES ──────────────────────────────────────────────────
    {
        "id": 1,
        "category": "Research Themes",
        "title": "Top diseases studied at Pennington",
        "description": "Most frequently mentioned diseases across all papers",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'disease'})
            RETURN b.display_name AS disease,
                   count(p) AS paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('disease'), maxlen=40):<42} {si(r.get('paper_count')):,} papers"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 2,
        "category": "Research Themes",
        "title": "Top chemicals/compounds studied",
        "description": "Most frequently mentioned chemicals, drugs, and compounds",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'chemical'})
            RETURN b.display_name AS chemical,
                   count(p) AS paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('chemical'), maxlen=40):<42} {si(r.get('paper_count')):,} papers"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 3,
        "category": "Research Themes",
        "title": "Top genes studied",
        "description": "Most frequently mentioned genes across all papers",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'gene'})
            RETURN b.display_name AS gene,
                   count(p) AS paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('gene'), maxlen=40):<42} {si(r.get('paper_count')):,} papers"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 4,
        "category": "Research Themes",
        "title": "Publication trends by year",
        "description": "How many papers were published each year (1990–present)",
        "params": [],
        "defaults": {},
        "cypher": """
            MATCH (p:Paper)
            WHERE p.publication_year >= 1990
            RETURN p.publication_year AS year,
                   count(p) AS papers
            ORDER BY year
        """,
        "format": lambda rows: "\n".join(
            f"  {sv(r.get('year'))}: {'█' * (si(r.get('papers')) // 20)} {si(r.get('papers'))}"
            for r in rows
        ),
    },
    {
        "id": 5,
        "category": "Research Themes",
        "title": "Top research topics (OpenAlex topics)",
        "description": "Most common research topic tags across the corpus",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:TAGGED_WITH]->(c:Concept {concept_type: 'topic'})
            RETURN c.display_name AS topic,
                   c.field AS field,
                   count(p) AS paper_count
            ORDER BY paper_count DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('topic'), maxlen=45):<47} {si(r.get('paper_count')):,} papers"
            for i, r in enumerate(rows)
        ),
    },

    # ── AUTHORS ───────────────────────────────────────────────────────────
    {
        "id": 6,
        "category": "Authors",
        "title": "Most prolific Pennington authors",
        "description": "Authors with the most papers in the corpus",
        "params": ["limit"],
        "defaults": {"limit": 20},
        "cypher": """
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            RETURN a.name AS author,
                   count(p) AS papers
            ORDER BY papers DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('author'), maxlen=40):<42} {si(r.get('papers')):,} papers"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 7,
        "category": "Authors",
        "title": "Papers by a specific author",
        "description": "Find all papers by an author (partial name search)",
        "params": ["author_name", "limit"],
        "defaults": {"limit": 20},
        "cypher": """
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            WHERE all(term IN split(toLower($author_name), ' ')
                      WHERE toLower(a.name) CONTAINS term)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
            RETURN p.title AS title,
                   p.publication_year AS year,
                   coalesce(j.name, 'Unknown journal') AS journal,
                   p.cited_by_count AS citations,
                   a.name AS author
            ORDER BY p.publication_year DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  [{sv(r.get('year'))}] {sv(r.get('title'), maxlen=70)}\n"
            f"         {sv(r.get('journal'), maxlen=50)}  |  {si(r.get('citations'))} citations"
            for r in rows
        ),
    },
    {
        "id": 8,
        "category": "Authors",
        "title": "Top co-authors for a researcher",
        "description": "Who collaborates most with a given author?",
        "params": ["author_name", "limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
            WHERE all(term IN split(toLower($author_name), ' ')
                      WHERE toLower(a1.name) CONTAINS term)
              AND a1 <> a2
            RETURN a2.name AS collaborator,
                   count(p) AS shared_papers
            ORDER BY shared_papers DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('collaborator'), maxlen=40):<42} {si(r.get('shared_papers'))} shared papers"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 9,
        "category": "Authors",
        "title": "Authors publishing on a specific disease",
        "description": "Who at Pennington publishes most on a given disease?",
        "params": ["disease_name", "limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:MENTIONS]->(b:BioEntity)
            WHERE b.entity_type = 'disease'
              AND toLower(b.display_name) CONTAINS toLower($disease_name)
            RETURN a.name AS author,
                   count(DISTINCT p) AS papers
            ORDER BY papers DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('author'), maxlen=40):<42} {si(r.get('papers'))} papers"
            for i, r in enumerate(rows)
        ),
    },

    # ── COLLABORATIONS ────────────────────────────────────────────────────
    {
        "id": 10,
        "category": "Collaborations",
        "title": "Top collaborating institutions",
        "description": "Institutions most frequently co-authoring with Pennington",
        "params": ["limit"],
        "defaults": {"limit": 20},
        "cypher": """
            MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution)
            WHERE i.name <> 'Pennington Biomedical Research Center'
            WITH i, count(DISTINCT a) AS author_count
            ORDER BY author_count DESC
            LIMIT $limit
            RETURN i.name AS institution,
                   author_count AS pennington_coauthors
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('institution'), maxlen=55):<57} {si(r.get('pennington_coauthors'))} co-authors"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 11,
        "category": "Collaborations",
        "title": "International collaboration countries",
        "description": "Which countries does Pennington collaborate with most?",
        "params": ["limit"],
        "defaults": {"limit": 20},
        "cypher": """
            MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution)
            WHERE i.name CONTAINS '('
            WITH split(i.name, '(')[1] AS country_raw
            WITH replace(country_raw, ')', '') AS country
            WHERE country <> 'United States'
            RETURN country, count(*) AS collaborations
            ORDER BY collaborations DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('country'), maxlen=30):<32} {si(r.get('collaborations'))} collaborations"
            for i, r in enumerate(rows)
        ),
    },

    # ── ENTITY CO-OCCURRENCE ──────────────────────────────────────────────
    {
        "id": 12,
        "category": "Entity Co-occurrence",
        "title": "Papers mentioning two entities together",
        "description": "Find papers where two biomedical terms co-occur",
        "params": ["entity1", "entity2", "limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(b1:BioEntity),
                  (p)-[:MENTIONS]->(b2:BioEntity)
            WHERE toLower(b1.display_name) CONTAINS toLower($entity1)
              AND toLower(b2.display_name) CONTAINS toLower($entity2)
              AND b1 <> b2
            RETURN p.title AS title,
                   p.publication_year AS year,
                   p.cited_by_count AS citations
            ORDER BY p.cited_by_count DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  [{sv(r.get('year'))}] {sv(r.get('title'), maxlen=75)}\n"
            f"         Citations: {si(r.get('citations'))}"
            for r in rows
        ),
    },
    {
        "id": 13,
        "category": "Entity Co-occurrence",
        "title": "Most common disease-gene pairs",
        "description": "Which diseases and genes are most frequently mentioned together?",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(d:BioEntity {entity_type: 'disease'}),
                  (p)-[:MENTIONS]->(g:BioEntity {entity_type: 'gene'})
            RETURN d.display_name AS disease,
                   g.display_name AS gene,
                   count(p) AS co_occurrences
            ORDER BY co_occurrences DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('disease'), maxlen=25):<27} + {sv(r.get('gene'), maxlen=20):<22} {si(r.get('co_occurrences'))} papers"
            for i, r in enumerate(rows)
        ),
    },

    # ── CORPUS STATISTICS ─────────────────────────────────────────────────
    {
        "id": 14,
        "category": "Corpus Statistics",
        "title": "Overall corpus summary",
        "description": "Key statistics about the knowledge graph",
        "params": [],
        "defaults": {},
        "cypher": """
            MATCH (p:Paper)
            WITH count(p) AS total_papers,
                 sum(p.cited_by_count) AS total_citations,
                 avg(p.cited_by_count) AS avg_citations,
                 min(p.publication_year) AS earliest,
                 max(p.publication_year) AS latest,
                 sum(CASE WHEN p.has_full_text THEN 1 ELSE 0 END) AS with_full_text
            RETURN total_papers, total_citations, avg_citations,
                   earliest, latest, with_full_text
        """,
        "format": lambda rows: (
            f"  Total papers       : {si(rows[0].get('total_papers')):,}\n"
            f"  Year range         : {sv(rows[0].get('earliest'))} – {sv(rows[0].get('latest'))}\n"
            f"  Total citations    : {si(rows[0].get('total_citations')):,}\n"
            f"  Avg citations/paper: {rows[0].get('avg_citations', 0):.1f}\n"
            f"  With full text     : {si(rows[0].get('with_full_text')):,}"
        ) if rows else "  No data",
    },
    {
        "id": 15,
        "category": "Corpus Statistics",
        "title": "Most cited papers",
        "description": "Top cited papers in the Pennington corpus",
        "params": ["limit"],
        "defaults": {"limit": 10},
        "cypher": """
            MATCH (p:Paper)
            WHERE p.cited_by_count > 0
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
            RETURN p.title AS title,
                   p.publication_year AS year,
                   p.cited_by_count AS citations,
                   coalesce(j.name, 'Unknown journal') AS journal
            ORDER BY citations DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. [{sv(r.get('year'))}] {sv(r.get('title'), maxlen=65)}\n"
            f"         {sv(r.get('journal'), maxlen=50)}  |  {si(r.get('citations')):,} citations"
            for i, r in enumerate(rows)
        ),
    },
    {
        "id": 16,
        "category": "Corpus Statistics",
        "title": "Publication count by work type",
        "description": "Breakdown of articles, reviews, preprints etc.",
        "params": [],
        "defaults": {},
        "cypher": """
            MATCH (p:Paper)
            RETURN p.work_type AS work_type,
                   count(p) AS count
            ORDER BY count DESC
        """,
        "format": lambda rows: "\n".join(
            f"  {sv(r.get('work_type'), default='unknown'):<20} {si(r.get('count')):,}"
            for r in rows
        ),
    },
    {
        "id": 17,
        "category": "Corpus Statistics",
        "title": "Top journals publishing Pennington research",
        "description": "Journals where Pennington researchers publish most",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:PUBLISHED_IN]->(j:Journal)
            RETURN j.name AS journal,
                   count(p) AS papers
            ORDER BY papers DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('journal'), maxlen=55):<57} {si(r.get('papers')):,} papers"
            for i, r in enumerate(rows)
        ),
    },

    # ── FUNDING ───────────────────────────────────────────────────────────
    {
        "id": 18,
        "category": "Funding",
        "title": "Top funding sources",
        "description": "Which funders support Pennington research most?",
        "params": ["limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:FUNDED_BY]->(g:Grant)
            WHERE g.funder_name <> ''
            RETURN g.funder_name AS funder,
                   count(p) AS papers_funded
            ORDER BY papers_funded DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('funder'), maxlen=55):<57} {si(r.get('papers_funded')):,} papers"
            for i, r in enumerate(rows)
        ),
    },

    # ── DISEASE-SPECIFIC ──────────────────────────────────────────────────
    {
        "id": 19,
        "category": "Disease-Specific",
        "title": "Research on a specific disease over time",
        "description": "How has publication volume on a disease changed by year?",
        "params": ["disease_name"],
        "defaults": {},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'disease'})
            WHERE toLower(b.display_name) CONTAINS toLower($disease_name)
              AND p.publication_year >= 1990
            RETURN p.publication_year AS year,
                   count(p) AS papers
            ORDER BY year
        """,
        "format": lambda rows: "\n".join(
            f"  {sv(r.get('year'))}: {'█' * (si(r.get('papers')) // 3)} {si(r.get('papers'))}"
            for r in rows
        ),
    },
    {
        "id": 20,
        "category": "Disease-Specific",
        "title": "Chemicals co-occurring with a disease",
        "description": "What compounds are studied in the context of a disease?",
        "params": ["disease_name", "limit"],
        "defaults": {"limit": 15},
        "cypher": """
            MATCH (p:Paper)-[:MENTIONS]->(d:BioEntity {entity_type: 'disease'}),
                  (p)-[:MENTIONS]->(c:BioEntity {entity_type: 'chemical'})
            WHERE toLower(d.display_name) CONTAINS toLower($disease_name)
            RETURN c.display_name AS chemical,
                   count(p) AS papers
            ORDER BY papers DESC
            LIMIT $limit
        """,
        "format": lambda rows: "\n".join(
            f"  {i+1:2}. {sv(r.get('chemical'), maxlen=40):<42} {si(r.get('papers'))} papers"
            for i, r in enumerate(rows)
        ),
    },
]


# ---------------------------------------------------------------------------
# LLM fallback — NL to Cypher via Claude API
# ---------------------------------------------------------------------------
# nl_to_cypher() and summarize_results() are imported from nl_query.py, the
# shared module used by both this CLI and 05_streamlit.py's web app. Keeping
# one copy means a fix to the schema/prompt (name matching, the
# unbounded-path/network-hang guardrail, etc.) only has to happen once.


# ---------------------------------------------------------------------------
# Parameter prompting
# ---------------------------------------------------------------------------

def prompt_params(query_def: dict) -> dict | None:
    """Prompt user for any required parameters."""
    params = dict(query_def["defaults"])
    for param in query_def["params"]:
        if param == "limit":
            val = input(f"  How many results? [{params.get('limit', 20)}]: ").strip()
            params["limit"] = int(val) if val.isdigit() else params.get("limit", 20)
        else:
            label = param.replace("_", " ").title()
            val = input(f"  Enter {label}: ").strip()
            if val:
                params[param] = val
            elif param not in params:
                print(f"  [!] {label} is required.")
                return None
    return params


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_header():
    print(f"\n{DIVIDER}")
    print("  Pennington Biomedical Research Center")
    print("  Knowledge Graph Query Interface")
    print(f"{DIVIDER}\n")


def print_menu():
    print("PRE-BUILT QUERIES\n")
    current_category = ""
    for q in PREBUILT_QUERIES:
        if q["category"] != current_category:
            current_category = q["category"]
            print(f"  ── {current_category} ──")
        print(f"  {q['id']:2}. {q['title']}")
    print()
    print("   0.  Free-text question (uses Claude API)")
    print("   q.  Quit")
    print()


def format_generic(rows: list[dict]) -> str:
    """Generic formatter for free-text query results."""
    if not rows:
        return "  No results."
    lines = []
    for i, row in enumerate(rows[:25]):
        lines.append(f"  Row {i+1}:")
        for k, v in row.items():
            val_str = sv(v, maxlen=80)
            lines.append(f"    {k}: {val_str}")
    return "\n".join(lines)


def display_results(query_def: dict, rows: list[dict],
                    question: str = "", cypher: str = "") -> None:
    """Format and display query results."""
    print(f"\n{DIVIDER}")
    title = query_def.get("title", "Results") if query_def else "Results"
    print(f"  {title.upper()}")
    print(f"{DIVIDER}")

    if not rows:
        print("  No results found.")
        print()
        return

    print(f"  {len(rows)} result(s)\n")

    try:
        if query_def and "format" in query_def:
            print(query_def["format"](rows))
        else:
            print(format_generic(rows))
    except Exception as e:
        print(f"  [FORMAT ERROR] {e}")
        print(format_generic(rows))

    # LLM summary for free-text queries
    if question and ANTHROPIC_API_KEY:
        summary = summarize_results(question, rows)
        if summary:
            print(f"\n  ── Summary ──")
            for line in textwrap.wrap(summary, width=60):
                print(f"  {line}")

    print()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def run_interactive(driver):
    """Run the interactive CLI query loop."""
    print_header()
    print("  Type a number to run a pre-built query.")
    print("  Type 0 to ask a free-text question (requires Claude API key).")
    print("  Type 'q' to quit.\n")

    while True:
        print_menu()
        choice = input("  Your choice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\n  Goodbye!\n")
            break

        # Free-text query via Claude
        if choice == "0":
            if not ANTHROPIC_API_KEY:
                print("\n  [!] Free-text queries require ANTHROPIC_API_KEY in .env")
                print("  Add it and restart, or choose a pre-built query.\n")
                continue

            question = input("\n  Your question: ").strip()
            if not question:
                continue

            print("\n  Translating to Cypher...")
            cypher, error = nl_to_cypher(question)

            if error:
                print(f"\n  [ERROR] {error}\n")
                continue

            if cypher == "CANNOT_ANSWER":
                print("\n  This question cannot be answered from the available data.\n")
                continue

            preview = cypher[:120] + "..." if len(cypher) > 120 else cypher
            print(f"\n  Generated Cypher:\n  {preview}\n")

            try:
                rows = run_cypher(driver, cypher)
                mock_def = {"title": question, "format": format_generic}
                display_results(mock_def, rows, question=question, cypher=cypher)
            except Exception as e:
                print(f"\n  [ERROR] Query failed: {e}\n")
            continue

        # Pre-built query by number
        try:
            query_id = int(choice)
        except ValueError:
            print("  Please enter a number.\n")
            continue

        query_def = next((q for q in PREBUILT_QUERIES if q["id"] == query_id), None)
        if not query_def:
            print(f"  No query with ID {query_id}. Choose 1-{len(PREBUILT_QUERIES)} or 0.\n")
            continue

        print(f"\n  {query_def['title']}")
        print(f"  {query_def['description']}\n")

        params = prompt_params(query_def)
        if params is None:
            continue

        try:
            rows = run_cypher(driver, query_def["cypher"], params)
            display_results(query_def, rows)
        except Exception as e:
            print(f"\n  [ERROR] Query failed: {e}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Query the Pennington Biomedical Knowledge Graph"
    )
    p.add_argument("--query", "-q", type=str, default=None,
                   help="Free-text question (uses Claude API)")
    p.add_argument("--cypher", "-c", type=str, default=None,
                   help="Run a raw Cypher query directly")
    p.add_argument("--list",   "-l", action="store_true",
                   help="List all pre-built queries")
    p.add_argument("--id",     type=int, default=None,
                   help="Run a pre-built query by ID non-interactively")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    driver = get_driver()

    try:
        if args.list:
            print_header()
            print_menu()

        elif args.cypher:
            rows = run_cypher(driver, args.cypher)
            print(format_generic(rows))

        elif args.query:
            if not ANTHROPIC_API_KEY:
                print("[ERROR] ANTHROPIC_API_KEY not set in .env")
                sys.exit(1)
            cypher, error = nl_to_cypher(args.query)
            if error:
                print(f"[ERROR] {error}")
            elif cypher == "CANNOT_ANSWER":
                print("This question cannot be answered from the available data.")
            else:
                rows = run_cypher(driver, cypher)
                print(format_generic(rows))
                summary = summarize_results(args.query, rows)
                if summary:
                    print(f"\nSummary: {summary}")

        elif args.id:
            query_def = next(
                (q for q in PREBUILT_QUERIES if q["id"] == args.id), None
            )
            if query_def:
                params = prompt_params(query_def)
                if params:
                    rows = run_cypher(driver, query_def["cypher"], params)
                    display_results(query_def, rows)
            else:
                print(f"No pre-built query with ID {args.id}")

        else:
            run_interactive(driver)

    finally:
        driver.close()
