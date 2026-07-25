"""
nl_query.py — Shared free-text NL -> Cypher query logic for the Pennington
Biomedical KG query interfaces.

Both 05_query.py (CLI) and 05_streamlit.py (web app) offer a Tier 2
free-text question mode that uses the Claude API to translate a plain
English question into Cypher, run it, and summarize the results. Until
now each script had its own copy of the schema description, prompt rules,
and API calls — which meant a fix to one (e.g. the name-matching bug, or
the unbounded-path/network-hang guardrail) had to be remembered and
re-applied to the other. This module is the single source of truth for
that logic; both scripts should import from here rather than
re-implementing it locally.

Usage:
    from nl_query import nl_to_cypher, summarize_results

    cypher, error = nl_to_cypher("Who publishes most on obesity?")
    ...
    summary = summarize_results(question, rows)
"""

import json
import urllib.request

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

# ---------------------------------------------------------------------------
# Graph schema description given to Claude for NL -> Cypher translation
# ---------------------------------------------------------------------------
GRAPH_SCHEMA = """
Node labels and key properties:
  Paper       (paper_id, title, publication_year, work_type, cited_by_count,
               abstract, is_oa, has_full_text, doi, pmid)
  Author      (openalex_author_id, name, orcid)
  Institution (ror_id, name)
  Concept     (concept_id, display_name, concept_type ['topic','mesh'], field, domain)
  BioEntity   (normalized_id, entity_type ['gene','disease','chemical','species',
               'cell_line','biomedical'], display_name, source)
  Grant       (grant_id, funder_name, award_id)
  Journal     (journal_id, name, issn)

Relationships:
  (Paper)-[:AUTHORED_BY {position, is_corresponding}]->(Author)
  (Author)-[:AFFILIATED_WITH]->(Institution)
  (Paper)-[:TAGGED_WITH {score, is_major_topic}]->(Concept)
  (Paper)-[:MENTIONS {count}]->(BioEntity)
  (Paper)-[:FUNDED_BY]->(Grant)
  (Paper)-[:PUBLISHED_IN]->(Journal)

Important notes:
  - Paper nodes do NOT have a journal_name property
  - Journal names are on Journal nodes, linked via PUBLISHED_IN
  - Always use OPTIONAL MATCH for journal lookups
  - All papers are from Pennington Biomedical Research Center
  - Use toLower() for case-insensitive text matching
  - Always include LIMIT (max 25)
  - Return clean column names (no special characters)

CRITICAL RULES:
  - When matching a person's name, NEVER use a single CONTAINS on the full
    string (a stored name like "Philip R. Schauer" will not contain the
    substring "Philip Schauer" because the middle initial breaks it).
    Instead split the supplied name on spaces and require every resulting
    word to be present in a.name, e.g.:
      WHERE all(term IN split(toLower('Philip Schauer'), ' ')
                WHERE toLower(a.name) CONTAINS term)
  - NEVER use an unbounded or wide variable-length relationship pattern
    (e.g. -[*]-, -[*..6]-, or any hop count above 3) to connect two named
    authors or build a "network"/"connection" between people. If asked for
    a network or connection between two specific named people, instead:
      (a) MATCH each author separately using the name-matching rule above,
      (b) find shared papers via a single AUTHORED_BY hop on each side, and
      (c) return the two author names plus the list/count of shared papers
          or shared co-authors — do not attempt a graph traversal between
          the two Author nodes.
  - Every query runs under a hard server-side timeout (see
    config.QUERY_TIMEOUT_SECONDS) — keep traversals narrow and always
    filter before expanding relationships.
"""


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from Claude's response, if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()
    return text


def nl_to_cypher(question: str) -> tuple[str, str]:
    """Use Claude API to translate a natural language question to Cypher.

    Returns (cypher, error). On success error is "". If the question
    cannot be answered from the schema, cypher is the literal string
    "CANNOT_ANSWER".
    """
    if not ANTHROPIC_API_KEY:
        return "", "ANTHROPIC_API_KEY not set in .env — cannot use LLM fallback."

    prompt = (
        f"You are a Neo4j Cypher expert for a biomedical knowledge graph.\n\n"
        f"Graph schema:\n{GRAPH_SCHEMA}\n\n"
        f'Translate this question to a valid Cypher query: "{question}"\n\n'
        "Return ONLY the raw Cypher query with no explanation, "
        "no markdown formatting, and no code fences. "
        "If the question cannot be answered from this schema, return: CANNOT_ANSWER"
    )

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            cypher = strip_code_fences(data["content"][0]["text"])
            return cypher, ""
    except Exception as e:
        return "", f"Claude API error: {e}"


def summarize_results(question: str, results: list[dict]) -> str:
    """Use Claude to summarize query results in plain English."""
    if not ANTHROPIC_API_KEY or not results:
        return ""

    results_str = json.dumps(results[:20], indent=2)
    prompt = (
        f'A researcher asked: "{question}"\n\n'
        f"The following data was retrieved from the Pennington Biomedical "
        f"knowledge graph:\n{results_str}\n\n"
        "Write a concise, plain-English summary of these results (2-4 sentences). "
        "Focus on what is most interesting or significant. Be specific with numbers. "
        "Do not mention Cypher or databases."
    )

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()
    except Exception:
        return ""
