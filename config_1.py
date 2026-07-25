"""
config.py — Shared configuration for the Pennington Biomedical KG pipeline.

Loads credentials from .env and exposes paths and constants used
across all pipeline stages. Import this module at the top of every
pipeline script instead of hard-coding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one directory up from any subdirectory,
# or the current directory if running scripts from the project root).
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
ENTITIES_DIR = DATA_DIR / "entities"

# Ensure directories exist on first import
for _dir in [RAW_DIR, PARSED_DIR, ENTITIES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Institution constants
# ---------------------------------------------------------------------------
PENNINGTON_ROR_ID = "https://ror.org/01vx35703"
PENNINGTON_OPENALEX_ID = "I2800723778"  # OpenAlex institution ID for Pennington

# ---------------------------------------------------------------------------
# OpenAlex API
# ---------------------------------------------------------------------------
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")  # for polite pool access

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---------------------------------------------------------------------------
# GROBID
# ---------------------------------------------------------------------------
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")

# ---------------------------------------------------------------------------
# Anthropic / Claude API
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Ingest settings
# ---------------------------------------------------------------------------
# Maximum papers to retrieve per run (set to None for full corpus)
INGEST_MAX_RESULTS = None

# OpenAlex cursor pagination page size (max allowed: 200)
INGEST_PAGE_SIZE = 200

# Fields to request from OpenAlex Works endpoint.
# Keeping this selective reduces response size and speeds up pagination.
OPENALEX_SELECT_FIELDS = ",".join([
    "doi",
    "ids",                  # includes pmid, pmcid
    "title",
    "publication_year",
    "publication_date",
    "type",
    "open_access",
    "authorships",          # author name, ORCID, institution affiliation
    "cited_by_count",
    "biblio",               # volume, issue, pages
    "primary_location",     # journal/source info
    "concepts",             # OpenAlex concept tags with scores
    "topics",               # newer topic tags (more granular than concepts)
    "keywords",
    "mesh_terms",           # MeSH headings directly from PubMed
    "grants",               # funder + award ID
    "referenced_works",     # outgoing citations (DOI list)
    "abstract_inverted_index",  # pyalex converts this to plaintext on the fly
])
