"""
05_streamlit.py — Streamlit web interface for the Pennington Biomedical KG.

Usage:
    streamlit run 05_streamlit.py

Features:
    - 20 pre-built queries organized by category
    - Interactive Plotly charts (bar, trend line, network, pie)
    - Author collaboration network visualization
    - CSV export of any query result
    - Free-text NL query via Claude API (optional)
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEO4J_URI         = os.getenv("NEO4J_URI",         "bolt://localhost:7687")
NEO4J_USER        = os.getenv("NEO4J_USER",        "neo4j")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD",    "")

# Shared with 05_query.py / nl_query.py so every entry point agrees on the
# API key, the model, and the hard query timeout.
from config import ANTHROPIC_API_KEY, QUERY_TIMEOUT_SECONDS
from nl_query import nl_to_cypher, summarize_results

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pennington KG Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
    }
    .main { background-color: #F8F7F4; }

    .stButton > button {
        background-color: #1B4F72;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #154360;
        color: white;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #1B4F72;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1B4F72;
        font-family: 'DM Serif Display', serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pennington-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86AB 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .pennington-header h1 {
        color: white !important;
        margin: 0;
        font-size: 1.8rem;
    }
    .pennington-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #EEF2F7;
    }
    .sidebar-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #1B4F72;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Neo4j connection (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(
            f"Cannot connect to Neo4j: {e}\n\n"
            "Make sure Docker is running: `docker start neo4j`"
        )
        st.stop()


@st.cache_data(ttl=300)
def run_query(cypher: str, params: dict = None) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {}, timeout=QUERY_TIMEOUT_SECONDS)
        return [dict(r) for r in result]


# ---------------------------------------------------------------------------
# Pre-built query definitions
# ---------------------------------------------------------------------------

QUERIES = {
    "Research Themes": [
        {
            "id": "diseases",
            "title": "Top diseases studied",
            "description": "Most frequently mentioned diseases across all papers",
            "params": {"limit": (10, 50, 15)},
            "cypher": """
                MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'disease'})
                RETURN b.display_name AS Disease, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Disease", "y": "Papers",
        },
        {
            "id": "chemicals",
            "title": "Top chemicals/compounds studied",
            "description": "Most frequently mentioned chemicals and drugs",
            "params": {"limit": (10, 50, 15)},
            "cypher": """
                MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'chemical'})
                RETURN b.display_name AS Chemical, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Chemical", "y": "Papers",
        },
        {
            "id": "genes",
            "title": "Top genes studied",
            "description": "Most frequently mentioned genes",
            "params": {"limit": (10, 50, 15)},
            "cypher": """
                MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'gene'})
                RETURN b.display_name AS Gene, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Gene", "y": "Papers",
        },
        {
            "id": "pub_trends",
            "title": "Publication trends by year",
            "description": "Annual publication volume from 1990 to present",
            "params": {},
            "cypher": """
                MATCH (p:Paper)
                WHERE p.publication_year >= 1990
                RETURN p.publication_year AS Year, count(p) AS Papers
                ORDER BY Year
            """,
            "chart": "line", "x": "Year", "y": "Papers",
        },
        {
            "id": "topics",
            "title": "Top research topics",
            "description": "Most common OpenAlex topic tags",
            "params": {"limit": (10, 50, 15)},
            "cypher": """
                MATCH (p:Paper)-[:TAGGED_WITH]->(c:Concept {concept_type: 'topic'})
                RETURN c.display_name AS Topic, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Topic", "y": "Papers",
        },
    ],
    "Authors": [
        {
            "id": "top_authors",
            "title": "Most prolific authors",
            "description": "Authors with the most papers in the corpus",
            "params": {"limit": (10, 50, 20)},
            "cypher": """
                MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
                RETURN a.name AS Author, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Author", "y": "Papers",
        },
        {
            "id": "author_papers",
            "title": "Papers by a specific author",
            "description": "Search by partial name",
            "params": {"limit": (5, 50, 20), "author_name": ""},
            "cypher": """
                MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
                WHERE all(term IN split(toLower($author_name), ' ')
                          WHERE toLower(a.name) CONTAINS term)
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                RETURN p.title AS Title,
                       p.publication_year AS Year,
                       coalesce(j.name, 'Unknown') AS Journal,
                       p.cited_by_count AS Citations,
                       a.name AS Author
                ORDER BY p.publication_year DESC LIMIT $limit
            """,
            "chart": "table",
        },
        {
            "id": "coauthors",
            "title": "Top co-authors for a researcher",
            "description": "Collaboration network for a given author",
            "params": {"limit": (5, 30, 15), "author_name": ""},
            "cypher": """
                MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
                WHERE all(term IN split(toLower($author_name), ' ')
                          WHERE toLower(a1.name) CONTAINS term)
                  AND a1 <> a2
                RETURN a2.name AS Collaborator, count(p) AS SharedPapers
                ORDER BY SharedPapers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Collaborator", "y": "SharedPapers",
        },
        {
            "id": "author_disease",
            "title": "Authors by disease focus",
            "description": "Who publishes most on a given disease?",
            "params": {"limit": (5, 30, 15), "disease_name": "obesity"},
            "cypher": """
                MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:MENTIONS]->(b:BioEntity)
                WHERE b.entity_type = 'disease'
                  AND toLower(b.display_name) CONTAINS toLower($disease_name)
                RETURN a.name AS Author, count(DISTINCT p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Author", "y": "Papers",
        },
    ],
    "Collaborations": [
        {
            "id": "institutions",
            "title": "Top collaborating institutions",
            "description": "Institutions most frequently co-authoring with Pennington",
            "params": {"limit": (10, 50, 20)},
            "cypher": """
                MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution)
                WHERE i.name <> 'Pennington Biomedical Research Center'
                WITH i, count(DISTINCT a) AS CoAuthors
                ORDER BY CoAuthors DESC LIMIT $limit
                RETURN i.name AS Institution, CoAuthors
            """,
            "chart": "bar", "x": "Institution", "y": "CoAuthors",
        },
        {
            "id": "countries",
            "title": "International collaborations",
            "description": "Countries Pennington collaborates with most",
            "params": {"limit": (10, 40, 20)},
            "cypher": """
                MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution)
                WHERE i.name CONTAINS '('
                WITH split(i.name, '(')[1] AS cr
                WITH replace(cr, ')', '') AS Country
                WHERE Country <> 'United States'
                RETURN Country, count(*) AS Collaborations
                ORDER BY Collaborations DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Country", "y": "Collaborations",
        },
    ],
    "Entity Co-occurrence": [
        {
            "id": "entity_cooccur",
            "title": "Papers mentioning two entities",
            "description": "Find papers where two biomedical terms co-occur",
            "params": {
                "limit": (5, 30, 15),
                "entity1": "leptin",
                "entity2": "obesity",
            },
            "cypher": """
                MATCH (p:Paper)-[:MENTIONS]->(b1:BioEntity),
                      (p)-[:MENTIONS]->(b2:BioEntity)
                WHERE toLower(b1.display_name) CONTAINS toLower($entity1)
                  AND toLower(b2.display_name) CONTAINS toLower($entity2)
                  AND b1 <> b2
                RETURN p.title AS Title,
                       p.publication_year AS Year,
                       p.cited_by_count AS Citations
                ORDER BY Citations DESC LIMIT $limit
            """,
            "chart": "table",
        },
        {
            "id": "disease_gene",
            "title": "Top disease–gene pairs",
            "description": "Most frequently co-mentioned diseases and genes",
            "params": {"limit": (10, 40, 15)},
            "cypher": """
                MATCH (p:Paper)-[:MENTIONS]->(d:BioEntity {entity_type: 'disease'}),
                      (p)-[:MENTIONS]->(g:BioEntity {entity_type: 'gene'})
                RETURN d.display_name AS Disease,
                       g.display_name AS Gene,
                       count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "table",
        },
        {
            "id": "disease_chemicals",
            "title": "Chemicals co-occurring with a disease",
            "description": "What compounds are studied with a specific disease?",
            "params": {"limit": (10, 40, 15), "disease_name": "diabetes"},
            "cypher": """
                MATCH (p:Paper)-[:MENTIONS]->(d:BioEntity {entity_type: 'disease'}),
                      (p)-[:MENTIONS]->(c:BioEntity {entity_type: 'chemical'})
                WHERE toLower(d.display_name) CONTAINS toLower($disease_name)
                RETURN c.display_name AS Chemical, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Chemical", "y": "Papers",
        },
    ],
    "Corpus Statistics": [
        {
            "id": "most_cited",
            "title": "Most cited papers",
            "description": "Top papers by citation count",
            "params": {"limit": (5, 30, 10)},
            "cypher": """
                MATCH (p:Paper)
                WHERE p.cited_by_count > 0
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
                RETURN p.title AS Title,
                       p.publication_year AS Year,
                       p.cited_by_count AS Citations,
                       coalesce(j.name, 'Unknown') AS Journal
                ORDER BY Citations DESC LIMIT $limit
            """,
            "chart": "table",
        },
        {
            "id": "journals",
            "title": "Top publishing journals",
            "description": "Journals where Pennington researchers publish most",
            "params": {"limit": (10, 40, 15)},
            "cypher": """
                MATCH (p:Paper)-[:PUBLISHED_IN]->(j:Journal)
                RETURN j.name AS Journal, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Journal", "y": "Papers",
        },
        {
            "id": "work_types",
            "title": "Publications by type",
            "description": "Articles, reviews, preprints, datasets etc.",
            "params": {},
            "cypher": """
                MATCH (p:Paper)
                RETURN p.work_type AS Type, count(p) AS Count
                ORDER BY Count DESC
            """,
            "chart": "pie", "x": "Type", "y": "Count",
        },
        {
            "id": "funding",
            "title": "Top funding sources",
            "description": "Which funders support Pennington research most?",
            "params": {"limit": (10, 40, 15)},
            "cypher": """
                MATCH (p:Paper)-[:FUNDED_BY]->(g:Grant)
                WHERE g.funder_name <> ''
                RETURN g.funder_name AS Funder, count(p) AS Papers
                ORDER BY Papers DESC LIMIT $limit
            """,
            "chart": "bar", "x": "Funder", "y": "Papers",
        },
    ],
}


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

PENNINGTON_COLORS = [
    "#1B4F72", "#2E86AB", "#A23B72", "#F18F01",
    "#C73E1D", "#3B1F2B", "#44BBA4", "#E94F37",
]


def render_chart(df: pd.DataFrame, query_def: dict) -> None:
    """Render the appropriate chart type for a query result."""
    chart_type = query_def.get("chart", "table")
    x = query_def.get("x")
    y = query_def.get("y")

    if chart_type == "bar" and x and y and x in df.columns and y in df.columns:
        fig = px.bar(
            df, x=y, y=x,
            orientation="h",
            color=y,
            color_continuous_scale=["#AED6F1", "#1B4F72"],
            labels={x: "", y: "Count"},
        )
        fig.update_layout(
            height=max(350, len(df) * 28),
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            font={"family": "DM Sans"},
        )
        fig.update_traces(texttemplate="%{x}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "line" and x and y and x in df.columns and y in df.columns:
        df_sorted = df.sort_values(x)
        df_sorted["Rolling avg"] = df_sorted[y].rolling(5, center=True).mean()
        fig = go.Figure()
        fig.add_bar(
            x=df_sorted[x], y=df_sorted[y],
            marker_color="#AED6F1", name="Annual",
        )
        fig.add_scatter(
            x=df_sorted[x], y=df_sorted["Rolling avg"],
            mode="lines",
            line=dict(color="#1B4F72", width=2),
            name="5-yr avg",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"family": "DM Sans"},
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "pie" and x and y and x in df.columns and y in df.columns:
        fig = px.pie(
            df, names=x, values=y,
            color_discrete_sequence=PENNINGTON_COLORS,
            hole=0.4,
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            font={"family": "DM Sans"},
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_network(rows: list[dict], center_name: str) -> None:
    """Render an author collaboration network using Plotly."""
    if not rows:
        st.info("No collaboration data found.")
        return

    import math

    collaborators = [
        (r.get("Collaborator", ""), r.get("SharedPapers", 0))
        for r in rows if r.get("Collaborator")
    ]

    n = len(collaborators)
    nodes_x, nodes_y = [0.5], [0.5]
    nodes_text = [center_name]
    nodes_size = [30]
    nodes_color = ["#1B4F72"]
    edge_x, edge_y = [], []

    for i, (name, count) in enumerate(collaborators):
        angle = 2 * math.pi * i / n
        r = 0.38
        cx = 0.5 + r * math.cos(angle)
        cy = 0.5 + r * math.sin(angle)
        nodes_x.append(cx)
        nodes_y.append(cy)
        nodes_text.append(f"{name}<br>{count} papers")
        nodes_size.append(max(10, min(25, count * 2)))
        nodes_color.append("#2E86AB")
        edge_x += [0.5, cx, None]
        edge_y += [0.5, cy, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#CCCCCC"),
        hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y,
        mode="markers+text",
        marker=dict(size=nodes_size, color=nodes_color,
                    line=dict(width=1, color="white")),
        text=[t.split("<br>")[0] for t in nodes_text],
        textposition="top center",
        hovertext=nodes_text,
        hoverinfo="text",
        textfont=dict(size=9, family="DM Sans"),
    ))
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# LLM free-text query
# ---------------------------------------------------------------------------
# nl_to_cypher() and summarize_results() are imported from nl_query.py, the
# shared module used by both this app and 05_query.py's CLI free-text mode.
# Keeping one copy means a fix to the schema/prompt (name matching, the
# unbounded-path/network-hang guardrail, etc.) only has to happen once.


# ---------------------------------------------------------------------------
# KG stats for dashboard header (cached 1 hour)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_kg_stats() -> dict:
    try:
        rows = run_query("""
            MATCH (p:Paper)
            WITH count(p) AS papers,
                 sum(p.cited_by_count) AS citations,
                 min(p.publication_year) AS earliest,
                 max(p.publication_year) AS latest
            RETURN papers, citations, earliest, latest
        """)
        authors  = run_query("MATCH (a:Author) RETURN count(a) AS n")[0]["n"]
        entities = run_query("MATCH (b:BioEntity) RETURN count(b) AS n")[0]["n"]
        return {
            "papers":    rows[0]["papers"],
            "citations": rows[0]["citations"],
            "authors":   authors,
            "entities":  entities,
            "earliest":  rows[0]["earliest"],
            "latest":    rows[0]["latest"],
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="pennington-header">
        <h1>🔬 Pennington Biomedical KG Explorer</h1>
        <p>Research intelligence across 35+ years of institutional publications</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Navigation</div>',
                    unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Query Explorer", "💬 Ask a Question"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="sidebar-title">About</div>',
                    unsafe_allow_html=True)
        st.caption(
            "This tool queries the Pennington Biomedical "
            "Knowledge Graph — a Neo4j graph database of "
            "9,000+ institutional publications."
        )
        if ANTHROPIC_API_KEY:
            st.success("Claude API: connected", icon="✅")
        else:
            st.warning(
                "Claude API: not configured\n\n"
                "Add ANTHROPIC_API_KEY to .env for free-text queries.",
                icon="⚠️",
            )

    # ── DASHBOARD ────────────────────────────────────────────────────────
    if page == "📊 Dashboard":
        stats = get_kg_stats()

        if stats:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('papers', 0):,}</div>
                    <div class="metric-label">Publications</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('authors', 0):,}</div>
                    <div class="metric-label">Authors</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('entities', 0):,}</div>
                    <div class="metric-label">Biomedical Entities</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                yr = f"{stats.get('earliest','?')}–{stats.get('latest','?')}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.4rem">{yr}</div>
                    <div class="metric-label">Year Range</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Publication Trends")
            rows = run_query("""
                MATCH (p:Paper) WHERE p.publication_year >= 1990
                RETURN p.publication_year AS Year, count(p) AS Papers ORDER BY Year
            """)
            if rows:
                df = pd.DataFrame(rows).sort_values("Year")
                df["Rolling"] = df["Papers"].rolling(5, center=True).mean()
                fig = go.Figure()
                fig.add_bar(x=df["Year"], y=df["Papers"],
                            marker_color="#AED6F1", name="Annual")
                fig.add_scatter(x=df["Year"], y=df["Rolling"],
                                mode="lines",
                                line=dict(color="#1B4F72", width=2),
                                name="5-yr avg")
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h"),
                    font={"family": "DM Sans"},
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Top Diseases")
            rows = run_query("""
                MATCH (p:Paper)-[:MENTIONS]->(b:BioEntity {entity_type: 'disease'})
                RETURN b.display_name AS Disease, count(p) AS Papers
                ORDER BY Papers DESC LIMIT 12
            """)
            if rows:
                df = pd.DataFrame(rows)
                fig = px.bar(df, x="Papers", y="Disease", orientation="h",
                             color="Papers",
                             color_continuous_scale=["#AED6F1", "#1B4F72"])
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False, plot_bgcolor="white",
                    paper_bgcolor="white", coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending"},
                    font={"family": "DM Sans"},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Authors")
        rows = run_query("""
            MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
            RETURN a.name AS Author, count(p) AS Papers
            ORDER BY Papers DESC LIMIT 15
        """)
        if rows:
            df = pd.DataFrame(rows)
            fig = px.bar(df, x="Author", y="Papers",
                         color="Papers",
                         color_continuous_scale=["#AED6F1", "#1B4F72"])
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False, plot_bgcolor="white",
                paper_bgcolor="white", coloraxis_showscale=False,
                font={"family": "DM Sans"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── QUERY EXPLORER ───────────────────────────────────────────────────
    elif page == "🔍 Query Explorer":
        st.subheader("Query Explorer")
        st.caption("Select a category and query, set parameters, and explore the results.")

        category = st.selectbox("Category", list(QUERIES.keys()))
        query_options = QUERIES[category]
        query_titles = [q["title"] for q in query_options]
        selected_title = st.selectbox("Query", query_titles)
        query_def = next(q for q in query_options if q["title"] == selected_title)

        st.caption(f"_{query_def['description']}_")

        # Parameter inputs
        params = {}
        param_items = list(query_def["params"].items())
        if param_items:
            cols = st.columns(min(3, len(param_items)))
            for idx, (param_name, param_val) in enumerate(param_items):
                col = cols[idx % len(cols)]
                if param_name == "limit":
                    mn, mx, default = param_val
                    params["limit"] = col.slider(
                        "Number of results", mn, mx, default
                    )
                elif isinstance(param_val, str):
                    label = param_name.replace("_", " ").title()
                    params[param_name] = col.text_input(label, value=param_val)

        if st.button("Run Query", type="primary"):
            with st.spinner("Querying graph..."):
                try:
                    rows = run_query(query_def["cypher"], params)
                except Exception as e:
                    st.error(f"Query error: {e}")
                    rows = []

            if not rows:
                st.info("No results found. Try adjusting the parameters.")
            else:
                df = pd.DataFrame(rows)
                tab1, tab2 = st.tabs(["📈 Chart", "📋 Table"])

                with tab1:
                    if query_def["id"] == "coauthors" and params.get("author_name"):
                        render_network(rows, params["author_name"])
                    else:
                        render_chart(df, query_def)

                with tab2:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv,
                        file_name=f"pennington_{query_def['id']}.csv",
                        mime="text/csv",
                    )

    # ── ASK A QUESTION ───────────────────────────────────────────────────
    elif page == "💬 Ask a Question":
        st.subheader("Ask a Question")
        st.caption(
            "Type a research question in plain English. "
            "Claude will translate it to a graph query and summarize the results."
        )

        if not ANTHROPIC_API_KEY:
            st.error(
                "This feature requires an Anthropic API key.\n\n"
                "Add `ANTHROPIC_API_KEY=your_key` to your `.env` file and restart."
            )
            st.stop()

        st.markdown("**Example questions:**")
        examples = [
            "Which authors publish most on caloric restriction?",
            "Find papers about GLP-1 and weight loss published after 2020",
            "What are the most common genes studied with type 2 diabetes?",
            "Which journals publish the most Pennington obesity research?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}"):
                st.session_state["nl_question"] = ex

        question = st.text_area(
            "Your question",
            value=st.session_state.get("nl_question", ""),
            height=80,
            placeholder="e.g. Who are the top researchers on leptin and energy expenditure?",
        )

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Translating question to Cypher..."):
                cypher, error = nl_to_cypher(question)

            if error:
                st.error(f"API error: {error}")
                st.stop()

            if cypher == "CANNOT_ANSWER":
                st.warning(
                    "This question cannot be answered from the available graph data."
                )
                st.stop()

            with st.expander("Generated Cypher query", expanded=False):
                st.code(cypher, language="cypher")

            with st.spinner("Running query..."):
                try:
                    rows = run_query(cypher)
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    st.stop()

            if not rows:
                st.info("No results found.")
            else:
                with st.spinner("Summarizing results..."):
                    summary = summarize_results(question, rows)

                if summary:
                    st.info(f"**Summary:** {summary}")

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=csv,
                    file_name="pennington_query_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
