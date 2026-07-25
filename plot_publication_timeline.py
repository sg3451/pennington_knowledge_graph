"""
plot_publication_timeline.py — Publication timeline visualization for the
Pennington Biomedical corpus.

Generates a polished bar chart showing publication counts by year,
with annotations for key institutional milestones.

Output: data/figures/publication_timeline.png

Usage:
    python plot_publication_timeline.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import RAW_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

def load_year_counts(min_year: int = 1988) -> tuple[list[int], list[int], dict]:
    """Load publication year counts from the filtered corpus."""
    candidates = sorted(RAW_DIR.glob("works_*_filtered.jsonl"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No filtered JSONL found. Run 01b_filter.py first.")

    input_path = candidates[0]
    print(f"Reading: {input_path.name}")

    year_counts = Counter()
    work_type_by_year: dict[int, Counter] = {}

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yr = r.get("publication_year")
            if yr and int(yr) >= min_year:
                yr = int(yr)
                year_counts[yr] += 1
                if yr not in work_type_by_year:
                    work_type_by_year[yr] = Counter()
                work_type_by_year[yr][r.get("work_type", "other")] += 1

    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]
    return years, counts, work_type_by_year


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_timeline(
    years: list[int],
    counts: list[int],
    work_type_by_year: dict,
    output_path: Path,
) -> None:
    """Generate and save the publication timeline chart."""

    # Color scheme — stacked by article vs review vs other
    COLOR_ARTICLE  = "#1B6CA8"   # deep blue
    COLOR_REVIEW   = "#2CA05A"   # green
    COLOR_PREPRINT = "#E8A838"   # amber
    COLOR_OTHER    = "#B0BEC5"   # light grey

    # Build stacked arrays
    articles  = [work_type_by_year.get(y, Counter()).get("article", 0)  for y in years]
    reviews   = [work_type_by_year.get(y, Counter()).get("review", 0)   for y in years]
    preprints = [work_type_by_year.get(y, Counter()).get("preprint", 0) for y in years]
    others    = [
        counts[i] - articles[i] - reviews[i] - preprints[i]
        for i in range(len(years))
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    x = np.arange(len(years))
    bar_width = 0.85

    # Stacked bars
    b1 = ax.bar(x, articles,  bar_width, color=COLOR_ARTICLE,  label="Article",  zorder=3)
    b2 = ax.bar(x, reviews,   bar_width, bottom=articles,
                color=COLOR_REVIEW,   label="Review",   zorder=3)
    b3 = ax.bar(x, preprints, bar_width,
                bottom=[articles[i] + reviews[i] for i in range(len(years))],
                color=COLOR_PREPRINT, label="Preprint", zorder=3)
    b4 = ax.bar(x, others,    bar_width,
                bottom=[articles[i] + reviews[i] + preprints[i] for i in range(len(years))],
                color=COLOR_OTHER,    label="Other",    zorder=3)

    # Gridlines
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # X axis — show every 5 years to avoid crowding
    tick_positions = [i for i, y in enumerate(years) if y % 5 == 0]
    tick_labels = [str(years[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=13, rotation=45, ha="right")
    ax.tick_params(axis="x", length=0)

    # Y axis
    ax.set_ylabel("Publications per year", fontsize=14, labelpad=10)
    ax.tick_params(axis="y", labelsize=13)

    # Spine cleanup
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")

    # Title and subtitle
    total = sum(counts)
    fig.text(
        0.13, 0.96,
        "Pennington Biomedical Research Center — Publication Output",
        fontsize=15, fontweight="bold", color="#1A1A2E",
    )
    fig.text(
        0.13, 0.91,
        f"Source: OpenAlex  ·  {total:,} publications  ·  {min(years)}–{max(years)}",
        fontsize=10, color="#666666",
    )

    # Milestone annotations — only confirmed institutional facts
    milestones = [
        (1988, "PBRC\nfounded"),
    ]

    for yr, label in milestones:
        if yr in years:
            xi = years.index(yr)
            yval = counts[xi]
            ax.annotate(
                label,
                xy=(xi, yval),
                xytext=(xi, yval + max(counts) * 0.08),
                fontsize=8,
                color="#555555",
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )

    # 5-year average trend line
    window = 5
    smoothed = np.convolve(counts, np.ones(window) / window, mode="same")
    ax.plot(
        x, smoothed,
        color="#E53935", linewidth=1.8, linestyle="--",
        alpha=0.7, label=f"{window}-yr moving avg", zorder=4
    )

    # Legend
    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor="white",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save PNG
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"PNG saved to: {output_path}")

    # Save PDF (vector format — ideal for publications and presentations)
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"PDF saved to: {pdf_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    years, counts, work_type_by_year = load_year_counts(min_year=1988)
    output_path = FIGURES_DIR / "publication_timeline.png"
    plot_timeline(years, counts, work_type_by_year, output_path)

    print(f"\nSummary:")
    print(f"  Years covered : {min(years)}–{max(years)}")
    print(f"  Total papers  : {sum(counts):,}")
    print(f"  Peak year     : {years[counts.index(max(counts))]} ({max(counts)} papers)")
