"""
Cosine Similarity zwischen Studienorten
=======================================
Vergleicht die SoP-Sprache der 10 Studienorte über Vektoren ihrer
Lexikon-Terme (aus hits.csv), auf zwei Arten:

1. Frequenz-basiert: Vektor = Summe occurrence_count je Term, z.B.
       Oeschinensee = [schön: 45, idyllisch: 12, Juwel: 8, Stille: 23, ...]
       Seealpsee    = [schön: 38, idyllisch: 9,  Juwel: 15, Stille: 19, ...]
   Cosine normiert pro Vektor auf dessen Länge — Standorte mit mehr/weniger
   Artikeln sind also direkt vergleichbar, aber sehr häufig genutzte Terme
   dominieren die Similarity stärker als selten genutzte.

2. Presence-basiert (binär): Vektor = 1 wenn der Term am Standort mind.
   einmal vorkommt, sonst 0, z.B.
       Oeschinensee = [schön: 1, idyllisch: 1, Juwel: 1, Stille: 1, ...]
       Seealpsee    = [schön: 1, idyllisch: 1, Juwel: 1, Stille: 1, ...]
   Ignoriert, wie oft ein Term vorkommt — zählt nur ob das Vokabular-Set
   geteilt wird. Weniger anfällig für Dominanz einzelner Vielnutzer-Terme,
   dafür unempfindlich gegenüber Häufigkeitsunterschieden.

Beide ergeben eine 10x10 Similarity-Matrix.

Verwendung:
    python cosine_similarity_sites.py --hits hits.csv --output figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from visualize_categories_v2 import SITE_ORDER, SITE_DISPLAY
from plot_style import RCPARAMS, FONT_SIZES, HEATMAP_CMAP, heatmap_text_color, save_figure

plt.rcParams.update(RCPARAMS)


def build_term_freq_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Standort x Term Matrix der Treffer-Häufigkeiten (Summe occurrence_count)."""
    matrix = (
        df.groupby(["site", "term_label"])["occurrence_count"]
        .sum().unstack(fill_value=0)
    )
    ordered = [s for s in SITE_ORDER if s in matrix.index]
    return matrix.reindex(ordered)


def build_term_presence_matrix(freq_matrix: pd.DataFrame) -> pd.DataFrame:
    """Standort x Term Matrix, binär: 1 wenn Term am Standort vorkommt, sonst 0."""
    return (freq_matrix > 0).astype(int)


def cosine_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Paarweise Cosine Similarity zwischen den Zeilenvektoren (Standorten)."""
    values = matrix.to_numpy(dtype=float)
    norms  = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Standorte ohne Treffer -> Similarity 0 statt NaN
    normalised = values / norms
    sim = normalised @ normalised.T
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def plot_similarity_heatmap(sim: pd.DataFrame, output_path: Path, title: str) -> None:
    labels = [SITE_DISPLAY.get(s, s) for s in sim.index]
    values = sim.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    # HEATMAP_CMAP geht hell->dunkel: heatmap_text_color() statt der alten,
    # auf viridis (dunkel->hell) zugeschnittenen Schwellenwert-Logik.
    for i in range(len(labels)):
        for j in range(len(labels)):
            val   = values[i, j]
            color = heatmap_text_color(val, vmax=1.0)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=FONT_SIZES["heatmap_cell"], color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_SIZES["cbar_ticks"])
    cbar.set_label("Cosine similarity", fontsize=FONT_SIZES["cbar_label"])
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  → {output_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits",   required=True, help="Pfad zur hits.csv")
    parser.add_argument("--output", default="figures/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCosine Similarity zwischen Studienorten")

    df = pd.read_csv(args.hits, encoding="utf-8")

    term_matrix     = build_term_freq_matrix(df)
    presence_matrix = build_term_presence_matrix(term_matrix)

    sim_freq     = cosine_similarity_matrix(term_matrix)
    sim_presence = cosine_similarity_matrix(presence_matrix)

    term_matrix.rename(index=SITE_DISPLAY).to_csv(
        output_dir / "site_term_freq_matrix.csv", encoding="utf-8")
    print(f"  → site_term_freq_matrix.csv  "
          f"({term_matrix.shape[0]} sites x {term_matrix.shape[1]} terms)")

    presence_matrix.rename(index=SITE_DISPLAY).to_csv(
        output_dir / "site_term_presence_matrix.csv", encoding="utf-8")
    print(f"  → site_term_presence_matrix.csv  "
          f"({presence_matrix.shape[0]} sites x {presence_matrix.shape[1]} terms)")

    sim_freq.rename(index=SITE_DISPLAY, columns=SITE_DISPLAY).to_csv(
        output_dir / "site_cosine_similarity_freq.csv", encoding="utf-8")
    print("  → site_cosine_similarity_freq.csv")

    sim_presence.rename(index=SITE_DISPLAY, columns=SITE_DISPLAY).to_csv(
        output_dir / "site_cosine_similarity_presence.csv", encoding="utf-8")
    print("  → site_cosine_similarity_presence.csv")

    plot_similarity_heatmap(
        sim_freq, output_dir / "12_site_cosine_similarity_freq.png",
        "Cosine Similarity of SoP Term Frequency\nbetween Study Sites")

    plot_similarity_heatmap(
        sim_presence, output_dir / "13_site_cosine_similarity_presence.png",
        "Cosine Similarity of SoP Term Presence\nbetween Study Sites")

    print("\n✓ Fertig.")


if __name__ == "__main__":
    main()
