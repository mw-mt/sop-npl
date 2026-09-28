"""
Wordcloud-Generator
=======================================
Erzeugt Wordclouds aus den CSV-Ergebnissen von query_hits.py
(--query top_terms bzw. --query top_terms_per_site).

Wörter werden nach Kategorie eingefärbt (CATEGORY_COLORS aus plot_style.py —
eigene Palette, anderer Grundton als die Standort-Farben in
visualize_categories_v2.py, aber gleicher gedeckter "Vibe"), Größe =
total_occurrences. Schriftart: Segoe UI (modern, serifenlos).

Voraussetzung:
    pip install wordcloud

Verwendung:
    # zuerst die Query-Ergebnisse erzeugen
    python query_hits.py --hits hits.csv --articles articles.csv --query top_terms --output results/
    python query_hits.py --hits hits.csv --articles articles.csv --query top_terms_per_site --output results/

    # dann Wordclouds daraus bauen
    python generate_wordclouds.py \
        --top-terms results/top_terms.csv \
        --top-terms-per-site results/top_terms_per_site.csv \
        --output figures/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud

from visualize_categories_v2 import SITE_ORDER, SITE_DISPLAY, CATEGORY_ORDER, CATEGORY_EN
from plot_style import RCPARAMS, FONT_SIZES, CATEGORY_COLORS, save_figure, get_wordcloud_font_path

plt.rcParams.update(RCPARAMS)


def make_color_func(term_to_category: dict):
    """Liefert eine color_func für WordCloud, die pro Wort dessen Kategorie-Farbe (CATEGORY_COLORS) zurückgibt."""
    def color_func(word, font_size=None, position=None, orientation=None,
                   random_state=None, **kwargs):
        cat = term_to_category.get(word)
        return CATEGORY_COLORS.get(cat, "#333333")
    return color_func


def make_wordcloud(freq: dict, color_func, width: int, height: int, font_path: str | None) -> WordCloud:
    wc = WordCloud(
        width=width, height=height,
        background_color="white",
        color_func=color_func,
        font_path=font_path,
        prefer_horizontal=0.9,
        max_words=len(freq) if freq else 1,
        relative_scaling=0.5,
    )
    wc.generate_from_frequencies(freq)
    return wc


def add_category_legend(fig, cats_present: list, legend_y: float, ncol: int):
    """Platziert die Kategorie-Legende innerhalb der Figure-Fläche (0..1).
    Gibt das Legend-Objekt zurück (für die Abstandsmessung, s. legend_top_fraction).
    """
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in cats_present]
    labels  = [CATEGORY_EN.get(c, c) for c in cats_present]
    legend = fig.legend(handles, labels, title="Category", loc="lower center",
                         ncol=ncol, bbox_to_anchor=(0.5, legend_y))
    return legend


def legend_top_fraction(fig, legend) -> float:
    """
    Tatsächliche Oberkante der gerenderten Legende in Figure-Fraction-
    Koordinaten (0..1) — gemessen statt geschätzt. Damit lässt sich der
    Achsen-Rand exakt an die Legende anschliessen, egal wie viele Spalten/
    Zeilen/welche Fontgrösse sie hat (vorher: fester Wert pro Zeile, der bei
    ncol=3 zu grosszügig war und eine sichtbare Lücke Bild<->Legende liess).
    """
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer())
    return bbox.transformed(fig.transFigure.inverted()).y1


def plot_overall(df: pd.DataFrame, output_path: Path, font_path: str | None) -> None:
    freq        = df.groupby("term_label")["total_occurrences"].sum().to_dict()
    term_to_cat = dict(zip(df["term_label"], df["category"]))
    color_func  = make_color_func(term_to_cat)
    wc          = make_wordcloud(freq, color_func, width=1600, height=900, font_path=font_path)

    cats_present = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    # Wenige Spalten -> Legende schmaler und höher, damit sie nicht breiter
    # als der Wordcloud-Inhalt wird und das Bild künstlich verbreitert
    # (bbox_inches='tight' würde sonst den Canvas an die Legende anpassen).
    ncol = min(len(cats_present), 3)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent SoP Terms (overall)", fontweight="bold")

    legend = add_category_legend(fig, cats_present, legend_y=0.01, ncol=ncol)
    # Achsen-Unterkante exakt an die gemessene Legenden-Oberkante anschliessen
    # (statt an einen pauschalen Schätzwert) — schliesst die Lücke zwischen
    # Wordcloud und Legende.
    bottom = legend_top_fraction(fig, legend) + 0.02
    fig.subplots_adjust(bottom=bottom)

    save_figure(fig, output_path)
    print(f"  → {output_path.name}")


def plot_per_site_grid(df: pd.DataFrame, output_path: Path, font_path: str | None) -> None:
    """
    Grid mit einer Wordcloud pro Standort. Layout: 2 Spalten x 5 Zeilen
    (statt 5x2) — passt damit besser auf eine vertikale A4-Seite.
    """
    sites_ordered = [s for s in SITE_ORDER if s in df["site"].unique()]
    n_sites       = len(sites_ordered)
    ncols         = 2
    nrows         = (n_sites + ncols - 1) // ncols

    cats_present = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    # Wenige Spalten -> Legende schmaler und höher (s. plot_overall).
    ncol         = min(len(cats_present), 3)
    legend_rows  = -(-len(cats_present) // ncol)
    # Feste Ränder oben (Suptitle) und unten (Legende), Rest verteilt sich auf die Achsen.
    # top_margin muss sowohl den Suptitle (fontsize 16) als auch das automatische
    # Padding der Achsen-Titel der ersten Zeile (fontsize 14) aufnehmen — sonst
    # überlappen sich beide (das Achsen-Titel-Padding liegt ausserhalb der
    # per subplots_adjust(top=...) reservierten Achsenfläche).
    # bottom_margin ist hier nur eine grobe Schätzung für die Canvas-Grösse —
    # der tatsächliche Achsen/Legende-Abstand wird weiter unten exakt
    # vermessen (legend_top_fraction), überschüssiger Platz unterhalb der
    # Legende wird von save_figure()/bbox_inches='tight' weggeschnitten.
    top_margin    = 0.9                         # Zoll für Suptitle + Titel-Padding Zeile 1
    bottom_margin = 0.5 + 0.3 * legend_rows      # grobe Schätzung, s.o.
    row_height    = 3.0                         # Zoll pro Standort-Zeile
    col_width     = 6.0                         # Zoll pro Spalte
    fig_width     = ncols * col_width
    fig_height    = nrows * row_height + top_margin + bottom_margin

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # Panel-Seitenverhältnis der Wordcloud-Bitmaps an col_width:row_height anpassen
    wc_width, wc_height = 900, 450

    for i, site in enumerate(sites_ordered):
        sub         = df[df["site"] == site]
        freq        = sub.groupby("term_label")["total_occurrences"].sum().to_dict()
        term_to_cat = dict(zip(sub["term_label"], sub["category"]))
        color_func  = make_color_func(term_to_cat)
        wc          = make_wordcloud(freq, color_func, width=wc_width, height=wc_height, font_path=font_path)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(SITE_DISPLAY.get(site, site), fontsize=FONT_SIZES["subplot_title"],
                           fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Oberen Rand in Zoll → Figure-Fraction umrechnen und reservieren, bevor
    # Suptitle platziert wird.
    top_frac = 1 - top_margin / fig_height
    fig.suptitle("Most Frequent SoP Terms per Study Site", fontsize=FONT_SIZES["suptitle"],
                 fontweight="bold", y=1 - 0.2 / fig_height)

    # Legende erst platzieren (unabhängig von der Achsenposition, da
    # fig.legend() figure-weite Koordinaten nutzt) und danach ihre
    # tatsächliche Oberkante messen, um den unteren Achsen-Rand exakt
    # daran anzuschliessen — schliesst die Lücke zwischen Wordclouds
    # und Legende.
    legend      = add_category_legend(fig, cats_present, legend_y=0.01, ncol=ncol)
    bottom_frac = legend_top_fraction(fig, legend) + 0.015
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac, hspace=0.45, wspace=0.1)

    save_figure(fig, output_path)
    print(f"  → {output_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-terms",          required=True, help="CSV von query_hits.py --query top_terms")
    parser.add_argument("--top-terms-per-site", required=True, help="CSV von query_hits.py --query top_terms_per_site")
    parser.add_argument("--output", default="figures/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nWordcloud Generator")

    top_terms          = pd.read_csv(args.top_terms, encoding="utf-8")
    top_terms_per_site = pd.read_csv(args.top_terms_per_site, encoding="utf-8")

    font_path = get_wordcloud_font_path()

    plot_overall(top_terms, output_dir / "10_wordcloud_overall.png", font_path)
    plot_per_site_grid(top_terms_per_site, output_dir / "11_wordcloud_per_site.png", font_path)

    print("\n✓ 2 Wordcloud-Diagramme gespeichert.")


if __name__ == "__main__":
    main()
