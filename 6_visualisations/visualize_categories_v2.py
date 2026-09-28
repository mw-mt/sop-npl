"""
Category Visualization — überarbeitet
=======================================
Produziert 7 Kategorie-Diagramme und 2 Verteilungsdiagramme.

Verwendung:
    python visualize_categories.py
        --hits     hits.csv
        --articles articles.csv
        --catalog  search_catalog.json
        --output   figures/
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from collections import Counter

from plot_style import (
    RCPARAMS, FONT_SIZES, SITE_COLORS, CATEGORY_COLORS, HEATMAP_CMAP,
    heatmap_text_color, save_figure,
)

# Einheitliche Basis-Schriftgrössen für alle Skripte (siehe plot_style.py).
# Detail-Elemente (Subplot-Titel, Zellbeschriftungen etc.) nutzen FONT_SIZES.
plt.rcParams.update(RCPARAMS)


# ── Standort-Mapping ──────────────────────────────────────────────────────────
# Reihenfolge der Säulen in den Diagrammen (intern → Anzeigename)

SITE_ORDER = [
    "oeschinensee",
    "seealpsee",
    "thur_thurauen",
    "reuss_bremgarten",
    "robenhuserriet",
    "aegeriried",
    "ufschoetti",
    "zuerichhorn",
    "hochwacht_laegern",
    "hochwacht_pfannenstiel",
]

SITE_DISPLAY = {
    "oeschinensee":          "Oeschinensee",
    "seealpsee":             "Seealpsee",
    "thur_thurauen":         "River Thur",
    "reuss_bremgarten":      "River Reuss",
    "robenhuserriet":        "Robenhuserriet",
    "aegeriried":            "Ägeriried",
    "ufschoetti":            "Ufschötti",
    "zuerichhorn":           "Zürichhorn",
    "hochwacht_laegern":     "Lägern",
    "hochwacht_pfannenstiel":"Pfannenstiel",
}

# ── Kategorie-Reihenfolge und vollständige englische Namen ────────────────────

CATEGORY_ORDER = [
    "Ästhetische Qualität",
    "Überwältigung und Staunen",
    "Erlebnisintensität",
    "Wert und Bedeutsamkeit",
    "Wildheit und Ursprünglichkeit",
    "Naturverbundenheit",
    "Erholung und Entschleunigung",
    "Ruhe und Stille",
    "Persönliche emotionale Bindung",
    "Allgemeine positive Emotion",
    "Soziale Dimension",
    "Kollektive Wertschätzung",
    "Konflikt durch Popularität",
    "Bedrohung und Besorgnis",
]

CATEGORY_EN = {
    "Ästhetische Qualität":           "Aesthetic Quality",
    "Überwältigung und Staunen":      "Awe and Wonder",
    "Erlebnisintensität":             "Experience Intensity",
    "Wert und Bedeutsamkeit":         "Value and Significance",
    "Wildheit und Ursprünglichkeit":  "Wildness and Pristineness",
    "Naturverbundenheit":             "Connection to Nature",
    "Erholung und Entschleunigung":   "Relaxation and Deceleration",
    "Persönliche emotionale Bindung": "Personal Emotional Attachment",
    "Allgemeine positive Emotion":    "General Positive Emotion",
    "Soziale Dimension":              "Social Dimension",
    "Kollektive Wertschätzung":       "Collective Appreciation",
    "Konflikt durch Popularität":     "Conflict through Popularity",
    "Ruhe und Stille":                "Tranquility",
    "Bedrohung und Besorgnis":        "Threat and Concern",
}


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def order_sites(pivot: pd.DataFrame) -> pd.DataFrame:
    """Ordnet Spalten (Standorte) in der gewünschten Reihenfolge."""
    ordered = [s for s in SITE_ORDER if s in pivot.columns]
    return pivot[ordered]


def rename_sites(pivot: pd.DataFrame) -> pd.DataFrame:
    """Benennt Spalten von internen Namen zu Anzeigenamen."""
    return pivot.rename(columns=SITE_DISPLAY)


def rename_categories(pivot: pd.DataFrame) -> pd.DataFrame:
    """Benennt Index (Kategorien) zu englischen Namen."""
    return pivot.rename(index=CATEGORY_EN)


def make_plot(
    pivot: pd.DataFrame,
    title: str,
    ylabel: str,
    output_path: Path,
    float_format: bool = False,
    figsize: tuple = (18, 8),
) -> None:
    """Erstellt ein gruppiertes Säulendiagramm."""
    # Standort-Reihenfolge und -Farben VOR dem Umbenennen bestimmen, damit
    # SITE_COLORS (keyed auf interne Standort-Namen) korrekt zugeordnet wird.
    pivot        = order_sites(pivot)
    site_keys    = pivot.columns.tolist()
    site_colors  = [SITE_COLORS.get(s, "#888888") for s in site_keys]

    pivot = rename_categories(rename_sites(pivot))

    sites   = pivot.columns.tolist()
    cats    = pivot.index.tolist()
    n_sites = len(sites)
    x       = np.arange(len(cats))
    width   = 0.8 / n_sites

    fig, ax = plt.subplots(figsize=figsize)

    for i, (site, color) in enumerate(zip(sites, site_colors)):
        values = pivot[site].fillna(0).tolist()
        offset = (i - n_sites / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=site, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    # Legende ins Diagramm hineingezogen (oben links) statt aussen rechts:
    # die Säulen erreichen nie die volle Höhe, daher wird nichts verdeckt.
    ax.legend(title="Study Site", loc="upper right")
    if not float_format:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  → {output_path.name}")


# ── Verteilungsdiagramme ──────────────────────────────────────────────────────

def compute_sop_distribution(
    df: pd.DataFrame,
    art: pd.DataFrame,
    max_shown: int = 5,
) -> pd.DataFrame:
    """
    Berechnet für jede (site, sentence) die Summe der SoP-Treffer
    (occurrence_count, über alle Kategorien hinweg — nicht getrennt).
    Sätze ohne Treffer erhalten 0.

    Gezählt wird auf Ebene des Lexikon-Labels (term_label): occurrence_count
    ist pro (Satz, term_label) bereits die Summe über alle Schreibvarianten
    dieses Begriffs; hier wird zusätzlich über alle term_labels im Satz
    summiert.

    Returns DataFrame mit Spalten: site, n_sop, count
    """
    # Summe der SoP-Treffer pro Satz (alle Kategorien zusammen)
    sents_with_hits = (
        df.groupby(["site", "article_id", "sentence_idx"])["occurrence_count"]
        .sum()
        .reset_index(name="n_sop")
    )

    # Gesamtzahl Sätze pro Standort (aus articles.csv)
    total_per_site = (
        art.groupby("site")["sentences_selected"].sum()
        .reset_index(name="total")
    )

    # Sätze mit ≥1 Treffer pro Standort
    hits_per_site = (
        sents_with_hits.groupby("site").size()
        .reset_index(name="with_hits")
    )

    # Sätze mit 0 Treffern = total - with_hits
    zero_df = total_per_site.merge(hits_per_site, on="site", how="left")
    zero_df["with_hits"] = zero_df["with_hits"].fillna(0).astype(int)
    zero_df["n_sop"]     = 0
    zero_df["count"]     = zero_df["total"] - zero_df["with_hits"]

    # Sätze mit 1+ Treffer: Häufigkeitsverteilung
    dist = (
        sents_with_hits
        .assign(n_sop=lambda x: x["n_sop"].clip(upper=max_shown)
                .map(lambda v: f"{max_shown}+" if v == max_shown else str(v)))
        .groupby(["site", "n_sop"])
        .size()
        .reset_index(name="count")
    )

    # 0er-Zeilen zusammenführen
    zero_rows = zero_df[["site", "n_sop", "count"]].copy()
    zero_rows["n_sop"] = zero_rows["n_sop"].astype(str)

    result = pd.concat([zero_rows, dist], ignore_index=True)
    return result


def compute_sop_distribution_articles(
    df: pd.DataFrame,
    art: pd.DataFrame,
    max_shown: int = 10,
) -> pd.DataFrame:
    """
    Analog zu compute_sop_distribution, aber auf ARTIKEL-Ebene statt
    Satz-Ebene: für jeden Artikel wird die Summe der SoP-Treffer
    (occurrence_count, über alle Kategorien hinweg — nicht getrennt) über
    alle seine Sätze berechnet. Artikel ohne Treffer erhalten 0.

    Gleiche Metrik wie bei den Satz-Diagrammen (Summe occurrence_count,
    gezählt auf Ebene des Lexikon-Labels), nur eine Ebene höher aggregiert
    — daher auch ein höherer Standard-max_shown (10 statt 5): ein ganzer
    Artikel hat naturgemäss mehr Treffer als ein einzelner Satz.

    Returns DataFrame mit Spalten: site, n_sop, count
    """
    # Summe der SoP-Treffer pro Artikel (alle Kategorien zusammen)
    arts_with_hits = (
        df.groupby(["site", "article_id"])["occurrence_count"]
        .sum()
        .reset_index(name="n_sop")
    )

    # Gesamtzahl Artikel pro Standort (aus articles.csv)
    total_per_site = (
        art[["site", "article_id"]].drop_duplicates()
        .groupby("site").size()
        .reset_index(name="total")
    )

    # Artikel mit ≥1 Treffer pro Standort
    hits_per_site = (
        arts_with_hits.groupby("site").size()
        .reset_index(name="with_hits")
    )

    # Artikel mit 0 Treffern = total - with_hits
    zero_df = total_per_site.merge(hits_per_site, on="site", how="left")
    zero_df["with_hits"] = zero_df["with_hits"].fillna(0).astype(int)
    zero_df["n_sop"]     = 0
    zero_df["count"]     = zero_df["total"] - zero_df["with_hits"]

    # Artikel mit 1+ Treffer: Häufigkeitsverteilung
    dist = (
        arts_with_hits
        .assign(n_sop=lambda x: x["n_sop"].clip(upper=max_shown)
                .map(lambda v: f"{max_shown}+" if v == max_shown else str(v)))
        .groupby(["site", "n_sop"])
        .size()
        .reset_index(name="count")
    )

    # 0er-Zeilen zusammenführen
    zero_rows = zero_df[["site", "n_sop", "count"]].copy()
    zero_rows["n_sop"] = zero_rows["n_sop"].astype(str)

    result = pd.concat([zero_rows, dist], ignore_index=True)
    return result


def plot_distribution_facets(
    dist: pd.DataFrame,
    output_path: Path,
    max_shown: int = 5,
    zoom: bool = False,
    show_labels: bool = False,
    unit: str = "sentence",
    subtitle_full: str = "(sentences from analysis unit T-1/T/T+1; 0 = no SoP match)",
    subtitle_zoom: str = "(sentences from analysis unit T-1/T/T+1; only sentences with ≥1 SoP match shown)",
) -> None:
    """
    Kleine Mehrfach-Histogramme (Subplots) — ein Plot pro Standort.
    X = Summe der SoP-Treffer (occurrence_count, alle Kategorien zusammen),
    Y = Anzahl Sätze/Artikel (je nach unit).

    zoom=True: 0er-Balken (Einheiten ohne SoP-Treffer) wird ausgeblendet,
    damit die Verteilung unter den Standorten mit ≥1 Treffer besser
    vergleichbar ist.
    show_labels=True: exakte Anzahl über jedem Balken anzeigen (0-Balken
    bleiben unbeschriftet, s. Docstring-Diskussion in der Konversation —
    eine "0" auf einem unsichtbaren Balken an der Grundlinie liest sich
    schlecht und trägt keine zusätzliche Information).
    unit: "sentence" oder "article" — steuert Achsen-/Titelbeschriftung.
    """
    unit_plural   = f"{unit}s"
    sites_ordered = [s for s in SITE_ORDER if s in dist["site"].unique()]
    n_sites       = len(sites_ordered)
    ncols         = 2  # bei 10 Standorten -> 5 Zeilen; Seitenverhältnis ~A4-Hochformat
    nrows         = (n_sites + ncols - 1) // ncols
    bins          = [str(i) for i in range(1, max_shown)] + [f"{max_shown}+"]
    if not zoom:
        bins = ["0"] + bins

    # 2 Spalten x 5 Zeilen ergibt bei col_width=5"/row_height=2.8" ein
    # Seitenverhältnis von 10:14 (~0.71) — nahe A4-Hochformat (0.707) — und
    # gleichzeitig deutlich breitere Panels als bei 4 Spalten, damit auch die
    # Bin-Beschriftungen (0,1,2,...,10+) wieder bequem Platz haben.
    col_width  = 5
    row_height = 2.8
    # sharex=False (nicht True): bei sharex=True blendet Matplotlib die
    # x-Tick-Beschriftungen aller Zeilen ausser der untersten automatisch
    # aus. Da jedes Panel dieselben Bins (0,1,2,...) zeigt, bringt das
    # Teilen der Achse hier keinen Vorteil, nur die versteckten Zahlen.
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * col_width, nrows * row_height),
                             sharey=False, sharex=False)
    axes = axes.flatten()

    for i, site in enumerate(sites_ordered):
        ax     = axes[i]
        color  = SITE_COLORS.get(site, "#888888")
        sub    = dist[dist["site"] == site].set_index("n_sop")["count"]

        values = [sub.get(b, 0) for b in bins]
        bars   = ax.bar(bins, values, color=color, alpha=0.9, edgecolor="white")
        if show_labels:
            # 0-Balken bleiben unbeschriftet (nicht lesbar/informativ an der Grundlinie)
            labels = [str(int(v)) if v > 0 else "" for v in values]
            ax.bar_label(bars, labels=labels, fontsize=FONT_SIZES["bar_label"], padding=2)
        ax.set_title(SITE_DISPLAY.get(site, site), fontsize=FONT_SIZES["subplot_title"], fontweight="bold")
        # Kurz gehalten (nicht "... per sentence/article"): die Einheit steht
        # bereits im Obertitel und im y-Achsenlabel — bei den jetzt schmaleren
        # Spalten würde der längere Text am rechten Bildrand abgeschnitten.
        ax.set_xlabel("SoP occurrences")
        ax.set_ylabel(f"Number of {unit_plural}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Leere Subplots ausblenden
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    subtitle = subtitle_zoom if zoom else subtitle_full
    fig.suptitle(
        f"Distribution of SoP Occurrences per {unit.capitalize()} by Study Site\n{subtitle}",
        fontsize=FONT_SIZES["suptitle"], fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save_figure(fig, output_path, dpi=220)
    print(f"  → {output_path.name}")


def plot_distribution_heatmap(
    dist: pd.DataFrame,
    output_path: Path,
    max_shown: int = 5,
    zoom: bool = False,
    unit: str = "sentence",
) -> None:
    """
    Heatmap Standort × Summe der SoP-Treffer pro Satz/Artikel (unit).
    Ersetzt das frühere Liniendiagramm: die Trefferzahl ist eine diskrete
    Größe, eine Linie würde fälschlich Kontinuität zwischen den Kategorien
    suggerieren. Farbe = Anteil (%) der Sätze/Artikel je Standort (Zeile).

    zoom=True: 0er-Bin (Einheiten ohne SoP-Treffer) wird ausgeschlossen,
    sowohl aus der Anzeige als auch aus dem Normierungsnenner je Zeile. Die
    Prozentwerte beziehen sich dann nur auf Einheiten mit ≥1 SoP-Treffer,
    was den Vergleich der Verteilungs-Form zwischen Standorten mit
    unterschiedlicher SoP-Trefferquote erleichtert.
    unit: "sentence" oder "article" — steuert Achsen-/Titelbeschriftung.
    """
    unit_plural   = f"{unit}s"
    sites_ordered = [s for s in SITE_ORDER if s in dist["site"].unique()]
    bins          = [str(i) for i in range(1, max_shown)] + [f"{max_shown}+"]
    if not zoom:
        bins = ["0"] + bins

    matrix = np.zeros((len(sites_ordered), len(bins)))
    for i, site in enumerate(sites_ordered):
        sub    = dist[dist["site"] == site].set_index("n_sop")["count"]
        values = np.array([sub.get(b, 0) for b in bins], dtype=float)
        total  = values.sum()
        matrix[i] = values / total * 100 if total > 0 else values

    site_labels = [SITE_DISPLAY.get(s, s) for s in sites_ordered]

    fig, ax = plt.subplots(figsize=(max(8, len(bins) * 1.4), max(5, len(sites_ordered) * 0.6)))
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto")

    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_yticks(np.arange(len(sites_ordered)))
    ax.set_yticklabels(site_labels)
    ax.set_xlabel(f"SoP occurrences per {unit}")

    # Zellbeschriftung; Textfarbe je nach Zellhelligkeit für Lesbarkeit.
    # HEATMAP_CMAP geht hell->dunkel, daher heatmap_text_color() (nicht die
    # alte viridis-Logik, die die umgekehrte Richtung annahm).
    vmax = matrix.max() if matrix.size else 0
    for i in range(len(sites_ordered)):
        for j in range(len(bins)):
            val   = matrix[i, j]
            color = heatmap_text_color(val, vmax)
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=FONT_SIZES["heatmap_cell"], color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=FONT_SIZES["cbar_ticks"])
    if zoom:
        cbar.set_label(f"Proportion of {unit_plural} with ≥1 SoP match (%)", fontsize=FONT_SIZES["cbar_label"])
        title = (
            f"Distribution of SoP Occurrences per {unit.capitalize()}\n"
            f"(normalised by {unit_plural} with ≥1 SoP match per site, in %)"
        )
    else:
        cbar.set_label(f"Proportion of {unit_plural} (%)", fontsize=FONT_SIZES["cbar_label"])
        title = (
            f"Distribution of SoP Occurrences per {unit.capitalize()}\n"
            f"(normalised by total {unit_plural} per site, in %)"
        )
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  → {output_path.name}")


# ── Mean of Ratios ────────────────────────────────────────────────────────────

def compute_mean_of_ratios(
    df: pd.DataFrame,
    art: pd.DataFrame,
    cats_present: list,
) -> pd.DataFrame:
    """
    Mean of Ratios (statt Ratio of Sums) für Diagramm 04:
    pro Artikel wird Treffer / ausgewählte Sätze (sentences_selected) je
    Kategorie berechnet, danach über alle Artikel des Standorts gemittelt.

    Unterschied zum Ratio of Sums (Summe Treffer / Summe Sätze je Standort):
    jeder Artikel geht mit gleichem Gewicht ein, lange Artikel (viele
    ausgewählte Sätze) dominieren den Wert also nicht mehr.

    Population: NUR SoP-aktive Artikel, also Artikel mit mindestens einem
    Treffer (beliebige Kategorie, d.h. alle Artikel in hits.csv) — nicht alle
    Artikel des Corpus. Nenner der Mittelung ist die Anzahl dieser Artikel
    je Standort; SoP-aktive Artikel ohne Treffer in einer bestimmten
    Kategorie gehen dort mit Ratio 0 ein. Die ausgewählten Sätze
    (sentences_selected) für die Ratio kommen aus articles.csv.
    Artikel-Schlüssel ist (site, article_id), da dieselbe article_id bei
    mehreren Standorten vorkommen kann.

    Returns: DataFrame Kategorie x Standort (Index: cats_present).
    """
    art_u = (
        art[["site", "article_id", "sentences_selected"]]
        .assign(article_id=lambda x: x["article_id"].astype(str))
        .drop_duplicates(["site", "article_id"])
    )

    per_art_cat = (
        df.assign(article_id=df["article_id"].astype(str))
        .groupby(["site", "article_id", "category"])["occurrence_count"]
        .sum().reset_index()
        .merge(art_u, on=["site", "article_id"], how="left")
    )

    n_missing = (
        per_art_cat.loc[per_art_cat["sentences_selected"].isna(), ["site", "article_id"]]
        .drop_duplicates().shape[0]
    )
    if n_missing:
        print(f"  ⚠ {n_missing} Treffer-Artikel nicht in articles.csv gefunden "
              f"— werden im Mean of Ratios ignoriert")
    per_art_cat = per_art_cat[per_art_cat["sentences_selected"] > 0].copy()
    per_art_cat["ratio"] = per_art_cat["occurrence_count"] / per_art_cat["sentences_selected"]

    ratio_sum  = (
        per_art_cat.groupby(["category", "site"])["ratio"]
        .sum().unstack(fill_value=0).reindex(cats_present).fillna(0)
    )
    # Nenner: nur SoP-aktive Artikel (mind. 1 Treffer, beliebige Kategorie)
    n_articles = (
        per_art_cat[["site", "article_id"]].drop_duplicates()
        .groupby("site").size()
    )
    return ratio_sum.div(n_articles.reindex(ratio_sum.columns), axis=1)


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits",     required=True)
    parser.add_argument("--articles", required=True)
    parser.add_argument("--catalog",  required=True)
    parser.add_argument("--output",   default="figures/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCategory Visualization")

    df  = pd.read_csv(args.hits,     encoding="utf-8")
    art = pd.read_csv(args.articles, encoding="utf-8")

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)
    cat_sizes = Counter(e["category"] for e in catalog)

    cats_present = [c for c in CATEGORY_ORDER if c in df["category"].unique()]

    # Normierungsnenner aus articles.csv
    sents_with_sop     = (
        df[["site", "article_id", "sentence_idx"]].drop_duplicates()
        .groupby("site").size()
    )
    # Artikel mit mindestens 1 SoP-Treffer, unabhängig von der Kategorie
    # (jeder Artikel in hits.csv hat per Definition ≥1 Treffer)
    articles_with_sop_per_site = (
        df[["site", "article_id"]].drop_duplicates()
        .groupby("site").size()
    )

    # Pivot-Tabellen
    pivot_raw = (
        df.groupby(["category", "site"])["occurrence_count"]
        .sum().unstack(fill_value=0).reindex(cats_present)
    )
    pivot_sents = (
        df.groupby(["category", "site"])["sentence_has_match"]
        .sum().unstack(fill_value=0).reindex(cats_present)
    )
    pivot_by_terms = pivot_raw.copy().astype(float)
    for cat in pivot_by_terms.index:
        pivot_by_terms.loc[cat] /= cat_sizes.get(cat, 1)

    # Diagramm 04: Mean of Ratios (Mittel der Artikel-Ratios Treffer/Sätze)
    pivot_mean_of_ratios = compute_mean_of_ratios(df, art, cats_present)

    pivot_by_sop_sents = pivot_sents.copy().astype(float)
    for site in pivot_by_sop_sents.columns:
        pivot_by_sop_sents[site] /= sents_with_sop.get(site, 1)

    pivot_by_sop_articles = pivot_raw.copy().astype(float)
    for site in pivot_by_sop_articles.columns:
        pivot_by_sop_articles[site] /= articles_with_sop_per_site.get(site, 1)

    total_hits_per_site = pivot_raw.sum(axis=0)
    pivot_share = pivot_raw.copy().astype(float)
    for site in pivot_share.columns:
        pivot_share[site] = pivot_share[site] / total_hits_per_site.get(site, 1) * 100

    # ── Säulendiagramme ───────────────────────────────────────────────────────
    print("\nKategorie-Diagramme:")
    make_plot(pivot_raw,
        "Number of SoP Term Occurrences per Category and Study Site",
        "Total occurrences",
        output_dir / "01_raw_occurrences.png")

    make_plot(pivot_sents,
        "Number of Sentences with SoP Match per Category and Study Site",
        "Number of sentences",
        output_dir / "02_sentence_level.png")

    make_plot(pivot_by_terms,
        "SoP Occurrences per Category and Study Site\n"
        "(normalised by number of search terms per category)",
        "Occurrences per search term",
        output_dir / "03_normalised_by_terms.png",
        float_format=True)

    make_plot(pivot_mean_of_ratios,
        "SoP Occurrences per Category and Study Site\n"
        "(mean of per-article ratios: occurrences / selected sentences,\n"
        "articles with ≥1 SoP match only)",
        "Mean occurrences per selected sentence",
        output_dir / "04_normalised_by_selected_sentences.png",
        float_format=True)

    make_plot(pivot_by_sop_articles,
        "SoP Occurrences per Category and Study Site\n"
        "(normalised by number of articles with ≥1 SoP match, any category)",
        "Occurrences per SoP-active article",
        output_dir / "04b_normalised_by_sop_articles.png",
        float_format=True)

    make_plot(pivot_by_sop_sents,
        "SoP Sentences per Category and Study Site\n"
        "(normalised by sentences with at least one SoP match)",
        "Proportion of SoP sentences",
        output_dir / "05_normalised_by_sop_sentences.png",
        float_format=True)

    make_plot(pivot_share,
        "Share of SoP Occurrences per Category and Study Site\n"
        "(normalised by total occurrences per site, in %)",
        "Share of site's SoP occurrences (%)",
        output_dir / "06_share_of_site_hits.png",
        float_format=True)

    # Gestapeltes Balkendiagramm
    pivot_share_cat = rename_categories(rename_sites(order_sites(pivot_share)))
    sites_disp      = pivot_share_cat.columns.tolist()

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom  = np.zeros(len(sites_disp))
    for cat_de in cats_present:
        cat_en = CATEGORY_EN.get(cat_de, cat_de)
        values = [pivot_share_cat.loc[cat_en, s]
                  if cat_en in pivot_share_cat.index and s in pivot_share_cat.columns
                  else 0 for s in sites_disp]
        ax.bar(sites_disp, values, bottom=bottom,
               label=cat_en, color=CATEGORY_COLORS.get(cat_de, "#888888"), alpha=0.95)
        bottom += np.array(values)

    ax.set_ylabel("Share of SoP occurrences (%)")
    ax.set_title("SoP Category Profile per Study Site\n"
                 "(share of total site occurrences per category)",
                 fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(sites_disp)))
    ax.set_xticklabels(sites_disp, rotation=35, ha="right")
    # Legende bleibt aussen (nicht wie beim gruppierten Balkendiagramm nach
    # innen gezogen): die gestapelten Balken füllen hier die volle Höhe
    # (0-100%), eine interne Legende würde also Daten verdecken.
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, output_dir / "07_stacked_site_profiles.png")
    print("  → 07_stacked_site_profiles.png")

    # ── Verteilungsdiagramme (Satz-Ebene) ─────────────────────────────────────
    print("\nVerteilungsdiagramme (Satz-Ebene):")
    dist = compute_sop_distribution(df, art)

    plot_distribution_facets(dist, output_dir / "08_distribution_facets.png",
                              show_labels=True)
    plot_distribution_facets(dist, output_dir / "08b_distribution_facets_zoom.png",
                              zoom=True, show_labels=True)
    plot_distribution_heatmap(dist, output_dir / "09_distribution_heatmap.png")
    plot_distribution_heatmap(dist, output_dir / "09b_distribution_heatmap_zoom.png",
                               zoom=True)

    # ── Verteilungsdiagramme (Artikel-Ebene) ──────────────────────────────────
    # Analog zu 08/08b/09/09b, aber ein Artikel statt ein Satz ist die Einheit:
    # X = Summe der SoP-Treffer pro Artikel (über alle seine Sätze hinweg).
    print("\nVerteilungsdiagramme (Artikel-Ebene):")
    dist_articles = compute_sop_distribution_articles(df, art)

    plot_distribution_facets(
        dist_articles, output_dir / "08c_distribution_facets_articles.png",
        max_shown=10, unit="article", show_labels=True,
        subtitle_full="(0 = no SoP match)",
    )
    plot_distribution_facets(
        dist_articles, output_dir / "08d_distribution_facets_articles_zoom.png",
        max_shown=10, zoom=True, show_labels=True, unit="article",
        subtitle_zoom="(only articles with ≥1 SoP match shown)",
    )
    plot_distribution_heatmap(
        dist_articles, output_dir / "09c_distribution_heatmap_articles.png",
        max_shown=10, unit="article",
    )
    plot_distribution_heatmap(
        dist_articles, output_dir / "09d_distribution_heatmap_articles_zoom.png",
        max_shown=10, zoom=True, unit="article",
    )

    print("\n✓ 16 Diagramme gespeichert.")
    print("  01  raw occurrences")
    print("  02  sentence level")
    print("  03  normalised by search terms")
    print("  04  mean of per-article ratios (hits / selected sentences), nur SoP-aktive Artikel")
    print("  04b hits normalised by SoP-active articles (any category)")
    print("  05  normalised by SoP sentences")
    print("  06  share of site hits (%)")
    print("  07  stacked site profiles")
    print("  08  distribution facets, Satz-Ebene (subplots pro Standort)")
    print("  08b distribution facets, Satz-Ebene, gezoomt auf ≥1 Treffer, mit Beschriftung")
    print("  08c distribution facets, Artikel-Ebene (subplots pro Standort)")
    print("  08d distribution facets, Artikel-Ebene, gezoomt auf ≥1 Treffer, mit Beschriftung")
    print("  09  distribution heatmap, Satz-Ebene (normiert, alle Standorte)")
    print("  09b distribution heatmap, Satz-Ebene, gezoomt auf ≥1 Treffer")
    print("  09c distribution heatmap, Artikel-Ebene (normiert, alle Standorte)")
    print("  09d distribution heatmap, Artikel-Ebene, gezoomt auf ≥1 Treffer")


if __name__ == "__main__":
    main()
