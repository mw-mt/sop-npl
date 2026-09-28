"""
Post-hoc Detail-Tabellen (Appendix)
=====================================
Erstellt aus posthoc_mannwhitney_results.csv, hits.csv und articles.csv
eine LaTeX-Tabelle PRO KATEGORIE mit den vollständigen Testwerten:
n, Anteil Artikel mit mind. 1 Treffer (%), U, p.

Nur die Variante 'all_10_sites' wird ausgegeben (siehe frühere Begründung:
die 'no_moors' Variante liefert für die 6 verbleibenden Paare identische
U/p-Werte, nur der Bonferroni-Schwellenwert ändert sich).

Die Prozentwerte werden direkt aus hits.csv/articles.csv berechnet, auf
derselben Populationsbasis wie die statistischen Tests: nur Artikel mit
mindestens einem SoP-Treffer in IRGENDEINER Kategorie.

Verwendung:
    python build_posthoc_detail_tables.py
        --posthoc posthoc_mannwhitney_results.csv
        --hits hits.csv
        --articles articles.csv
        --output posthoc_detail_tables.tex
"""

import argparse
import pandas as pd
from pathlib import Path


CATEGORY_ALIASES = [
    ("Aesthetic Quality",              ["Ästhetische Qualität"]),
    ("Awe and Wonder",                 ["Überwältigung und Staunen"]),
    ("Experience Intensity",           ["Erlebnisintensität"]),
    ("Value and Significance",         ["Wert und Bedeutsamkeit"]),
    ("Wildness and Pristineness",      ["Wildheit und Ursprünglichkeit"]),
    ("Connection to Nature",           ["Naturverbundenheit"]),
    ("Relaxation and Deceleration",    ["Erholung und Entschleunigung"]),
    ("Tranquility",                    ["Ruhe", "Ruhe und Stille"]),
    ("Personal Emotional Attachment",  ["Persönliche emotionale Bindung"]),
    ("General Positive Emotion",       ["Allgemeine positive Emotion"]),
    ("Social Dimension",               ["Soziale Dimension"]),
    ("Collective Appreciation",        ["Kollektive Wertschätzung"]),
    ("Conflict through Popularity",    ["Konflikt durch Popularität"]),
    ("Threat and Concern",             ["Bedrohung und Besorgnis"]),
]

TYPE_ORDER = ["Mountain Lakes", "Rivers", "Moors", "Hills", "Urban Lakes"]

# Standort-Schlüssel (wie in hits.csv/articles.csv) -> Landschaftstyp
SITE_TO_TYPE = {
    "oeschinensee":            "Mountain Lakes",
    "seealpsee":               "Mountain Lakes",
    "thur_thurauen":           "Rivers",
    "reuss_bremgarten":        "Rivers",
    "robenhuserriet":          "Moors",
    "aegeriried":              "Moors",
    "hochwacht_laegern":       "Hills",
    "hochwacht_pfannenstiel":  "Hills",
    "ufschoetti":              "Urban Lakes",
    "zuerichhorn":             "Urban Lakes",
}


def resolve_category_names(present_in_data: set) -> list[tuple[str, str]]:
    present_stripped = {str(p).strip(): p for p in present_in_data}
    resolved = []
    for english_name, german_variants in CATEGORY_ALIASES:
        match = None
        for variant in german_variants:
            if variant in present_stripped:
                match = present_stripped[variant]
                break
        if match is None:
            print(f"  ⚠ Kategorie nicht gefunden: '{english_name}' "
                  f"(gesucht: {german_variants}) — übersprungen")
            continue
        resolved.append((match, english_name))
    return resolved


def format_p(p: float) -> str:
    if p < .001:
        return r"$<.001$"
    return f"${p:.3f}$".replace("0.", ".")


def order_pair(t1: str, t2: str) -> tuple[str, str]:
    i1, i2 = TYPE_ORDER.index(t1), TYPE_ORDER.index(t2)
    return (t1, t2) if i1 < i2 else (t2, t1)


def build_presence_percentages(hits: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet pro (Kategorie, Landschaftstyp) den Anteil SoP-aktiver
    Artikel die mindestens einen Treffer dieser Kategorie enthalten.

    Populationsbasis identisch zu den statistischen Tests: Artikel mit
    mindestens einem SoP-Treffer in irgendeiner Kategorie.
    """
    hits = hits.copy()
    articles = articles.copy()
    hits["landscape_type"] = hits["site"].map(SITE_TO_TYPE)
    articles["landscape_type"] = articles["site"].map(SITE_TO_TYPE)

    # Populationsbasis: SoP-aktive Artikel (mind. 1 Treffer, egal welche Kategorie)
    sop_active = hits[["site", "article_id", "landscape_type"]].drop_duplicates()

    # Occurrence pro (Artikel, Kategorie) aufsummieren
    occ = (
        hits.groupby(["site", "article_id", "category"])["occurrence_count"]
        .sum()
        .reset_index()
    )

    # Cross-Join: jede SoP-aktive Artikel-Zeile mit jeder Kategorie kombinieren
    categories = hits["category"].unique()
    cross = sop_active.merge(pd.DataFrame({"category": categories}), how="cross")
    cross = cross.merge(
        occ, on=["site", "article_id", "category"], how="left"
    )
    cross["occurrence_count"] = cross["occurrence_count"].fillna(0)
    cross["has_match"] = (cross["occurrence_count"] > 0).astype(int)

    # Pro (Landschaftstyp, Kategorie): n und Anteil mit Treffer.
    # WICHTIG: "n" zählt eindeutige (site, article_id)-Kombinationen, NICHT
    # eindeutige article_id-Werte allein. Ein Artikel kann bei zwei
    # verschiedenen Standorten desselben Landschaftstyps mit derselben
    # article_id auftauchen (z.B. ein Artikel der sowohl Oeschinensee als
    # auch Seealpsee erwähnt und dadurch in beide standortspezifischen
    # Frequency-Corpora aufgenommen wurde). nunique("article_id") würde
    # solche Fälle fälschlicherweise nur einmal statt zweimal zählen, weil
    # es den Standort ignoriert. Da 'cross' bereits genau eine Zeile pro
    # (site, article_id, category)-Kombination enthält, zählt "count"
    # (nicht "nunique") hier korrekt.
    summary = (
        cross.groupby(["landscape_type", "category"])
        .agg(n=("article_id", "count"), n_present=("has_match", "sum"))
        .reset_index()
    )
    summary["pct_present"] = summary["n_present"] / summary["n"] * 100

    return summary


def build_category_table(
    posthoc: pd.DataFrame,
    presence: pd.DataFrame,
    cat_de: str,
    cat_en: str,
    heading_style: str,
) -> str:
    sub = posthoc[
        (posthoc["category"].astype(str).str.strip() == cat_de.strip()) &
        (posthoc["variant"] == "all_10_sites")
    ].copy()

    sub[["type_1", "type_2"]] = sub.apply(
        lambda r: pd.Series(order_pair(r["type_1"], r["type_2"])), axis=1
    )
    sub["sort_key"] = sub.apply(
        lambda r: (TYPE_ORDER.index(r["type_1"]), TYPE_ORDER.index(r["type_2"])),
        axis=1
    )
    sub = sub.sort_values("sort_key")

    # Presence-Prozentwerte für diese Kategorie nachschlagen
    pres_cat = presence[presence["category"].astype(str).str.strip() == cat_de.strip()]
    pct_lookup = dict(zip(pres_cat["landscape_type"], pres_cat["pct_present"]))

    lines = []

    # Überschrift
    if heading_style == "subsubsection":
        lines.append(rf"\subsubsection*{{{cat_en}}}")
    elif heading_style == "paragraph":
        lines.append(rf"\paragraph{{{cat_en}}}")
    lines.append("")

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{|l|l|r|r|r|r|r|l|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Type 1} & \textbf{Type 2} & \textbf{$n_1$} & \textbf{$n_2$} & "
        r"\textbf{\%$_1$} & \textbf{\%$_2$} & \textbf{$U$} & \textbf{$p$} & \textbf{Sig.} \\"
    )
    lines.append(r"\hline")

    for _, row in sub.iterrows():
        sig_mark = r"\checkmark" if row["significant_bonferroni"] else "--"
        p_str = format_p(row["p"])
        pct1 = pct_lookup.get(row["type_1"], float("nan"))
        pct2 = pct_lookup.get(row["type_2"], float("nan"))

        # Konsistenzcheck: stimmt n aus posthoc mit frisch berechnetem n überein?
        n_check1 = presence[
            (presence["category"].astype(str).str.strip() == cat_de.strip()) &
            (presence["landscape_type"] == row["type_1"])
        ]["n"]
        n_check2 = presence[
            (presence["category"].astype(str).str.strip() == cat_de.strip()) &
            (presence["landscape_type"] == row["type_2"])
        ]["n"]
        if not n_check1.empty and int(n_check1.iloc[0]) != int(row["n_1"]):
            print(f"  ⚠ Inkonsistenz [{cat_en}] {row['type_1']}: "
                  f"posthoc n_1={int(row['n_1'])} vs. berechnet n={int(n_check1.iloc[0])} "
                  f"— stammen posthoc- und hits/articles-Dateien vom selben Lauf?")
        if not n_check2.empty and int(n_check2.iloc[0]) != int(row["n_2"]):
            print(f"  ⚠ Inkonsistenz [{cat_en}] {row['type_2']}: "
                  f"posthoc n_2={int(row['n_2'])} vs. berechnet n={int(n_check2.iloc[0])} "
                  f"— stammen posthoc- und hits/articles-Dateien vom selben Lauf?")

        lines.append(
            f"{row['type_1']} & {row['type_2']} & "
            f"{int(row['n_1'])} & {int(row['n_2'])} & "
            f"{pct1:.1f} & {pct2:.1f} & "
            f"{row['U']:.1f} & {p_str} & {sig_mark} \\\\"
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Pairwise Mann-Whitney U comparisons between landscape types "
        rf"for \textit{{{cat_en}}} (all 10 sites, two-sided, "
        rf"Bonferroni-corrected $\alpha' = .005$). "
        rf"$n_1$, $n_2$: number of SoP-active articles (i.e.\ articles containing "
        rf"at least one sense of place match in any category) in each landscape "
        rf"type. \%$_1$, \%$_2$: percentage of these articles containing at least "
        rf"one match of \textit{{{cat_en}}}. $U$: Mann-Whitney U statistic. "
        rf"$p$: two-sided p-value. Sig.: significant after Bonferroni correction "
        rf"(\checkmark) or not (--).}}"
    )
    safe_label = cat_en.lower().replace(" ", "")
    lines.append(rf"\label{{tab:posthoc_{safe_label}}}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc Detail-Tabellen pro Kategorie (LaTeX, Appendix)"
    )
    parser.add_argument("--posthoc", required=True)
    parser.add_argument("--hits", required=True, help="Pfad zur hits.csv")
    parser.add_argument("--articles", required=True, help="Pfad zur articles.csv")
    parser.add_argument("--output", default="posthoc_detail_tables.tex")
    parser.add_argument(
        "--heading-style", default="subsubsection",
        choices=["subsubsection", "paragraph", "none"],
        help="Wie die Kategorie als Titel formatiert wird (Standard: subsubsection*)"
    )
    args = parser.parse_args()

    # article_id als string erzwingen — verhindert dass Pandas lange numerische
    # IDs als float64 interpretiert und dabei durch Rundung unterschiedliche
    # IDs fälschlicherweise zusammenführt (bekanntes Problem, siehe
    # sentence_cosine_similarity.py)
    posthoc = pd.read_csv(args.posthoc)
    hits = pd.read_csv(args.hits, dtype={"article_id": str})
    articles = pd.read_csv(args.articles, dtype={"article_id": str})

    print(f"Post-hoc Zeilen: {len(posthoc)}")
    print(f"Hits Zeilen: {len(hits)}")
    print(f"Artikel Zeilen: {len(articles)}")

    # Prüfen ob alle Standorte im Mapping bekannt sind
    unknown_sites = set(hits["site"].unique()) - set(SITE_TO_TYPE.keys())
    if unknown_sites:
        print(f"  ⚠ Unbekannte Standort-Schlüssel in hits.csv: {unknown_sites}")
        print(f"    Bitte SITE_TO_TYPE im Skript ergänzen.")

    presence = build_presence_percentages(hits, articles)
    print(f"\nPresence-Prozentwerte berechnet für "
          f"{presence['category'].nunique()} Kategorien x "
          f"{presence['landscape_type'].nunique()} Landschaftstypen")

    resolved = resolve_category_names(set(posthoc["category"].unique()))
    print(f"Kategorien in Post-hoc-Ergebnissen: {len(resolved)} von {len(CATEGORY_ALIASES)}")

    all_tables = []
    for cat_de, cat_en in resolved:
        table_tex = build_category_table(
            posthoc, presence, cat_de, cat_en, args.heading_style
        )
        all_tables.append(table_tex)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(all_tables), encoding="utf-8")

    print(f"\n✓ {len(all_tables)} Tabellen geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()
