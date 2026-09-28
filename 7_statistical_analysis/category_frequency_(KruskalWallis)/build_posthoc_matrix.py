"""
Post-hoc Signifikanz-Matrix
=============================
Erstellt aus posthoc_mannwhitney_results.csv eine kompakte Matrix:
Zeilen = Kategorien, Spalten = Landschaftstyp-Paare, Zellen = Signifikanz-Sterne.

Output: LaTeX-Tabellencode (direkt einfügbar) + CSV zur Kontrolle.

Verwendung:
    python build_posthoc_matrix.py --posthoc posthoc_mannwhitney_results.csv --output posthoc_matrix
"""

import argparse
import pandas as pd
from pathlib import Path


# Kategorie-Reihenfolge in der finalen (englischen) Anzeige-Reihenfolge.
# Jeder Eintrag listet ALLE bekannten deutschen Schreibweisen dieser Kategorie
# (z.B. "Ruhe" vs. "Ruhe und Stille") — das Skript matcht robust gegen
# jede davon, egal welche exakt in der CSV steht.

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


def resolve_category_names(present_in_data: set) -> list[tuple[str, str]]:
    """
    Gleicht CATEGORY_ALIASES gegen die tatsächlich in der CSV vorkommenden
    Kategorienamen ab (mit Leerzeichen-Trimming). Gibt eine Liste von
    (deutscher_name_wie_in_csv, englischer_anzeigename) in der gewünschten
    finalen Reihenfolge zurück. Nicht gefundene Kategorien werden übersprungen
    und als Warnung ausgegeben, statt sie stillschweigend fallen zu lassen.
    """
    present_stripped = {str(p).strip(): p for p in present_in_data}
    resolved = []
    for english_name, german_variants in CATEGORY_ALIASES:
        match = None
        for variant in german_variants:
            if variant in present_stripped:
                match = present_stripped[variant]
                break
        if match is None:
            print(f"  ⚠ Kategorie nicht gefunden in CSV: '{english_name}' "
                  f"(gesucht: {german_variants}) — wird übersprungen")
            continue
        resolved.append((match, english_name))
    return resolved

# Landschaftstyp-Reihenfolge und Kürzel für Spaltenköpfe
TYPE_ORDER = ["Mountain Lakes", "Rivers", "Moors", "Hills", "Urban Lakes"]
TYPE_ABBR = {
    "Mountain Lakes":   "ML",
    "Rivers":           "Ri",
    "Moors":            "Mo",
    "Hills":            "Hi",
    "Urban Lakes":      "Ur",
}


def build_pair_columns() -> list[tuple[str, str, str]]:
    """
    Erzeugt alle 10 Paare in fester Reihenfolge.
    Gibt Liste von (type_1, type_2, spaltenkürzel) zurück.
    """
    pairs = []
    for i, t1 in enumerate(TYPE_ORDER):
        for t2 in TYPE_ORDER[i+1:]:
            col_label = f"{TYPE_ABBR[t1]}-{TYPE_ABBR[t2]}"
            pairs.append((t1, t2, col_label))
    return pairs


def significance_stars(p: float) -> str:
    """Wandelt p-Wert in Sterne um."""
    if pd.isna(p):
        return "?"
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    return "--"


def find_p_column(df: pd.DataFrame) -> str:
    """Findet die p-Wert-Spalte unabhängig vom genauen Namen."""
    candidates = ["p_value", "p", "p_val", "pvalue"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Keine p-Wert-Spalte gefunden. Vorhandene Spalten: {df.columns.tolist()}"
    )


def build_matrix(posthoc: pd.DataFrame, alpha_bonferroni: float) -> pd.DataFrame:
    """
    Erstellt die Kategorie x Paar Matrix mit Signifikanz-Symbolen.
    Nutzt den Bonferroni-korrigierten p-Wert-Vergleich (p < alpha_bonferroni)
    für die Sternvergabe, gestaffelt nach unkorrigiertem p für die Stärke.
    """
    pairs = build_pair_columns()
    p_col = find_p_column(posthoc)

    resolved = resolve_category_names(set(posthoc["category"].unique()))
    if not resolved:
        raise ValueError(
            "Keine Kategorien aus CATEGORY_ALIASES konnten der CSV zugeordnet "
            f"werden. Kategorien in der CSV: {sorted(posthoc['category'].unique())}"
        )

    matrix = pd.DataFrame(
        index=[english for _, english in resolved],
        columns=[p[2] for p in pairs]
    )

    for cat_de, cat_en in resolved:
        sub = posthoc[posthoc["category"].astype(str).str.strip() == cat_de.strip()]
        for t1, t2, col in pairs:
            # Paar kann in beiden Richtungen gespeichert sein (type_1/type_2 vertauscht)
            row = sub[
                ((sub["type_1"] == t1) & (sub["type_2"] == t2)) |
                ((sub["type_1"] == t2) & (sub["type_2"] == t1))
            ]
            if row.empty:
                matrix.loc[cat_en, col] = "n/a"
                continue

            p_val = row.iloc[0][p_col]

            # Nur signifikant wenn unter Bonferroni-Schwelle
            if pd.isna(p_val):
                matrix.loc[cat_en, col] = "?"
            elif p_val < alpha_bonferroni:
                matrix.loc[cat_en, col] = significance_stars(p_val)
            else:
                matrix.loc[cat_en, col] = "--"

    return matrix


def matrix_to_latex(matrix: pd.DataFrame) -> str:
    """Erzeugt LaTeX-Tabellencode aus der Matrix."""
    cols = matrix.columns.tolist()
    col_spec = "|l|" + "c|" * len(cols)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    header = ["\\textbf{Category}"] + [f"\\textbf{{{c}}}" for c in cols]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")

    for cat_en in matrix.index:
        row_vals = [str(v) for v in matrix.loc[cat_en]]
        lines.append(" & ".join([cat_en] + row_vals) + r" \\")
        lines.append(r"\hline")

    lines.append(
        r"\multicolumn{" + str(len(cols)+1) +
        r"}{l}{\footnotesize \textit{ML = Mountain Lakes, Ri = Rivers, Mo = Moors, "
        r"Hi = Hills, Ur = Urban Lakes.}} \\"
    )
    lines.append(
        r"\multicolumn{" + str(len(cols)+1) +
        r"}{l}{\footnotesize \textit{* $p<.05$, ** $p<.01$, *** $p<.001$ "
        r"(Bonferroni-corrected); -- not significant; n/a not tested "
        r"(omnibus test not significant).}} \\"
    )
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Significance of pairwise Mann-Whitney U comparisons "
        r"between landscape types, per SoP category (all 10 sites).}"
    )
    lines.append(r"\label{tab:posthocmatrix}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc Signifikanz-Matrix aus Mann-Whitney Ergebnissen"
    )
    parser.add_argument("--posthoc", required=True, help="Pfad zur posthoc_mannwhitney_results.csv")
    parser.add_argument("--output", default="posthoc_matrix", help="Basisname für Output-Dateien")
    parser.add_argument(
        "--n-pairs", type=int, default=10,
        help="Anzahl Paare für Bonferroni-Korrektur (Standard: 10, bei 5 Gruppen)"
    )
    args = parser.parse_args()

    posthoc = pd.read_csv(args.posthoc)
    print(f"Post-hoc Zeilen eingelesen: {len(posthoc)}")
    print(f"Spalten: {posthoc.columns.tolist()}")

    alpha_bonf = 0.05 / args.n_pairs
    print(f"Bonferroni-korrigiertes Alpha: {alpha_bonf:.4f}")

    print(f"\nKategorien in der CSV (roh): {sorted(posthoc['category'].unique())}\n")

    matrix = build_matrix(posthoc, alpha_bonf)
    print(f"\nKategorien in der finalen Tabelle: {len(matrix.index)} "
          f"(erwartet: {len(CATEGORY_ALIASES)})")

    # CSV speichern
    csv_path = Path(f"{args.output}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(csv_path, encoding="utf-8")
    print(f"\n✓ Matrix als CSV: {csv_path}")

    # LaTeX speichern
    latex_code = matrix_to_latex(matrix)
    tex_path = Path(f"{args.output}.tex")
    tex_path.write_text(latex_code, encoding="utf-8")
    print(f"✓ LaTeX-Code: {tex_path}")

    print("\nVorschau:")
    print(matrix.to_string())


if __name__ == "__main__":
    main()
