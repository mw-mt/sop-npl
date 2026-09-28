"""
Signifikanztest: SoP-Kategorien zwischen Landschaftstypen
=============================================================
Methodisch angelehnt an Wartmann & Purves (2018):
Kruskal-Wallis-Test pro Sense-of-Place-Facette über die Landschaftstypen,
bei signifikantem Ergebnis Post-hoc paarweise Mann-Whitney-U-Tests mit
Bonferroni-korrigiertem Signifikanzniveau (analog Dunn 1961).

WICHTIGE METHODISCHE ABWEICHUNGEN VOM ORIGINAL (Hinweis):
-------------------------------------------------------------
1. Beobachtungseinheit: ARTIKEL statt PERSON.
   W&P zählten pro Teilnehmer:in (n=300, 60/Landschaftstyp) die genannten
   Begriffe pro Facette. Bei Zeitungsartikeln gibt es keine Personen, als
   Analogon wird hier der EINZELNE ARTIKEL verwendet, nicht der einzelne
   Satz. Grund: mehrere SoP-Sätze im selben Artikel sind nicht unabhängig
   (gleicher Autor/Duktus/Thema) Der Artikel ist die kleinste plausibel 
   unabhängige Beobachtungseinheit in diesem Corpus, so wie die Person es 
   im Interview-Design von W&P ist. Anders als bei W&P (60 Personen/Typ, 
   exakt balanciert) ist die Anzahl Artikel pro Standort hier extrem 
   unbalanciert (8 bis >2000).

2. Populationsbasis: NUR Artikel mit mindestens einem SoP-Treffer
   (beliebige Kategorie).
   Artikel ohne jegliche SoP-Sprache (die grosse Mehrheit, s. Punkt 3)
   werden komplett aus der Analyse ausgeschlossen.
   Das ist eine bewusste Abweichung von W&P: dort gingen vermutlich auch
   Personen ohne Erwähnung einer bestimmten Facette mit 0 in den Test ein
   (Interviews eliziitierten SoP direkt, "Nicht-Erwähnung" ist selten und
   für alle Facetten gleich relevant). Bei Zeitungsartikeln enthalten aber
   ~90% der standortbezogenen Sätze GAR KEINE SoP-Sprache (s. Punkt 3) —
   würde man alle Artikel einbeziehen, würde der Test grösstenteils die
   Frage beantworten "wie wahrscheinlich ist SoP-Sprache überhaupt", nicht
   "welche Kategorien dominieren, WENN über SoP gesprochen wird". Durch die
   Beschränkung auf SoP-aktive Artikel testet dieses Skript gezielt die
   zweite, engere Frage: das Kategorie-PROFIL der SoP-Sprache zwischen
   Landschaftstypen, bedingt auf das Vorhandensein von SoP-Sprache.
   Innerhalb dieser Population gilt weiterhin: fehlende Kategorie-Artikel-
   Kombinationen = 0, nicht NaN (ein SoP-aktiver Artikel kann trotzdem 0
   Treffer für Kategorie X haben, wenn seine Treffer nur in Kategorie Y
   liegen).

3. Sparsity: Nur ca. 10% der standortbezogenen Sätze enthalten überhaupt
   SoP-Sprache (siehe visualize_categories.py, Diagramme 08/09). Das
   bedeutet viele Artikel mit 0 Treffern pro Kategorie und entsprechend
   viele Ties in den Kruskal-Wallis-Rangdaten. Kruskal-Wallis ist robust
   gegenüber Ties (Tie-Korrektur ist in scipy.stats.kruskal eingebaut),
   ABER die Teststärke sinkt bei sehr wenigen Artikeln/Treffern (v.a. an
   den Moor-Standorten). Eine separate Sparsity-Diagnostik-CSV macht das
   transparent, OHNE den Test selbst zu verändern.

4. Presence/Absence als separate SENSITIVITÄTSANALYSE (nicht der Haupttest).
   Kommt eine Kategorie in einem Artikel überhaupt vor (ja/nein)? Getestet
   mit Chi-Quadrat-Test der Unabhängigkeit (Kategorie-Präsenz x Landschafts-
   typ). Bei erwarteten Zellhäufigkeiten < 5 wird auf Fisher's Exact Test
   ausgewichen (nur exakt für 2x2-Tabellen möglich); bei grösseren Tabellen
   (>2x2, hier 4x2 oder 5x2) gibt es in scipy keine native r×c-Fisher-
   Exact-Funktion — stattdessen wird eine Monte-Carlo-Simulation der
   Chi-Quadrat-Nullverteilung durchgeführt (Permutation der Presence-Werte
   bei fixierten Landschaftstyp-Gruppengrössen, analog zur Freeman-Halton-
   Erweiterung von Fisher's Exact Test). Diese Variante ist robuster
   gegenüber Ausreisser-Artikeln mit sehr vielen Treffern, ignoriert aber
   die Häufigkeit komplett — daher Ergänzung, kein Ersatz für den
   Kruskal-Wallis-Haupttest.

VARIANTEN (wie in cosine_similarity_sites.py / partial_mantel_test.py):
    1. Alle 10 Standorte (5 Landschaftstypen) — direkt vergleichbar mit W&P
    2. Ohne Moore (Robenhuserriet, Ägeriried; 8 Standorte, 4 Landschafts-
       typen) — robustere Variante, da die Moor-Standorte sehr wenige
       Artikel/Treffer haben und die Ergebnisse sonst leicht verzerren
       können (gleicher Grund wie in partial_mantel_test.py dokumentiert).

VERGLEICH MIT W&P — Referenzwerte aus dem Papertext (Abschnitt 3.3):
    Nur für 3 Facetten berichtet W&P exakte Zahlen im Fliesstext (für die
    übrigen 9 der 12 Facetten liegen nur Figure 2 / qualitative Aussagen
    vor, keine exakten H/p-Werte):
        - Sense of identity and belonging:
              H(4, N=300) = 30.541, p < .001  (signifikant;
              Mountain > River/Moor, Lake > Moor, Hill > River/Moor)
        - Sense of tranquility:
              H(4, N=300) = 4.96, p = .29     (nicht signifikant)
        - Sense of nature / connection to nature:
              H(4, N=300) = 8.049, p = .90    (nicht signifikant)
    Diese Werte werden NICHT neu berechnet, sondern nur als Referenz für
    den Vergleichsblock am Skriptende verwendet (siehe WP_COMPARISON_FACETS
    unten). Die Zuordnung unserer 14 Kategorien zu diesen 3 W&P-Facetten
    ist eine inhaltliche Näherung (Namens-/Konzeptähnlichkeit), KEINE
    validierte 1:1-Entsprechung zu W&Ps ursprünglicher 12-Facetten-
    Taxonomie — bei Bedarf in WP_COMPARISON_FACETS anpassen.

Verwendung:
    python category_landscape_test_SoP+1.py --hits hits.csv --articles articles.csv --output results/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

from visualize_categories_v2 import CATEGORY_ORDER, CATEGORY_EN, SITE_DISPLAY


# ── Konfiguration ─────────────────────────────────────────────────────────────

ALPHA = 0.05

LANDSCAPE_TYPES = {
    "Mountain Lakes":   ["Oeschinensee", "Seealpsee"],
    "Rivers":           ["River Thur", "River Reuss"],
    "Moors":            ["Robenhuserriet", "Ägeriried"],
    "Hills":            ["Lägern", "Pfannenstiel"],
    "Urban Lakes":      ["Ufschötti", "Zürichhorn"],
}

MOOR_SITES = ["Robenhuserriet", "Ägeriried"]

N_PERMUTATIONS_DEFAULT = 10_000
RANDOM_SEED_DEFAULT    = 42


# ── Hilfsfunktionen: Datenaufbereitung ─────────────────────────────────────────

def build_site_to_landscape_type() -> dict:
    """Interner Standort-Key (z.B. 'oeschinensee') -> Landschaftstyp."""
    display_to_type = {}
    for ltype, site_list in LANDSCAPE_TYPES.items():
        for disp in site_list:
            display_to_type[disp] = ltype

    result = {}
    for internal, disp in SITE_DISPLAY.items():
        if disp in display_to_type:
            result[internal] = display_to_type[disp]
    return result


def build_article_category_matrix(
    hits: pd.DataFrame,
    cats_present: list,
    site_to_type: dict,
) -> pd.DataFrame:
    """
    Ein Wert pro (site, article_id, category): Summe occurrence_count.
    Populationsbasis: NUR Artikel mit mindestens einem SoP-Treffer
    (beliebige Kategorie) — s. Docstring, Abweichung 2. Fehlende
    Kategorie-Kombinationen innerhalb dieser Population werden explizit
    mit 0 aufgefüllt (nicht NaN).
    """
    sop_articles = hits[["site", "article_id"]].drop_duplicates()
    sop_articles = sop_articles[sop_articles["site"].isin(site_to_type)]

    cat_df = pd.DataFrame({"category": cats_present})
    full = sop_articles.merge(cat_df, how="cross")

    hits_agg = (
        hits.groupby(["site", "article_id", "category"])["occurrence_count"]
        .sum().reset_index()
    )

    merged = full.merge(hits_agg, how="left", on=["site", "article_id", "category"])
    merged["occurrence_count"] = merged["occurrence_count"].fillna(0)
    merged["presence"]        = (merged["occurrence_count"] > 0).astype(int)
    merged["landscape_type"]  = merged["site"].map(site_to_type)
    return merged


# ── Hilfsfunktionen: Kruskal-Wallis + Effektgrösse ─────────────────────────────

def epsilon_squared(H: float, n: int) -> float:
    """
    Epsilon-Quadrat (ε²) für Kruskal-Wallis:
        ε² = H / ((n²-1)/(n+1))  ==  H / (n-1)   [algebraisch äquivalent]
    n = Gesamtstichprobengrösse (alle Gruppen zusammen).
    """
    if n <= 1:
        return float("nan")
    return H / (n - 1)


def epsilon_squared_label(eps2: float) -> str:
    """
    Verbale Einstufung der Effektstärke für ε², nach den in der Literatur
    gebräuchlichen Schwellenwerten für Rang-Epsilon-Quadrat (Tomczak &
    Tomczak 2014: < .01 vernachlässigbar, < .04 schwach, < .16 moderat,
    < .36 relativ stark, < .64 stark, sonst sehr stark).
    """
    if eps2 != eps2:  # nan
        return "n/a"
    if eps2 < 0.01:
        return "vernachlässigbar"
    if eps2 < 0.04:
        return "schwach"
    if eps2 < 0.16:
        return "moderat"
    if eps2 < 0.36:
        return "relativ stark"
    if eps2 < 0.64:
        return "stark"
    return "sehr stark"


def run_kruskal_for_category(
    matrix: pd.DataFrame,
    category: str,
    types_order: list,
) -> dict:
    sub = matrix[matrix["category"] == category]
    groups   = [sub.loc[sub["landscape_type"] == t, "occurrence_count"].to_numpy() for t in types_order]
    group_ns = {t: len(g) for t, g in zip(types_order, groups)}
    n_total  = sum(group_ns.values())
    k        = len(groups)

    if any(len(g) == 0 for g in groups):
        raise ValueError(
            f"Kategorie '{category}': mindestens ein Landschaftstyp hat 0 Artikel — "
            f"Kruskal-Wallis nicht durchführbar. Gruppengrössen: {group_ns}"
        )

    H, p = stats.kruskal(*groups)
    eps2 = epsilon_squared(H, n_total)

    return {
        "category":         category,
        "df":               k - 1,
        "H":                H,
        "p":                p,
        "epsilon_squared":  eps2,
        "epsilon_label":    epsilon_squared_label(eps2),
        "significant":      p < ALPHA,
        "n_articles_total": n_total,
        "group_ns":         group_ns,
    }


def run_posthoc_mannwhitney(
    matrix: pd.DataFrame,
    category: str,
    types_order: list,
) -> list:
    """
    Paarweise Mann-Whitney-U-Tests zwischen allen Landschaftstyp-Paaren,
    Bonferroni-korrigiertes Signifikanzniveau p^a = ALPHA / Anzahl Paare
    (analog Dunn 1961 im Original — Vergleich des rohen p mit dem
    korrigierten alpha, nicht Multiplikation des p-Werts).
    """
    sub   = matrix[matrix["category"] == category]
    pairs = list(combinations(types_order, 2))
    adjusted_alpha = ALPHA / len(pairs)

    results = []
    for t1, t2 in pairs:
        v1 = sub.loc[sub["landscape_type"] == t1, "occurrence_count"].to_numpy()
        v2 = sub.loc[sub["landscape_type"] == t2, "occurrence_count"].to_numpy()
        u, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        results.append({
            "category":               category,
            "type_1":                 t1,
            "type_2":                 t2,
            "n_1":                    len(v1),
            "n_2":                    len(v2),
            "median_1":               float(np.median(v1)),
            "median_2":               float(np.median(v2)),
            "U":                      u,
            "p":                      p,
            "bonferroni_alpha":       adjusted_alpha,
            "significant_bonferroni": p < adjusted_alpha,
        })
    return results


# ── Konsolen-Ausgabe ────────────────────────────────────────────────────────────

def print_kruskal_result(res: dict) -> None:
    sig = "✓ SIGNIFICANT" if res["significant"] else "✗ not significant"
    en  = CATEGORY_EN.get(res["category"], res["category"])
    print(f"\n  {'─'*60}")
    print(f"  {en}  ({res['category']})")
    print(f"  {'─'*60}")
    groups_str = ", ".join(f"{t}: n={n}" for t, n in res["group_ns"].items())
    print(f"  Gruppen: {groups_str}")
    print(f"  H({res['df']}, N={res['n_articles_total']}) = {res['H']:.3f}, p = {res['p']:.4f}")
    print(f"  ε² = {res['epsilon_squared']:.4f} ({res['epsilon_label']})")
    print(f"  → {sig} at α = .05")


def print_posthoc_results(results: list) -> None:
    if not results:
        return
    alpha_a = results[0]["bonferroni_alpha"]
    print(f"\n  Post-hoc Mann-Whitney-U (Bonferroni-korrigiert, p^a = {alpha_a:.5f}):")
    for r in results:
        sig = "✓" if r["significant_bonferroni"] else "✗"
        print(f"    {sig} {r['type_1']:<15} vs {r['type_2']:<15} "
              f"U={r['U']:>8.1f}, p={r['p']:.4f}  "
              f"(Median {r['median_1']:.1f} vs {r['median_2']:.1f})")



# ── Variante ausführen ───────────────────────────────────────────────────────

def run_variant(
    label: str,
    matrix: pd.DataFrame,
    types_order: list,
    cats_present: list,
    n_permutations: int,
    seed: int,
) -> dict:
    print("\n" + "═"*65)
    print(f"  {label}")
    print(f"  Landschaftstypen: {', '.join(types_order)}")
    print("═"*65)

    kw_results       = []
    posthoc_results  = []
    presence_results = []

    for cat in cats_present:
        res = run_kruskal_for_category(matrix, cat, types_order)
        print_kruskal_result(res)
        kw_results.append(res)

        if res["significant"]:
            ph = run_posthoc_mannwhitney(matrix, cat, types_order)
            print_posthoc_results(ph)
            posthoc_results.extend(ph)

    return {
        "kruskal":   kw_results,
        "posthoc":   posthoc_results,
        "presence":  presence_results,
    }


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kruskal-Wallis-Test: SoP-Kategorien zwischen Landschaftstypen (Artikel-Ebene)"
    )
    parser.add_argument("--hits",     required=True, help="Pfad zur hits.csv")
    parser.add_argument("--articles", required=True, help="Pfad zur articles.csv")
    parser.add_argument("--output",   default="results/", help="Ordner für CSV-Output")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS_DEFAULT,
                        help=f"Anzahl Permutationen für die Chi-Quadrat-Sensitivitätsanalyse (Standard {N_PERMUTATIONS_DEFAULT})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT, help="Random Seed für Reproduzierbarkeit")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSignifikanztest: SoP-Kategorien zwischen Landschaftstypen")
    print("Methodisch angelehnt an Wartmann & Purves (2018), Abschnitt 3.3")
    print("Beobachtungseinheit: ARTIKEL (nicht Satz/Person — s. Docstring)")
    print("Population: NUR Artikel mit >=1 SoP-Treffer, beliebige Kategorie (s. Docstring)\n")

    hits     = pd.read_csv(args.hits,     encoding="utf-8", dtype={"article_id": str})
    articles = pd.read_csv(args.articles, encoding="utf-8", dtype={"article_id": str})

    cats_present = [c for c in CATEGORY_ORDER if c in hits["category"].unique()]
    print(f"  Kategorien: {len(cats_present)}")

    n_corpus_total = articles[["site", "article_id"]].drop_duplicates().shape[0]
    n_sop_active   = hits[["site", "article_id"]].drop_duplicates().shape[0]
    pct_active     = n_sop_active / n_corpus_total * 100 if n_corpus_total else float("nan")
    print(f"  Artikel im Corpus gesamt:                 {n_corpus_total:,}")
    print(f"  Artikel mit ≥1 SoP-Treffer (Testbasis):    {n_sop_active:,} ({pct_active:.1f}%)")

    site_to_type = build_site_to_landscape_type()
    matrix_full  = build_article_category_matrix(hits, cats_present, site_to_type)

    # ── Variante 1: alle 10 Standorte ─────────────────────────────────────────
    types_v1 = list(LANDSCAPE_TYPES.keys())
    res_v1 = run_variant(
        "VARIANTE 1: Alle 10 Standorte (inkl. Moore) — direkt vergleichbar mit W&P",
        matrix_full, types_v1, cats_present, args.n_permutations, args.seed,
    )

    # ── Variante 2: ohne Moore ────────────────────────────────────────────────
    # Grund (wie in partial_mantel_test.py): Moor-Standorte (Robenhuserriet,
    # Ägeriried) haben sehr wenige Artikel/SoP-Treffer und können die
    # Landschaftstyp-Vergleiche verzerren bzw. instabil machen.
    types_v2   = [t for t in LANDSCAPE_TYPES if t != "Moors"]
    matrix_nm  = matrix_full[matrix_full["landscape_type"] != "Moors"]
    res_v2 = run_variant(
        "VARIANTE 2: Ohne Moore (8 Standorte) — robustere Variante",
        matrix_nm, types_v2, cats_present, args.n_permutations, args.seed,
    )

    # ── CSV-Output ────────────────────────────────────────────────────────────
    def flatten_kruskal(results: list, variant_label: str) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "group_ns"}
            row["variant"] = variant_label
            row["group_ns"] = "; ".join(f"{t}={n}" for t, n in r["group_ns"].items())
            rows.append(row)
        return pd.DataFrame(rows)

    kw_df = pd.concat([
        flatten_kruskal(res_v1["kruskal"], "all_10_sites"),
        flatten_kruskal(res_v2["kruskal"], "no_moors_8_sites"),
    ], ignore_index=True)
    kw_out = output_dir / "kruskal_wallis_results.csv"
    kw_df.to_csv(kw_out, index=False, encoding="utf-8")

    def add_variant(rows: list, variant_label: str) -> list:
        for r in rows:
            r["variant"] = variant_label
        return rows

    posthoc_df = pd.DataFrame(
        add_variant(res_v1["posthoc"], "all_10_sites") +
        add_variant(res_v2["posthoc"], "no_moors_8_sites")
    )
    posthoc_out = output_dir / "posthoc_mannwhitney_results.csv"
    posthoc_df.to_csv(posthoc_out, index=False, encoding="utf-8")

    print(f"\n{'═'*65}")
    print("✓ Gespeichert:")
    print(f"  {kw_out.name}        (Kruskal-Wallis, Haupttest, beide Varianten)")
    print(f"  {posthoc_out.name}   (Post-hoc Mann-Whitney-U, nur signifikante Kategorien)")
    print("\nDone.")


if __name__ == "__main__":
    main()
