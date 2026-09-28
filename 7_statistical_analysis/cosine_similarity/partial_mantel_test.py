"""
Partial-Mantel-Test für die Cosine-Similarity-Analyse
=======================================================

FORSCHUNGSFRAGE
---------------
Sind Standorte desselben Landschaftstyps in ihrer Sense-of-Place-Sprache
ähnlicher als Standorte unterschiedlicher Typen, kontrolliert für die
Corpus-Grösse (Anzahl SoP-aktiver Artikel pro Standort)?

WARUM EIN MANTEL-TEST?
----------------------
Die 45 Standort-Paare (bei 10 Standorten) sind NICHT unabhängig: jeder
Standort steckt in 9 Paaren gleichzeitig. Der Mantel-Test
löst das über eine Permutation der STANDORTE (nicht der einzelnen
Paar-Werte): die Zeilen und Spalten einer Matrix werden gemeinsam
vertauscht, sodass die Struktur "jeder Standort taucht in mehreren Paaren
auf" in jeder Permutation erhalten bleibt.

EINGABE
-------
    --matrix-presence   site_cosine_similarity_presence.csv
    --matrix-freq       site_cosine_similarity_freq.csv
                        (10 x 10 Cosine-Similarity-Matrizen, erzeugt von
                        cosine_similarity_sites.py; Kopf- und Indexzeile
                        sind die Standort-Anzeigenamen)
    --hits              hits.csv (für die Standort-Grösse)

MATRIZEN
--------
Drei symmetrische Standort x Standort-Matrizen, alle mit DERSELBEN
Reihenfolge der Standorte (alphabetisch sortiert):

    A  Similarity        A[i,j] = Cosine Similarity der Standorte i und j
                         (aus der Matrix-CSV), Diagonale = 1
    B  Typ-Übereinstimmung  B[i,j] = 1, wenn Standort i und j denselben
                         Landschaftstyp haben, sonst 0, Diagonale = 1
    C  Grösse            C[i,j] = min(size_i, size_j), wobei size_i die
                         Anzahl eindeutiger article_id von Standort i in
                         hits.csv ist (= SoP-aktive Artikel), Diagonale = size_i

Verwendet wird ausschliesslich das obere Dreieck OHNE Diagonale
(45 Werte bei 10 Standorten, 28 bei 8) als Vektoren a, b, c. Die Diagonale
wird nie in eine Statistik einbezogen.

STATISTIK
---------
Partial-Mantel-Statistik r_partial:
    1. a wird linear auf c regressiert, Residuum_a gespeichert
    2. b wird linear auf c regressiert, Residuum_b gespeichert
    3. r_partial = Pearson-Korrelation(Residuum_a, Residuum_b)
Sie entspricht der partiellen Korrelation von a und b bei Kontrolle von c.

SIGNIFIKANZ (Permutationstest)
------------------------------
Pro Permutation (Standard 10'000):
    a) zufällige Permutation der Standort-Indizes ziehen
    b) Matrix B mit dieser Permutation NEU ordnen (Zeilen UND Spalten
       gleichzeitig); A und C bleiben unverändert
    c) aus der permutierten B wieder das obere Dreieck extrahieren
    d) r_partial mit dem permutierten b neu berechnen (Regression auf c,
       Korrelation der Residuen)
p-Wert (zweiseitig) = Anteil der permutierten Statistiken, deren
Absolutbetrag mindestens so gross ist wie |r_beobachtet|.
Das Vorgehen (eine der beiden Fokusmatrizen permutieren, die
Kontrollmatrix festhalten) entspricht dem üblichen Partial-Mantel-Ansatz,
z.B. vegan::mantel.partial in R.

VERGLEICHSWERT
--------------
Zusätzlich wird der einfache, NICHT-partialisierte Mantel-Test gerechnet
(Korrelation zwischen a und b direkt, ohne Grössen-Kontrolle). Er nutzt
dieselben Permutationen wie der Partial-Test, sodass die beiden Ergebnisse
direkt vergleichbar sind und sichtbar wird, wie stark die Grössen-Kontrolle
das Ergebnis verändert.

VARIANTEN (je für Presence- und Frequency-Matrix, also 4 Kombinationen)
    1. Alle 10 Standorte
    2. Ohne die zwei Moor-Standorte (Robenhuserriet, Ägeriried; 8 Standorte):
       die Moore haben sehr wenige SoP-Treffer/Artikel, ihre Similarity-
       Vektoren sind entsprechend instabil.

HINWEISE ZUR INTERPRETATION
---------------------------
- Reproduzierbarkeit: Für jede der vier Kombinationen wird der
  Zufallsgenerator frisch mit --seed (Standard 42) initialisiert. Das
  Ergebnis hängt damit nicht von der Ausführungsreihenfolge ab, und die
  Kombinationen mit gleicher Standortzahl verwenden dieselben Permutationen.
- Der p-Wert ist der einfache Anteil laut obiger Definition. Seine
  Auflösung beträgt 1/n_perm; ein p-Wert von 0 bedeutet "kleiner als
  1/n_perm" (die Ausgabe schreibt dann "p < ...").
- Die Aussagekraft ist begrenzt: es gibt nur 10 (bzw. 8) unabhängige
  Standorte und nur 5 (bzw. 4) Paare gleichen Typs. Die Zahl möglicher
  Permutationen ist zwar gross, viele davon sind aber äquivalent (z.B.
  Vertauschen der zwei Standorte eines Typs), die effektive Auflösung der
  Nullverteilung ist daher gröber, als n_perm suggeriert.
- "min(size_i, size_j)" ist eine pragmatische Grössen-Kennzahl pro Paar;
  andere Wahlen (z.B. Summe, geometrisches Mittel) würden die Kontrolle
  leicht anders gewichten.

Verwendung (vom Hauptordner des Repos aus; 6_visualisations muss im
Suchpfad stehen, siehe README — setup.ps1 richtet das automatisch ein):
    python 7_statistical_analysis\\cosine_similarity\\partial_mantel_test.py
        --matrix-presence output\\6_figures\\site_cosine_similarity_presence.csv
        --matrix-freq     output\\6_figures\\site_cosine_similarity_freq.csv
        --hits            output\\4_analysis\\hits.csv
        --output          output\\7_results\\partial_mantel
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from plot_style import SITE_LANDSCAPE_TYPE
from visualize_categories_v2 import SITE_DISPLAY


# ── Konfiguration ─────────────────────────────────────────────────────────────

ALPHA = 0.05
N_PERMUTATIONS_DEFAULT = 10_000
RANDOM_SEED_DEFAULT    = 42

# Anzeigenamen (wie in den Matrix-CSVs), die in Variante 2 ausgeschlossen werden
MOOR_SITES = ["Robenhuserriet", "Ägeriried"]

# Landschaftstyp je Standort, Schlüssel = Anzeigename (wie in den Matrix-CSVs)
SITE_TYPE_BY_DISPLAY = {
    SITE_DISPLAY[key]: ltype for key, ltype in SITE_LANDSCAPE_TYPE.items()
}

# Toleranzen für die Prüfung der Matrix-CSV (Rundung beim Schreiben)
MATRIX_SYMMETRY_TOL = 1e-9
MATRIX_DIAGONAL_TOL = 1e-6

# Toleranz beim Vergleich |r_perm| >= |r_beobachtet|, damit Rundungsrauschen
# im Bereich der Maschinengenauigkeit keine Permutation fälschlich ausschliesst
COMPARISON_TOLERANCE = 1e-12


# ── Einlesen und Prüfen der Eingabedaten ──────────────────────────────────────

def load_similarity_matrix(filepath: Path) -> pd.DataFrame:
    """
    Liest eine site_cosine_similarity_*.csv (10 x 10) und prüft, dass sie eine
    gültige, symmetrische Similarity-Matrix mit bekannten Standorten ist.
    Zeilen und Spalten werden alphabetisch sortiert (gemeinsame Reihenfolge
    für alle Matrizen, Schritt 1).
    """
    sim = pd.read_csv(filepath, encoding="utf-8", index_col=0)

    if set(sim.index) != set(sim.columns):
        raise SystemExit(
            f"\n❌ {filepath.name}: Zeilen- und Spaltenbeschriftung stimmen nicht überein\n"
        )
    sites = sorted(sim.index)
    if len(sites) != len(set(sim.index)):
        raise SystemExit(f"\n❌ {filepath.name}: doppelte Standortnamen\n")

    unknown = [s for s in sites if s not in SITE_TYPE_BY_DISPLAY]
    if unknown:
        raise SystemExit(
            f"\n❌ {filepath.name}: unbekannte Standorte {unknown}\n"
            f"   (erwartet: Anzeigenamen aus visualize_categories_v2.SITE_DISPLAY)\n"
        )
    if len(sites) < 4:
        raise SystemExit(f"\n❌ {filepath.name}: mindestens 4 Standorte nötig, gefunden {len(sites)}\n")

    sim = sim.loc[sites, sites].astype(float)
    values = sim.to_numpy()
    if np.isnan(values).any():
        raise SystemExit(f"\n❌ {filepath.name}: Matrix enthält fehlende Werte\n")
    if not np.allclose(values, values.T, atol=MATRIX_SYMMETRY_TOL):
        raise SystemExit(f"\n❌ {filepath.name}: Matrix ist nicht symmetrisch\n")
    if not np.allclose(np.diag(values), 1.0, atol=MATRIX_DIAGONAL_TOL):
        raise SystemExit(f"\n❌ {filepath.name}: Diagonale ist nicht 1 (keine Similarity-Matrix?)\n")
    if values.min() < -1 - 1e-9 or values.max() > 1 + 1e-9:
        raise SystemExit(f"\n❌ {filepath.name}: Werte ausserhalb [-1, 1]\n")

    return sim


def load_site_sizes(hits_path: Path) -> dict:
    """
    Schritt 1 (Matrix C): Anzahl eindeutiger article_id pro Standort in
    hits.csv, also die Zahl SoP-aktiver Artikel. Ergebnis mit
    Anzeigenamen als Schlüssel (wie in den Matrix-CSVs).
    """
    hits = pd.read_csv(hits_path, encoding="utf-8", usecols=["site", "article_id"],
                       dtype={"article_id": str})
    sizes_internal = hits.groupby("site")["article_id"].nunique()
    return {SITE_DISPLAY.get(site, site): int(n) for site, n in sizes_internal.items()}


# ── Schritt 1: Matrizen A, B, C aufbauen ──────────────────────────────────────

def build_matrices(sim: pd.DataFrame, sizes: dict) -> tuple:
    """
    Baut die drei Matrizen A (Similarity), B (Typ-Übereinstimmung) und
    C (Grösse) für DIESELBE, alphabetisch sortierte Standort-Reihenfolge.

    Returns: (A, B, C, sites)
    """
    sites = sorted(sim.index)

    absent = [s for s in sites if s not in sizes]
    if absent:
        raise SystemExit(
            f"\n❌ Für diese Standorte fehlt die Grösse in hits.csv: {absent}\n"
            f"   (Anzeigenamen der Matrix vs. site-Namen in hits.csv prüfen)\n"
        )

    # Matrix A: Similarity, Diagonale = 1
    A = sim.loc[sites, sites].to_numpy(dtype=float).copy()
    np.fill_diagonal(A, 1.0)

    # Matrix B: 1, wenn gleicher Typ (Diagonale = 1)
    types = np.array([SITE_TYPE_BY_DISPLAY[s] for s in sites])
    B = (types[:, None] == types[None, :]).astype(float)

    # Matrix C: min(size_i, size_j), Diagonale = size_i
    size_vec = np.array([sizes[s] for s in sites], dtype=float)
    C = np.minimum(size_vec[:, None], size_vec[None, :])

    return A, B, C, sites


# ── Schritt 3: Statistik ──────────────────────────────────────────────────────

def residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Residuum der linearen Regression y = b0 + b1 * x + Residuum (OLS,
    in geschlossener Form).
    """
    xc    = x - x.mean()
    denom = float(xc @ xc)
    if denom == 0.0:
        raise ValueError("Kontrollvariable c ist konstant - Regression nicht definiert")
    b1 = float(xc @ (y - y.mean())) / denom
    b0 = y.mean() - b1 * x.mean()
    return y - (b0 + b1 * x)


def pearson_r(u: np.ndarray, v: np.ndarray) -> float:
    """Pearson-Korrelation; nan, falls einer der Vektoren keine Varianz hat."""
    uc, vc = u - u.mean(), v - v.mean()
    denom = np.sqrt((uc @ uc) * (vc @ vc))
    return float("nan") if denom == 0.0 else float((uc @ vc) / denom)


def partial_mantel_r(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Schritt 3: partielle Korrelation von a und b bei Kontrolle von c.
      1. Regression a ~ c  -> Residuum_a
      2. Regression b ~ c  -> Residuum_b
      3. Pearson-r(Residuum_a, Residuum_b)
    """
    return pearson_r(residuals(a, c), residuals(b, c))


# ── Schritt 4: Permutationstest ───────────────────────────────────────────────

def permutation_test(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    n_perm: int,
    seed: int,
) -> dict:
    """
    Beobachtete Statistiken (partial und einfach) plus zweiseitige
    Permutations-p-Werte. Permutiert werden die Standort-Indizes von B
    (Zeilen UND Spalten gemeinsam); A und C bleiben unverändert.
    Beide Statistiken nutzen dieselben Permutationen.
    """
    n  = A.shape[0]
    iu = np.triu_indices(n, k=1)          # Schritt 2: oberes Dreieck ohne Diagonale
    a, b, c = A[iu], B[iu], C[iu]

    # Schritt 3: beobachtete Statistiken
    r_partial_obs = partial_mantel_r(a, b, c)
    r_simple_obs  = pearson_r(a, b)
    if np.isnan(r_partial_obs) or np.isnan(r_simple_obs):
        raise SystemExit(
            "\n❌ Beobachtete Statistik nicht definiert (a oder b ohne Varianz)\n"
        )

    rng = np.random.default_rng(seed)
    r_partial_perm = np.empty(n_perm)
    r_simple_perm  = np.empty(n_perm)

    for k in range(n_perm):
        perm   = rng.permutation(n)                # a) Permutation der Standort-Indizes
        B_perm = B[np.ix_(perm, perm)]             # b) Zeilen UND Spalten gemeinsam neu ordnen
        b_perm = B_perm[iu]                        # c) oberes Dreieck erneut extrahieren
        r_partial_perm[k] = partial_mantel_r(a, b_perm, c)   # d) Schritt 3 wiederholen
        r_simple_perm[k]  = pearson_r(a, b_perm)             #    (Vergleichswert)

    if np.isnan(r_partial_perm).any() or np.isnan(r_simple_perm).any():
        raise SystemExit("\n❌ Permutierte Statistik nicht definiert (nan) - Daten prüfen\n")

    # p-Wert: Anteil der Permutationen mit |r_perm| >= |r_beobachtet| (zweiseitig)
    n_ge_partial = int(np.sum(np.abs(r_partial_perm) >= abs(r_partial_obs) - COMPARISON_TOLERANCE))
    n_ge_simple  = int(np.sum(np.abs(r_simple_perm)  >= abs(r_simple_obs)  - COMPARISON_TOLERANCE))

    return {
        "n_sites":          n,
        "n_pairs":          len(a),
        "r_simple":         r_simple_obs,
        "p_simple":         n_ge_simple / n_perm,
        "n_ge_simple":      n_ge_simple,
        "r_partial":        r_partial_obs,
        "p_partial":        n_ge_partial / n_perm,
        "n_ge_partial":     n_ge_partial,
        "n_permutations":   n_perm,
        "seed":             seed,
    }


# ── Schritt 5: eine Kombination (Matrix-Art x Variante) rechnen ───────────────

def run_combination(
    matrix_label: str,
    variant_label: str,
    sim: pd.DataFrame,
    sizes: dict,
    drop_sites: list,
    n_perm: int,
    seed: int,
) -> dict:
    if drop_sites:
        sim = sim.drop(index=drop_sites, columns=drop_sites)
    A, B, C, sites = build_matrices(sim, sizes)
    res = permutation_test(A, B, C, n_perm, seed)
    res.update({
        "matrix":  matrix_label,
        "variant": variant_label,
        "sites":   sites,
        "n_same_type_pairs": int(B[np.triu_indices(len(sites), k=1)].sum()),
    })
    return res


# ── Ausgabe ───────────────────────────────────────────────────────────────────

def fmt_p(p: float, n_perm: int) -> str:
    """p-Wert als Text; 0 heisst 'kleiner als die Auflösung 1/n_perm'."""
    return f"< {1 / n_perm:.4f}" if p == 0 else f"= {p:.4f}"


def print_result(res: dict) -> None:
    n_perm = res["n_permutations"]
    sig_p  = "✓ SIGNIFICANT" if res["p_partial"] < ALPHA else "✗ not significant"
    sig_s  = "✓ significant"  if res["p_simple"]  < ALPHA else "✗ not significant"
    print(f"\n  {'─'*62}")
    print(f"  {res['matrix']} — {res['variant']}")
    print(f"  {'─'*62}")
    print(f"  n = {res['n_sites']} Standorte, {res['n_pairs']} Paare "
          f"({res['n_same_type_pairs']} davon gleicher Typ)")
    print(f"  Partial Mantel (Kontrolle Grösse):  r_partial = {res['r_partial']:+.4f}, "
          f"p {fmt_p(res['p_partial'], n_perm)}  ({res['n_ge_partial']}/{n_perm})")
    print(f"  Einfacher Mantel (Vergleichswert):  r         = {res['r_simple']:+.4f}, "
          f"p {fmt_p(res['p_simple'], n_perm)}  ({res['n_ge_simple']}/{n_perm})")
    print(f"  → Partial: {sig_p} (α = .05, zweiseitig) | einfach: {sig_s}")


def print_summary(results: list) -> None:
    print("\n" + "═" * 66)
    print("  ZUSAMMENFASSUNG")
    print("═" * 66)
    print(f"  {'Kombination':<36} {'r_partial':>9} {'p':>8}   {'r_einfach':>9} {'p':>8}")
    print(f"  {'─'*36} {'─'*9} {'─'*8}   {'─'*9} {'─'*8}")
    for r in results:
        name = f"{r['matrix']}, {r['variant']}"
        mark = "*" if r["p_partial"] < ALPHA else " "
        print(f"  {name:<36} {r['r_partial']:>+9.4f} {r['p_partial']:>8.4f}{mark}  "
              f"{r['r_simple']:>+9.4f} {r['p_simple']:>8.4f}")
    print("  (* = r_partial signifikant bei p < .05)\n")

    sig = [r for r in results if r["p_partial"] < ALPHA]
    if sig:
        print(f"  r_partial ist signifikant (p < .05) in {len(sig)} von {len(results)} Kombinationen:")
        for r in sig:
            print(f"    - {r['matrix']}, {r['variant']}: r_partial = {r['r_partial']:+.3f}, "
                  f"p {fmt_p(r['p_partial'], r['n_permutations'])}")
    else:
        print(f"  r_partial ist in KEINER der {len(results)} Kombinationen signifikant (p < .05).")

    print("\n  Einfluss der Grössen-Kontrolle (einfacher Mantel -> partial):")
    for r in results:
        s_simple  = r["p_simple"]  < ALPHA
        s_partial = r["p_partial"] < ALPHA
        if s_simple and not s_partial:
            verdict = "Signifikanz geht durch die Kontrolle VERLOREN"
        elif s_partial and not s_simple:
            verdict = "wird durch die Kontrolle erst SIGNIFIKANT"
        elif s_partial and s_simple:
            verdict = "bleibt signifikant"
        else:
            verdict = "bleibt nicht signifikant"
        print(f"    - {r['matrix']}, {r['variant']}: r {r['r_simple']:+.3f} -> {r['r_partial']:+.3f} "
              f"(Δ {r['r_partial'] - r['r_simple']:+.3f}), {verdict}")


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Partial-Mantel-Test: Landschaftstyp vs. Cosine Similarity, "
                    "kontrolliert für Corpus-Grösse"
    )
    parser.add_argument("--matrix-presence", required=True,
                        help="site_cosine_similarity_presence.csv")
    parser.add_argument("--matrix-freq", required=True,
                        help="site_cosine_similarity_freq.csv")
    parser.add_argument("--hits", required=True,
                        help="hits.csv (für die Standort-Grösse)")
    parser.add_argument("--output", default=None,
                        help="Ordner für CSV-Output (optional)")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS_DEFAULT,
                        help=f"Anzahl Permutationen (Standard {N_PERMUTATIONS_DEFAULT})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT,
                        help=f"Random Seed (Standard {RANDOM_SEED_DEFAULT})")
    args = parser.parse_args()

    for label, path in (("--matrix-presence", args.matrix_presence),
                        ("--matrix-freq", args.matrix_freq),
                        ("--hits", args.hits)):
        if not Path(path).is_file():
            raise SystemExit(f"\n❌ {label}: Datei nicht gefunden: {path}\n")
    if args.n_permutations < 1:
        raise SystemExit("\n❌ --n-permutations muss >= 1 sein\n")

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPartial-Mantel-Test: Landschaftstyp vs. Cosine Similarity")
    print("Kontrolle: Corpus-Grösse (SoP-aktive Artikel pro Standort)")
    print(f"  Permutationen: {args.n_permutations:,} | Seed: {args.seed}")

    matrices = {
        "Presence":  load_similarity_matrix(Path(args.matrix_presence)),
        "Frequency": load_similarity_matrix(Path(args.matrix_freq)),
    }
    if set(matrices["Presence"].index) != set(matrices["Frequency"].index):
        raise SystemExit("\n❌ Presence- und Frequency-Matrix enthalten unterschiedliche Standorte\n")
    sizes = load_site_sizes(Path(args.hits))

    # Standort-Übersicht ausgeben (Grundlage von Matrix B und C)
    all_sites = sorted(matrices["Presence"].index)
    print(f"\n  {'Standort':<16} {'Landschaftstyp':<16} {'SoP-aktive Artikel':>18}")
    for s in all_sites:
        print(f"  {s:<16} {SITE_TYPE_BY_DISPLAY[s]:<16} {sizes.get(s, 0):>18,}")

    # Schritt 5: vier Kombinationen (Reihenfolge wie in der Aufgabenstellung)
    variants = [
        ("alle 10 Standorte", []),
        ("ohne Moore (8 Standorte)", MOOR_SITES),
    ]
    results = []
    for matrix_label, sim in matrices.items():
        for variant_label, drop in variants:
            print(f"\n  Rechne {matrix_label}, {variant_label} ...")
            results.append(run_combination(
                matrix_label, variant_label, sim, sizes, drop,
                args.n_permutations, args.seed,
            ))

    # Schritt 6: Ergebnisse ausgeben
    print("\n" + "═" * 66)
    print("  ERGEBNISSE")
    print("═" * 66)
    for res in results:
        print_result(res)
    print_summary(results)

    if output_dir:
        rows = [{
            "matrix":            r["matrix"],
            "variant":           r["variant"],
            "n_sites":           r["n_sites"],
            "n_pairs":           r["n_pairs"],
            "n_same_type_pairs": r["n_same_type_pairs"],
            "r_partial":         round(r["r_partial"], 6),
            "p_partial":         r["p_partial"],
            "significant_partial": r["p_partial"] < ALPHA,
            "r_simple_mantel":   round(r["r_simple"], 6),
            "p_simple_mantel":   r["p_simple"],
            "significant_simple": r["p_simple"] < ALPHA,
            "n_permutations":    r["n_permutations"],
            "seed":              r["seed"],
        } for r in results]
        out_results = output_dir / "partial_mantel_results.csv"
        pd.DataFrame(rows).to_csv(out_results, index=False, encoding="utf-8")

        out_sites = output_dir / "partial_mantel_sites.csv"
        pd.DataFrame([{
            "site": s,
            "landscape_type": SITE_TYPE_BY_DISPLAY[s],
            "sop_active_articles": sizes[s],
        } for s in all_sites]).to_csv(out_sites, index=False, encoding="utf-8")

        print(f"\n✓ Gespeichert:")
        print(f"  {out_results}")
        print(f"  {out_sites}")

    print("\nDone.")


if __name__ == "__main__":
    main()
