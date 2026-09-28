"""
Query Tool für die Hits-Tabelle
=================================
Liest hits.csv UND articles.csv (beide von frequency_analysis.py erstellt).
articles.csv enthält alle Artikel — auch ohne SoP-Treffer — und ist
die korrekte Basis für Normierungen durch sentences_selected.

Für die Abfrage --query catalog_coverage wird zusätzlich der Suchkatalog
(search_catalog.json, von build_search_catalog.py erstellt) benötigt.

Verwendung:
    python query_hits.py --hits hits.csv --articles articles.csv --query sop_sentences_per_site
    python query_hits.py --hits hits.csv --articles articles.csv --query all --output results/

    # catalog_coverage benötigt zusätzlich --catalog:
    python query_hits.py --hits hits.csv --articles articles.csv --catalog search_catalog.json --query catalog_coverage
"""

import argparse
import json
import pandas as pd
from collections import Counter
from pathlib import Path


def load_hits(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, encoding="utf-8")


def load_articles(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, encoding="utf-8")


def load_catalog(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ── Abfragen ──────────────────────────────────────────────────────────────────

def q_sop_sentences_per_site(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Anzahl Sätze mit mind. 1 SoP-Treffer pro Standort."""
    return (
        df[["site", "article_id", "sentence_idx"]]
        .drop_duplicates()
        .groupby("site").size()
        .reset_index(name="sentences_with_sop")
        .sort_values("site")
    )


def q_total_sentences_per_site(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Gesamtzahl ausgewählter Sätze pro Standort (aus articles.csv — korrekt)."""
    return (
        art.groupby("site")["sentences_selected"]
        .sum()
        .reset_index(name="total_selected_sentences")
        .sort_values("site")
    )


def q_sop_rate_per_site(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Anteil SoP-Sätze an allen ausgewählten Sätzen pro Standort."""
    sop   = q_sop_sentences_per_site(df, art).set_index("site")
    total = q_total_sentences_per_site(df, art).set_index("site")
    result = sop.join(total)
    result["sop_rate_%"] = (
        result["sentences_with_sop"] / result["total_selected_sentences"] * 100
    ).round(2)
    return result.reset_index().sort_values("site")


def q_hits_per_category_site(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Rohe Treffer (occurrence_count) pro Kategorie und Standort."""
    return (
        df.groupby(["site", "category"])["occurrence_count"]
        .sum().reset_index(name="total_occurrences")
        .sort_values(["site", "total_occurrences"], ascending=[True, False])
    )


def q_hits_per_term_site(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Treffer pro Begriff und Standort."""
    return (
        df.groupby(["site", "term_label", "category"])["occurrence_count"]
        .sum().reset_index(name="total_occurrences")
        .sort_values(["site", "total_occurrences"], ascending=[True, False])
    )


def q_top_terms(df: pd.DataFrame, art: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top-20 häufigste Begriffe gesamt."""
    return (
        df.groupby(["term_label", "category"])["occurrence_count"]
        .sum().reset_index(name="total_occurrences")
        .sort_values("total_occurrences", ascending=False).head(n)
    )


def q_top_terms_per_site(df: pd.DataFrame, art: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top-N Begriffe pro Standort."""
    result = (
        df.groupby(["site", "term_label", "category"])["occurrence_count"]
        .sum().reset_index(name="total_occurrences")
    )
    return (
        result.sort_values(["site", "total_occurrences"], ascending=[True, False])
        .groupby("site").head(n).reset_index(drop=True)
    )


def q_articles_with_sop(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Wie viele Artikel haben mind. 1 SoP-Treffer pro Standort."""
    total    = art.groupby("site").size().reset_index(name="articles_total")
    with_sop = (
        df[["site", "article_id"]].drop_duplicates()
        .groupby("site").size().reset_index(name="articles_with_sop")
    )
    result = total.merge(with_sop, on="site", how="left").fillna(0)
    result["articles_with_sop"]    = result["articles_with_sop"].astype(int)
    result["articles_without_sop"] = result["articles_total"] - result["articles_with_sop"]
    result["pct_with_sop"]         = (
        result["articles_with_sop"] / result["articles_total"] * 100
    ).round(2)
    return result.sort_values("site")


def q_sop_per_article(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """
    SoP-Treffer pro Artikel — Verteilungsstatistik.
    Zeigt ob wenige Artikel viele SoP haben oder gleichmässige Verteilung.
    Artikel ohne Treffer werden mit 0 einbezogen.
    """
    hits_per_article = (
        df.groupby(["site", "article_id"])["occurrence_count"]
        .sum().reset_index(name="sop_hits")
    )
    # Alle Artikel joinen — auch ohne Treffer erhalten 0
    all_articles = art[["site", "article_id"]].drop_duplicates()
    merged = all_articles.merge(hits_per_article, on=["site", "article_id"], how="left").fillna(0)
    return (
        merged.groupby("site")["sop_hits"]
        .describe().round(2).reset_index()
    )


def q_category_share_per_site(df: pd.DataFrame, art: pd.DataFrame) -> pd.DataFrame:
    """Anteil jeder Kategorie am Total-SoP pro Standort (in %)."""
    hits    = df.groupby(["site", "category"])["occurrence_count"].sum().reset_index(name="hits")
    totals  = hits.groupby("site")["hits"].sum().reset_index(name="total")
    result  = hits.merge(totals, on="site")
    result["share_%"] = (result["hits"] / result["total"] * 100).round(2)
    return (
        result[["site", "category", "hits", "share_%"]]
        .sort_values(["site", "share_%"], ascending=[True, False])
    )


def q_catalog_coverage(df: pd.DataFrame, art: pd.DataFrame, catalog: list[dict]) -> pd.DataFrame:
    """
    Wie viele der Suchbegriffe aus dem Katalog (search_catalog.json) wurden
    auch im Corpus gefunden — mit mindestens einem Treffer, an irgendeinem
    Standort? Aufschlüsselung pro Kategorie plus eine TOTAL-Zeile.

    Ein Begriff zählt als "gefunden", wenn die Kombination (term_label,
    category) mindestens einmal in hits.csv vorkommt — dieselbe Identität
    wie im Katalog selbst (ein term_label kann theoretisch in mehreren
    Kategorien auftauchen).
    """
    catalog_terms = {(e["label"], e["category"]) for e in catalog}
    found_terms   = set(zip(df["term_label"], df["category"]))
    found         = catalog_terms & found_terms

    cat_total = Counter(cat for (_, cat) in catalog_terms)
    cat_found = Counter(cat for (_, cat) in found)

    rows = []
    for cat in sorted(cat_total):
        total = cat_total[cat]
        n_found = cat_found.get(cat, 0)
        rows.append({
            "category":        cat,
            "terms_total":     total,
            "terms_found":     n_found,
            "terms_not_found": total - n_found,
            "coverage_%":      round(n_found / total * 100, 2) if total else 0.0,
        })

    result = pd.DataFrame(rows)
    total_row = pd.DataFrame([{
        "category":        "TOTAL",
        "terms_total":      len(catalog_terms),
        "terms_found":      len(found),
        "terms_not_found":  len(catalog_terms) - len(found),
        "coverage_%":       round(len(found) / len(catalog_terms) * 100, 2) if catalog_terms else 0.0,
    }])
    return pd.concat([result, total_row], ignore_index=True)


def q_catalog_missing_terms(df: pd.DataFrame, art: pd.DataFrame, catalog: list[dict]) -> pd.DataFrame:
    """
    Liste der Suchbegriffe aus dem Katalog, die NICHT im Corpus gefunden
    wurden (Ergänzung zu q_catalog_coverage — dort nur die Zählung, hier
    die konkreten Begriffe).

    Manche (label, category)-Kombinationen kommen mehrfach im Katalog vor
    (z.B. wenn Varianten auf mehrere Zeilen aufgeteilt wurden, wie
    "unschlagbaren Charme" im echten Katalog). Jeder Begriff wird trotzdem
    nur einmal aufgeführt — die Varianten aller seiner Katalog-Zeilen werden
    dafür zusammengeführt, damit keine Variante verloren geht.
    """
    found_terms = set(zip(df["term_label"], df["category"]))

    entries: dict[tuple, dict] = {}
    for e in catalog:
        key = (e["label"], e["category"])
        if key in found_terms:
            continue
        if key not in entries:
            entries[key] = {
                "category":      e["category"],
                "term_label":    e["label"],
                "original_term": e.get("original_term", e["label"]),
                "variants":      list(e.get("variants", [])),
                "is_phrase":     e.get("is_phrase", False),
            }
        else:
            for v in e.get("variants", []):
                if v not in entries[key]["variants"]:
                    entries[key]["variants"].append(v)

    rows = []
    for entry in entries.values():
        entry["variants"] = ", ".join(entry["variants"])
        rows.append(entry)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["category", "term_label"]).reset_index(drop=True)
    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────

QUERIES = {
    "sop_sentences_per_site":   (q_sop_sentences_per_site,   "Sätze mit mind. 1 SoP-Treffer pro Standort"),
    "total_sentences_per_site": (q_total_sentences_per_site, "Gesamtzahl ausgewählter Sätze pro Standort"),
    "sop_rate_per_site":        (q_sop_rate_per_site,        "Anteil SoP-Sätze pro Standort (%)"),
    "hits_per_category_site":   (q_hits_per_category_site,   "Treffer pro Kategorie und Standort"),
    "hits_per_term_site":       (q_hits_per_term_site,        "Treffer pro Begriff und Standort"),
    "top_terms":                (q_top_terms,                 "Top-20 häufigste Begriffe gesamt"),
    "top_terms_per_site":       (q_top_terms_per_site,        "Top-10 Begriffe pro Standort"),
    "articles_with_sop":        (q_articles_with_sop,         "Artikel mit/ohne SoP-Treffer pro Standort"),
    "sop_per_article":          (q_sop_per_article,            "SoP-Treffer pro Artikel inkl. Nullen (Verteilung)"),
    "category_share_per_site":  (q_category_share_per_site,  "Anteil jeder Kategorie pro Standort (%)"),
    "catalog_coverage":         (q_catalog_coverage,         "Katalog-Abdeckung: Suchbegriffe mit/ohne Treffer im Corpus"),
}

# Abfragen, die zusätzlich den Suchkatalog (--catalog) benötigen
CATALOG_QUERIES = {"catalog_coverage"}


def run_query(
    name: str,
    df: pd.DataFrame,
    art: pd.DataFrame,
    output_dir: Path | None,
    catalog: list[dict] | None = None,
) -> None:
    fn, description = QUERIES[name]
    result = fn(df, art, catalog) if name in CATALOG_QUERIES else fn(df, art)
    print(f"\n{'═'*60}")
    print(f"  {description}")
    print(f"{'═'*60}")
    print(result.to_string(index=False))
    if output_dir:
        out = output_dir / f"{name}.csv"
        result.to_csv(out, index=False, encoding="utf-8")
        print(f"\n  → {out.name}")

    # catalog_coverage erzeugt zusätzlich eine Liste aller nicht gefundenen
    # Begriffe (die Zählung allein zeigt nicht, WELCHE Begriffe fehlen)
    if name == "catalog_coverage" and output_dir:
        missing = q_catalog_missing_terms(df, art, catalog)
        out_missing = output_dir / "catalog_coverage_missing_terms.csv"
        missing.to_csv(out_missing, index=False, encoding="utf-8")
        print(f"  → {out_missing.name}  ({len(missing)} nicht gefundene Begriffe)")


def main():
    parser = argparse.ArgumentParser(description="Query Tool für die Hits-Tabelle")
    parser.add_argument("--hits",     required=True, help="Pfad zur hits.csv")
    parser.add_argument("--articles", required=True, help="Pfad zur articles.csv")
    parser.add_argument(
        "--catalog", default=None,
        help="Pfad zur search_catalog.json (nur für --query catalog_coverage/all benötigt)"
    )
    parser.add_argument(
        "--query", required=True,
        choices=list(QUERIES.keys()) + ["all"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\nQuery Tool — Hits-Tabelle")
    df  = load_hits(Path(args.hits))
    art = load_articles(Path(args.articles))
    print(f"  Hits:     {len(df):,} Zeilen")
    print(f"  Artikel:  {len(art):,} Zeilen")

    catalog = load_catalog(Path(args.catalog)) if args.catalog else None
    if catalog is not None:
        print(f"  Katalog:  {len(catalog):,} Einträge")

    if args.query == "catalog_coverage" and catalog is None:
        raise SystemExit(
            "\n❌ --query catalog_coverage benötigt --catalog <search_catalog.json>\n"
        )

    if args.query == "all":
        for name in QUERIES:
            if name in CATALOG_QUERIES and catalog is None:
                print(f"\n  ⚠ Überspringe '{name}' — benötigt --catalog (nicht angegeben)")
                continue
            run_query(name, df, art, output_dir, catalog)
    else:
        run_query(args.query, df, art, output_dir, catalog)

    print("\nDone.")


if __name__ == "__main__":
    main()
