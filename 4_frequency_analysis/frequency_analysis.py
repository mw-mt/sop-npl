"""
Frequency Analysis — Hits Table
=================================
Durchsucht den Frequency-Corpus (JSON pro Standort) mit dem
Suchkatalog und produziert zwei Dateien:

    hits.csv     — eine Zeile pro Treffer (Satz × Begriff)
    articles.csv — alle Artikel mit Satzzahlen (auch ohne Treffer)
                   WICHTIG: Diese Datei ist die korrekte Basis für
                   Normierungen durch sentences_selected.

Verwendung:
    python frequency_analysis.py
        --corpus   frequency_corpus/
        --catalog  search_catalog.json
        --output   hits.csv
"""

import argparse
import json
import csv
import pandas as pd
from pathlib import Path


def load_catalog(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_corpus_file(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def search_sentence(
    lemmas: list[str],
    variants: list[str],
    is_phrase: bool,
) -> tuple[int, str]:
    lemma_str = " ".join(lemmas)
    total_count = 0
    matched_variant = ""
    for variant in variants:
        v = variant.lower().strip()
        count = lemma_str.count(v) if is_phrase else lemmas.count(v)
        if count > 0:
            total_count += count
            if not matched_variant:
                matched_variant = variant
    return total_count, matched_variant


def analyze_corpus(
    corpus_dir: Path,
    catalog: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Durchsucht alle Corpus-Dateien.
    Gibt zurück: (hits, article_metadata)
        hits             — Treffer-Liste
        article_metadata — alle Artikel mit Satzzahlen (auch ohne Treffer)
    """
    hits = []
    article_rows = []
    corpus_files = sorted(corpus_dir.glob("*.json"))

    if not corpus_files:
        print(f"  ❌ Keine JSON-Dateien in {corpus_dir}")
        return hits, article_rows

    for corpus_file in corpus_files:
        site = corpus_file.stem
        print(f"\n  Standort: {site}")
        articles = load_corpus_file(corpus_file)
        site_hits = 0

        for article in articles:
            article_id  = article.get("id", "")
            pubtime     = article.get("pubtime", "")
            medium      = article.get("medium_name", "")
            sents_sel   = article.get("sentences_selected", 0)
            sents_total = article.get("sentences_total", 0)
            sentences   = article.get("sentences", [])

            # Artikel-Metadaten für articles.csv (immer, auch ohne Treffer)
            article_rows.append({
                "site":               site,
                "article_id":         article_id,
                "pubtime":            pubtime,
                "medium_name":        medium,
                "sentences_selected": sents_sel,
                "sentences_total":    sents_total,
            })

            # Suche im Artikel
            for sent_idx, sent in enumerate(sentences):
                lemmas = sent.get("lemmas", [])
                if not lemmas:
                    continue
                for entry in catalog:
                    count, matched = search_sentence(
                        lemmas, entry["variants"], entry["is_phrase"]
                    )
                    if count > 0:
                        hits.append({
                            "site":               site,
                            "article_id":         article_id,
                            "pubtime":            pubtime,
                            "medium_name":        medium,
                            "sentences_selected": sents_sel,
                            "sentences_total":    sents_total,
                            "sentence_idx":       sent_idx,
                            "term_label":         entry["label"],
                            "category":           entry["category"],
                            "matched_variant":    matched,
                            "occurrence_count":   count,
                            "sentence_has_match": 1,
                            "original_term":      entry["original_term"],
                            "is_phrase":          entry["is_phrase"],
                        })
                        site_hits += 1

        print(f"  Treffer: {site_hits:,}")

    return hits, article_rows


def save_hits_csv(hits: list[dict], output_path: Path) -> None:
    if not hits:
        print("  ⚠ Keine Treffer gefunden.")
        return
    fieldnames = [
        "site", "article_id", "pubtime", "medium_name",
        "sentences_selected", "sentences_total", "sentence_idx",
        "term_label", "category", "matched_variant",
        "occurrence_count", "sentence_has_match",
        "original_term", "is_phrase",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hits)
    print(f"\n✓ Hits gespeichert: {output_path}")
    print(f"  Zeilen total: {len(hits):,}")


def save_articles_csv(article_rows: list[dict], output_path: Path) -> None:
    """
    Alle Artikel mit Satzzahlen — auch ohne SoP-Treffer.
    Korrekte Basis für sentences_selected-Normierungen.
    """
    fieldnames = [
        "site", "article_id", "pubtime", "medium_name",
        "sentences_selected", "sentences_total",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(article_rows)

    # Übersicht pro Standort
    df = pd.DataFrame(article_rows)
    summary = df.groupby("site").agg(
        articles        =("article_id",         "count"),
        sents_selected  =("sentences_selected",  "sum"),
        sents_total     =("sentences_total",      "sum"),
    )
    print(f"\n✓ Artikel-Metadaten gespeichert: {output_path}")
    print(f"  Artikel total: {len(article_rows):,}\n")
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Frequency Analysis — Hits Table + Articles Metadata"
    )
    parser.add_argument("--corpus",  required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output",  default="hits.csv")
    args = parser.parse_args()

    corpus_dir    = Path(args.corpus)
    output_path   = Path(args.output)
    articles_path = output_path.parent / "articles.csv"

    print("\nFrequency Analysis")
    print(f"  Corpus:   {corpus_dir}")
    print(f"  Katalog:  {args.catalog}")
    print(f"  Hits:     {output_path}")
    print(f"  Artikel:  {articles_path}\n")

    catalog = load_catalog(Path(args.catalog))
    print(f"  Katalogeinträge: {len(catalog)}")

    hits, article_rows = analyze_corpus(corpus_dir, catalog)

    save_articles_csv(article_rows, articles_path)

    print(f"\n  Treffer total: {len(hits):,}")
    save_hits_csv(hits, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
