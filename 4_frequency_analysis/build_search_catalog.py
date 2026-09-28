"""
Search Catalog Builder — neue Version
=======================================
Liest den Suchwort-Katalog (Excel, neues Format) ein und erstellt
einen strukturierten JSON-Suchkatalog für die Frequenzanalyse.

Format der Input-Excel:
    Spalte 0: Category          (nur erste Zeile pro Kategorie gefüllt)
    Spalte 1: Terms and Phrases (Oberbegriff aus Lexikon B)
    Spalte 2: Search Terms      (Suchvarianten, durch ; getrennt; NaN = ausgeschlossen)
    Spalte 3: Reason for Exclusion (NaN = aktiv)

Einträge mit leerem Suchterm (NaN in Spalte 2) werden als ausgeschlossen
behandelt und in einem separaten Status-Sheet dokumentiert.

Output JSON — Liste von Sucheinträgen, jeder mit:
    label:         Oberbegriff (aus Spalte 1)
    category:      Kategorie
    variants:      Liste der Suchvarianten
    is_phrase:     True wenn mind. eine Variante mehrere Wörter enthält
    original_term: identisch mit label

Verwendung:
    python build_search_catalog.py
        --lexicon search_term_catalogue.xlsx
        --output  search_catalog.json
"""

import argparse
import json
import re
import pandas as pd
from pathlib import Path
from collections import Counter


def build_catalog(filepath: Path) -> tuple[list[dict], list[dict]]:
    """
    Liest das Excel ein und gibt zurück:
        active:   Liste aktiver Sucheinträge
        excluded: Liste ausgeschlossener Einträge mit Begründung
    """
    df = pd.read_excel(filepath, header=0)

    # Spaltennamen normalisieren
    col_cat    = df.columns[0]  # Category
    col_term   = df.columns[1]  # Terms and Phrases
    col_search = df.columns[2]  # Search Terms
    col_reason = df.columns[3]  # Reason for Exclusion

    # Kategorie vorwärts füllen (nur erste Zeile pro Kategorie ist gefüllt)
    df[col_cat] = df[col_cat].ffill()

    active   = []
    excluded = []

    for _, row in df.iterrows():
        category   = str(row[col_cat]).strip()
        term       = row[col_term]
        search_raw = row[col_search]
        reason     = row[col_reason]

        # Leere Einträge überspringen
        if pd.isna(term) or str(term).strip() == "":
            continue

        term = str(term).strip()

        # Ausgeschlossene Einträge (Suchterm ist NaN)
        if pd.isna(search_raw):
            excluded.append({
                "label":    term,
                "category": category,
                "reason":   str(reason).strip() if not pd.isna(reason) else "",
            })
            continue

        search_str = str(search_raw).strip()

        # Leerer String nach Konvertierung = ausgeschlossen
        if not search_str:
            excluded.append({
                "label":    term,
                "category": category,
                "reason":   "empty search term",
            })
            continue

        # Varianten aufteilen — Semikolon als Trennzeichen
        # WICHTIG: kein Substring-Match auf "nan" — nur pd.isna() oben
        variants = [v.strip() for v in search_str.split(";") if v.strip()]

        if not variants:
            excluded.append({
                "label":    term,
                "category": category,
                "reason":   "no valid variants after parsing",
            })
            continue

        is_phrase = any(" " in v for v in variants)

        active.append({
            "label":         term,
            "category":      category,
            "variants":      variants,
            "is_phrase":     is_phrase,
            "original_term": term,
        })

    return active, excluded


def print_summary(active: list[dict], excluded: list[dict]) -> None:
    cats = Counter(e["category"] for e in active)
    phrases = sum(1 for e in active if e["is_phrase"])

    print(f"\n  Aktive Einträge:      {len(active)}")
    print(f"  davon Phrasen:        {phrases}")
    print(f"  davon Einzelwörter:   {len(active) - phrases}")
    print(f"  Ausgeschlossen:       {len(excluded)}")
    print(f"  Kategorien:           {len(cats)}")
    print()
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Search Catalog Builder — neues Format"
    )
    parser.add_argument(
        "--lexicon", required=True,
        help="Pfad zur Suchwort-Katalog Excel-Datei"
    )
    parser.add_argument(
        "--output", default="search_catalog.json",
        help="Ausgabepfad für den JSON-Katalog (Standard: search_catalog.json)"
    )
    parser.add_argument(
        "--excluded-output", default=None,
        help="Ausgabepfad für ausgeschlossene Einträge als JSON (optional)"
    )
    args = parser.parse_args()

    print("\nSearch Catalog Builder")
    print(f"  Lexikon: {args.lexicon}")

    active, excluded = build_catalog(Path(args.lexicon))
    print_summary(active, excluded)

    # Aktiven Katalog speichern
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(active, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Katalog gespeichert: {output_path}")

    # Ausgeschlossene Einträge optional speichern
    if args.excluded_output:
        excl_path = Path(args.excluded_output)
        excl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(excl_path, "w", encoding="utf-8") as f:
            json.dump(excluded, f, ensure_ascii=False, indent=2)
        print(f"✓ Ausgeschlossene Einträge: {excl_path}")

    print("\nNächster Schritt: frequency_analysis.py mit diesem Katalog ausführen.")


if __name__ == "__main__":
    main()
