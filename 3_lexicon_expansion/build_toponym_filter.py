"""
Toponym-Filter aus Swissnames 3D
==================================
Erstellt eine Liste aller Schweizer Ortsnamen aus den drei Swissnames 3D
CSV-Dateien (Punkt, Linie, Polygon) und ergänzt sie mit manuell definierten
Varianten. Das Ergebnis wird als JSON-Datei gespeichert und im
NLP-Preprocessing-Skript verwendet.

Zweck im Kontext des Projekts:
    Alle Ortsnamen werden aus dem Embedding-Training-Corpus entfernt,
    damit Ortsnamen keine künstlichen semantischen Hubs bilden, die die
    Vektorrepräsentationen von SoP-Begriffen verfälschen würden.
    (Empfehlung des Betreuers, methodisch begründet in Lin et al.)

Verwendung:
    python build_toponym_filter.py
        --punkt  swissnames3d_punkt.csv
        --linie  swissnames3d_linie.csv
        --polygon swissnames3d_polygon.csv
        --output toponym_filter.json
    python build_toponym_filter.py --punkt  swissnames3d\swissNAMES3D_PKT.csv --linie  swissnames3d\swissNAMES3D_LIN.csv --polygon swissnames3d\swissNAMES3D_PLY.csv --output toponym_filter.json


Danach wird toponym_filter.json im Preprocessing-Skript eingebunden.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


# ── Manuelle Varianten ────────────────────────────────────────────────────────
# Hier kannst du Ortsnamen-Varianten eintragen die in Swissnames 3D fehlen.
# Format: "kanonischer Name": ["Variante1", "Variante2", ...]
# Alle Einträge werden lowercase verarbeitet.

CUSTOM_VARIANTS = {
    "robenhuserriet": ["robenhuserriet", "robenhauserriet", "robenhauser ried", "robenhauserried"],
    "ägeriried": ["ägeriried", "aegeriried"],
    "oeschinensee": ["oeschinensee", "oschinensee", "öschinensee"],
    "lägern": ["lägern", "lägernhochwacht"]
}


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def normalise(name: str) -> list[str]:
    """
    Normalisiert einen Ortsnamen für den Vergleich mit Corpus-Tokens.
    
    Da spaCy Tokens lowercased und lemmatisiert verarbeitet, normalisieren
    wir die Swissnames-Einträge auf dieselbe Weise.

    """
    name = name.strip().lower()
    if name:
        return [name]
    return []


def load_swissnames_csv(filepath, name_col):
    print(f"  Lese: {filepath.name}")

    df = None
    # Swisstopo-Dateien nutzen oft Semikolon statt Komma
    for sep in [";", ","]:
        for encoding in ["utf-8-sig", "utf-8", "cp1252"]:
            try:
                df = pd.read_csv(
                    filepath,
                    sep=sep,
                    encoding=encoding,
                    low_memory=False,
                    on_bad_lines="skip"  # fehlerhafte Zeilen überspringen
                )
                if len(df.columns) > 1:
                    print(f"    Separator: '{sep}', Encoding: {encoding}")
                    break
            except Exception:
                continue
        if df is not None and len(df.columns) > 1:
            break

    print(f"    Spalten: {list(df.columns)}")

    if name_col not in df.columns:
        raise ValueError(
            f"\n❌ Spalte '{name_col}' nicht gefunden.\n"
            f"   Verfügbare Spalten: {list(df.columns)}\n"
            f"   Bitte --name-col anpassen.\n"
        )

    toponyms = set()
    raw_names = df[name_col].dropna().unique()
    for name in raw_names:
        for normalised in normalise(str(name)):
            toponyms.add(normalised)

    print(f"    Einträge: {len(raw_names):,}  →  Tokens: {len(toponyms):,}")
    return toponyms


def load_custom_variants() -> set[str]:
    """Lädt die manuell definierten Varianten aus CUSTOM_VARIANTS."""
    toponyms = set()
    for canonical, variants in CUSTOM_VARIANTS.items():
        for variant in variants:
            for normalised in normalise(variant):
                toponyms.add(normalised)
        # Auch den kanonischen Namen selbst aufnehmen
        for normalised in normalise(canonical):
            toponyms.add(normalised)
    return toponyms


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Toponym-Filter aus Swissnames 3D erstellen"
    )
    parser.add_argument(
        "--punkt",   required=True,
        help="Swissnames 3D Punkt-CSV (z.B. swissnames3d_punkt.csv)"
    )
    parser.add_argument(
        "--linie",   required=True,
        help="Swissnames 3D Linien-CSV"
    )
    parser.add_argument(
        "--polygon", required=True,
        help="Swissnames 3D Polygon-CSV"
    )
    parser.add_argument(
        "--output",  default="toponym_filter.json",
        help="Ausgabedatei (Standard: toponym_filter.json)"
    )
    parser.add_argument(
        "--name-col", default="NAME",
        help="Spaltenname mit den Ortsnamen (Standard: NAME)"
    )
    args = parser.parse_args()

    print("\nErstelle Toponym-Filter aus Swissnames 3D...\n")

    all_toponyms = set()

    # Alle drei Swissnames-Dateien laden und zusammenführen
    for filepath in [args.punkt, args.linie, args.polygon]:
        toponyms = load_swissnames_csv(Path(filepath), args.name_col)
        all_toponyms |= toponyms

    # Manuelle Varianten hinzufügen
    custom = load_custom_variants()
    print(f"\n  Manuelle Varianten: {len(custom)}")
    all_toponyms |= custom

    FORCE_EXCLUDE = {"auge", "neu", "himmel", "gut", "paradies", "herz", "kopf", "grün", "höhe", "eis", "geist", "kopf", "gerne", "hausberg", "heimat", "fallen", "sitzen", "ort", "garten", "still", "zberg", "bild"}  # SoP-Begriffe die trotz Toponym-Match behalten werden
    all_toponyms -= FORCE_EXCLUDE  

    print(f"\n  Toponyme total (nach Deduplizierung): {len(all_toponyms):,}")

    # Als JSON speichern (sortiert für bessere Lesbarkeit und Versionierung)
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted(all_toponyms), f, ensure_ascii=False, indent=2)

    print(f"  Gespeichert: {output_path}")
    print("\nDone.")
    print(
        "\nHinweis: toponym_filter.json wird im nlp_preprocessing.py"
        " über --toponym-filter eingebunden."
    )


if __name__ == "__main__":
    main()
