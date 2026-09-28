"""
Konvertierung der Swissdox TSV Datensätze in lesbare Formate
=============================================================
Dieses Skript liest TSV-Dateien ein und gibt sie als CSV, Excel oder lesbare Textdatei (eine Datei pro Artikel) aus.

Verwendung:
    python convert_datasets.py --input ordner/ --output output/ --format csv
    python convert_datasets.py --input ordner/ --output output/ --format excel
    python convert_datasets.py --input ordner/ --output output/ --format txt
    python convert_datasets.py --input ordner/ --output output/ --format all
"""

import pandas as pd
import argparse
import os
from pathlib import Path


# ── Spalten aus Swissdox ──────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "id", "pubtime", "medium_code", "medium_name", "rubric",
    "regional", "doctype", "doctype_description", "language",
    "char_count", "dateline", "head", "subhead",
    "article_link", "content_id", "content"
]


def load_file(filepath: Path) -> pd.DataFrame:
    """Liest eine TSV-Datei ein."""
    print(f"  Lese: {filepath.name}")
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,          
        on_bad_lines="warn" 
    )

    # Spaltencheck
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  ⚠️  Fehlende Spalten: {missing}")

    # pubtime als Datum
    # utc=True: die Swissdox-Rohdaten mischen Zeitzonen-Offsets (MEZ/MESZ,
    # +01:00 / +02:00) innerhalb derselben Spalte - ohne utc=True bricht
    # pandas>=2.x hier mit "Mixed timezones detected" ab.
    if "pubtime" in df.columns:
        df["pubtime"] = pd.to_datetime(df["pubtime"], errors="coerce", utc=True)

    return df


def to_csv(df: pd.DataFrame, output_path: Path):
    """Speichert als UTF-8 CSV."""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ CSV gespeichert: {output_path.name}")


def to_excel(df: pd.DataFrame, output_path: Path):
    """Speichert als Excel-Datei (ohne den langen content, der Excel überfordert)."""
    df_excel = df.copy()

    # content kürzen für Excel-Lesbarkeit (max 500 Zeichen)
    if "content" in df_excel.columns:
        df_excel["content_preview"] = df_excel["content"].str[:500] + "..."
        df_excel = df_excel.drop(columns=["content"])

    df_excel.to_excel(output_path, index=False, engine="openpyxl")
    print(f"  ✓ Excel gespeichert: {output_path.name}  (content gekürzt auf 500 Zeichen)")


def to_txt(df: pd.DataFrame, output_dir: Path, standort_name: str):
    """
    Speichert jeden Artikel als einzelne .txt Datei.
    Gut geeignet für das manuelle Lesen beim Open Coding.
    """
    txt_dir = output_dir / f"{standort_name}_artikel"
    txt_dir.mkdir(exist_ok=True)

    for _, row in df.iterrows():
        # Dateiname: id_datum_medium
        pubtime_str = str(row.get("pubtime", "unbekannt"))[:10]
        medium = str(row.get("medium_name", "unbekannt")).replace("/", "-")[:30]
        article_id = str(row.get("id", "000"))
        filename = f"{pubtime_str}_{medium}_{article_id}.txt"

        filepath = txt_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"ID:        {row.get('id', '')}\n")
            f.write(f"Datum:     {row.get('pubtime', '')}\n")
            f.write(f"Medium:    {row.get('medium_name', '')} ({row.get('medium_code', '')})\n")
            f.write(f"Rubrik:    {row.get('rubric', '')}\n")
            f.write(f"Regional:  {row.get('regional', '')}\n")
            f.write(f"Sprache:   {row.get('language', '')}\n")
            f.write(f"Link:      {row.get('article_link', '')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"TITEL: {row.get('head', '')}\n")
            if row.get("subhead"):
                f.write(f"UNTERTITEL: {row.get('subhead', '')}\n")
            f.write(f"\n{row.get('content', '')}\n")

    print(f"  ✓ {len(df)} Textdateien gespeichert in: {txt_dir.name}/")


def overview(df: pd.DataFrame, standort_name: str):
    """Gibt eine kurze Übersicht über den Datensatz aus."""
    print(f"\n  📊 Übersicht {standort_name}:")
    print(f"     Artikel total:  {len(df)}")

    if "pubtime" in df.columns:
        print(f"     Zeitraum:       {df['pubtime'].min().date()} – {df['pubtime'].max().date()}")

    if "medium_name" in df.columns:
        top_medien = df["medium_name"].value_counts().head(5)
        print(f"     Top 5 Medien:")
        for medium, count in top_medien.items():
            print(f"       {count:>4}x  {medium}")

    if "language" in df.columns:
        sprachen = df["language"].value_counts()
        print(f"     Sprachen:       {dict(sprachen)}")


def process_file(filepath: Path, output_dir: Path, fmt: str):
    """Verarbeitet eine einzelne Datei."""
    standort_name = filepath.stem  # Dateiname ohne .tsv

    df = load_file(filepath)
    overview(df, standort_name)

    if fmt in ("csv", "all"):
        to_csv(df, output_dir / f"{standort_name}.csv")

    if fmt in ("excel", "all"):
        to_excel(df, output_dir / f"{standort_name}.xlsx")

    if fmt in ("txt", "all"):
        to_txt(df, output_dir, standort_name)


def main():
    parser = argparse.ArgumentParser(description="Swissdox TSV → lesbare Formate")
    parser.add_argument("--input",  required=True, help="Ordner mit den .tsv.xz Dateien")
    parser.add_argument("--output", required=True, help="Zielordner für die Ausgabe")
    parser.add_argument(
        "--format",
        choices=["csv", "excel", "txt", "all"],
        default="csv",
        help="Ausgabeformat (default: csv)"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Alle .tsv Dateien im Eingabeordner finden
    files = list(input_dir.glob("*.tsv"))
    if not files:
        print(f"❌ Keine .tsv Dateien in {input_dir} gefunden.")
        return

    print(f"\n🔍 {len(files)} Datei(en) gefunden in: {input_dir}")
    print(f"📁 Ausgabe nach: {output_dir}")
    print(f"📄 Format: {args.format}\n")

    for f in sorted(files):
        print(f"\n{'─'*50}")
        process_file(f, output_dir, args.format)

    print(f"\n{'─'*50}")
    print("✅ Fertig!")


if __name__ == "__main__":
    main()
