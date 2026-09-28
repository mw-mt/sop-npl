"""
Bereinigung der Swissdox Datensätze
=====================================
Entfernt HTML-Tags, exakte Duplikate und Near-Duplicates.

Schritte:
    1. HTML-Tags entfernen (BeautifulSoup)
    2. Exakte Duplikate entfernen (content_id und Text-Hash)
    3. Near-Duplicates erkennen und entfernen (MinHash + LSH)
    4. Bereinigte Datei speichern mit Bericht

Installation benötigter Pakete:
    pip install beautifulsoup4 datasketch pandas

Verwendung:
    python clean_datasets.py --input daten/ --output output/
    python clean_datasets.py --input daten/ --output output/ --threshold 0.7
"""

import pandas as pd
import hashlib
import argparse
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH


# ── Konfiguration ─────────────────────────────────────────────────────────────

# Ähnlichkeitsschwellenwert für Near-Duplicates (0.0 – 1.0)
# 0.7 bedeutet: Artikel mit 70%+ Textübereinstimmung gelten als Duplikate
DEFAULT_THRESHOLD = 0.7

# Anzahl MinHash-Permutationen (höher = genauer, aber langsamer)
NUM_PERM = 128

# Minimale Textlänge (Zeichen) — sehr kurze Texte überspringen
MIN_CHARS = 100


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def remove_html(text: str) -> str:
    """Entfernt HTML-Tags und normalisiert Whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # BeautifulSoup für sauberes HTML-Parsing
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text(separator=" ")
    # Mehrfache Leerzeichen und Zeilenumbrüche normalisieren
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def make_hash(text: str) -> str:
    """Erstellt einen MD5-Hash für exakte Duplikat-Erkennung."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize_for_comparison(text: str) -> str:
    """
    Normalisiert Text für Vergleich:
    Kleinbuchstaben, keine Satzzeichen, keine Zahlen.
    Ziel: Kleine Unterschiede (Datum, Seitenzahl) ignorieren.
    """
    text = text.lower()
    text = re.sub(r'[^a-züäöàâéèêëïîôùûüÿœæ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def make_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """Erstellt einen MinHash-Fingerabdruck aus 3-Gramm (Zeichentrigrams)."""
    m = MinHash(num_perm=num_perm)
    # 3-Gramm auf Wortebene
    words = text.split()
    shingles = set()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i+3])
        shingles.add(shingle.encode("utf-8"))
    for s in shingles:
        m.update(s)
    return m


# ── Hauptfunktionen ───────────────────────────────────────────────────────────

def step1_clean_html(df: pd.DataFrame) -> pd.DataFrame:
    """Stufe 1: HTML aus content, head und subhead entfernen."""
    print("  → Stufe 1: HTML-Tags entfernen...")
    for col in ["content", "head", "subhead"]:
        if col in df.columns:
            df[col] = df[col].apply(remove_html)
    # Leere Artikel nach Bereinigung entfernen
    before = len(df)
    df = df[df["content"].str.len() >= MIN_CHARS].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"     {removed} Artikel entfernt (Inhalt zu kurz nach HTML-Bereinigung)")
    return df


def step2_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stufe 2: Exakte Duplikate entfernen.
    Zuerst über content_id (falls vorhanden), dann über Text-Hash.
    """
    print("  → Stufe 2: Exakte Duplikate entfernen...")
    before = len(df)
    removed_details = []

    # 2a: Über content_id
    if "content_id" in df.columns:
        dupes_cid = df[df.duplicated(subset=["content_id"], keep="first")]
        removed_details.append(dupes_cid.assign(duplikat_grund="content_id"))
        df = df.drop_duplicates(subset=["content_id"], keep="first").copy()

    # 2b: Über normalisierten Text-Hash (fängt kleinste Unterschiede wie Zeilenumbrüche)
    df["_text_hash"] = df["content"].apply(
        lambda x: make_hash(normalize_for_comparison(x))
    )
    dupes_hash = df[df.duplicated(subset=["_text_hash"], keep="first")]
    removed_details.append(dupes_hash.assign(duplikat_grund="text_hash"))
    df = df.drop_duplicates(subset=["_text_hash"], keep="first").copy()
    df = df.drop(columns=["_text_hash"])

    removed = before - len(df)
    print(f"     {removed} exakte Duplikate entfernt")

    dupes_df = pd.concat(removed_details, ignore_index=True) if removed_details else pd.DataFrame()
    return df, dupes_df


def step3_near_duplicates(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stufe 3: Near-Duplicates über MinHash + LSH erkennen.
    Behält jeweils den ersten Artikel (ältesten, da nach pubtime sortiert).
    """
    print(f"  → Stufe 3: Near-Duplicates erkennen (Schwellenwert: {threshold:.0%})...")

    # Nach Datum sortieren: bei Duplikaten bleibt der älteste Artikel
    if "pubtime" in df.columns:
        df = df.sort_values("pubtime").reset_index(drop=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    minhashes = {}
    to_remove = set()
    near_dupe_log = []

    for idx, row in df.iterrows():
        text = normalize_for_comparison(row["content"])
        if len(text.split()) < 10:  # zu wenig Text für sinnvollen Vergleich
            continue

        m = make_minhash(text)
        key = str(idx)

        # Ähnliche Artikel im Index suchen
        results = lsh.query(m)
        if results:
            # Nimm ersten Treffer als "Original"
            original_idx = int(results[0])
            to_remove.add(idx)
            near_dupe_log.append({
                "entfernt_id":   row.get("id", idx),
                "entfernt_head": row.get("head", ""),
                "original_id":   df.loc[original_idx, "id"] if original_idx in df.index else "",
                "original_head": df.loc[original_idx, "head"] if original_idx in df.index else "",
                "duplikat_grund": f"near_duplicate (≥{threshold:.0%})"
            })
        else:
            try:
                lsh.insert(key, m)
                minhashes[key] = m
            except ValueError:
                pass  # Schlüssel bereits vorhanden, überspringen

    df_clean = df[~df.index.isin(to_remove)].copy()
    dupes_df = pd.DataFrame(near_dupe_log)

    print(f"     {len(to_remove)} Near-Duplicates entfernt")
    return df_clean, dupes_df


def process_file(filepath: Path, output_dir: Path, threshold: float):
    """Verarbeitet eine einzelne TSV-Datei."""
    standort = filepath.stem
    print(f"\n{'─'*50}")
    print(f"  Standort: {standort}")

    # Einlesen
    df = pd.read_csv(filepath, sep="\t", dtype=str, on_bad_lines="warn")
    if "pubtime" in df.columns:
        # utc=True: die Swissdox-Rohdaten mischen Zeitzonen-Offsets (MEZ/MESZ,
        # +01:00 / +02:00) innerhalb derselben Spalte - ohne utc=True bricht
        # pandas>=2.x hier mit "Mixed timezones detected" ab.
        df["pubtime"] = pd.to_datetime(df["pubtime"], errors="coerce", utc=True)
    total_start = len(df)
    print(f"  Artikel beim Einlesen: {total_start}")

    # Bereinigungsschritte
    df = step1_clean_html(df)
    df, exact_dupes = step2_exact_duplicates(df)
    df, near_dupes  = step3_near_duplicates(df, threshold=threshold)

    total_end = len(df)
    print(f"\n  ✓ Behalten:  {total_end} Artikel")
    print(f"  ✗ Entfernt:  {total_start - total_end} Artikel total")

    # Bereinigte Datei speichern
    out_clean = output_dir / f"{standort}_clean.tsv"
    df.to_csv(out_clean, sep="\t", index=False)
    print(f"  💾 Gespeichert: {out_clean.name}")

    # Duplikat-Log speichern (zur Dokumentation)
    all_dupes = pd.concat([exact_dupes, near_dupes], ignore_index=True)
    if not all_dupes.empty:
        out_dupes = output_dir / f"{standort}_entfernte_duplikate.csv"
        all_dupes.to_csv(out_dupes, index=False, encoding="utf-8-sig")
        print(f"  📋 Duplikat-Log: {out_dupes.name}")


def main():
    parser = argparse.ArgumentParser(description="Swissdox Datensätze bereinigen")
    parser.add_argument("--input",     required=True,               help="Ordner mit .tsv Dateien")
    parser.add_argument("--output",    required=True,               help="Zielordner")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Ähnlichkeitsschwellenwert für Near-Duplicates (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.tsv"))
    if not files:
        print(f"❌ Keine .tsv Dateien in {input_dir} gefunden.")
        return

    print(f"\n🔍 {len(files)} Datei(en) gefunden")
    print(f"📁 Ausgabe nach: {output_dir}")
    print(f"⚙️  Near-Duplicate Schwellenwert: {args.threshold:.0%}")

    for f in sorted(files):
        process_file(f, output_dir, args.threshold)

    print(f"\n{'─'*50}")
    print("✅ Fertig!")


if __name__ == "__main__":
    main()
