"""
AntConc Corpus-Vorbereitung
============================
Erstellt zwei Versionen des Corpus für die Verwendung in AntConc:

    1. LEMMATISIERT:
       Für Phrasensuche und robuste Frequenzanalyse.
       Pro Standort eine .txt Datei, ein Satz pro Zeile.

    2. ORIGINALTEXT:
       Für KWIC und qualitative Kontextkontrolle.
       Pro Standort eine .txt Datei, ein Satz pro Zeile.

Ausgabe-Struktur:
    antconc/
        lemmatized/
            oeschinensee.txt
            seealpsee.txt
            ...
        original/
            oeschinensee.txt
            seealpsee.txt
            ...

Verwendung in AntConc:
    File → Open File(s) → alle gewünschten .txt Dateien auswählen

Verwendung:
    python prepare_antconc.py
        --input   cleaned/
        --output  antconc/
        --toponym-filter toponym_filter.json
"""

import argparse
import json
import re
import pandas as pd
import spacy
from pathlib import Path


# ── Konfiguration ─────────────────────────────────────────────────────────────

MIN_TOKENS_PER_SENTENCE = 2

KEEP_DESPITE_STOPWORD = {
    "nicht", "kein", "keine", "keinen", "keinem", "keiner",
    "sehr", "besonders", "äusserst", "überaus", "äußerst",
    "echt", "echtes", "wirklich", "voll", "absolut",
    "sieben", "besser", "gut", "natürlich", "endlich",
    "zusammen", "alle", "aller", "bekannt", "allein", "lange",
    # Präpositionen für Phrasensuche
    "auf", "in", "an", "bei", "mit", "für", "von", "zu", "über",
    "unter", "durch", "aus", "um", "bis",
}


# ── spaCy laden ───────────────────────────────────────────────────────────────

def load_spacy_model() -> spacy.Language:
    try:
        nlp = spacy.load("de_core_news_lg")
        print("  spaCy-Modell geladen: de_core_news_lg")
    except OSError:
        raise SystemExit(
            "\n❌ spaCy-Modell nicht gefunden.\n"
            "   Installieren mit: python -m spacy download de_core_news_lg\n"
        )
    for word in KEEP_DESPITE_STOPWORD:
        nlp.vocab[word].is_stop = False
    return nlp


# ── Toponym-Filter laden ──────────────────────────────────────────────────────

def load_toponym_filter(filepath: str | None) -> set[str]:
    if filepath is None:
        print("  Kein Toponym-Filter — Toponyme werden behalten.")
        return set()
    with open(filepath, encoding="utf-8") as f:
        toponyms = set(json.load(f))
    print(f"  Toponym-Filter geladen: {len(toponyms):,} Einträge")
    return toponyms


# ── Token-Filter ──────────────────────────────────────────────────────────────

def is_relevant_token_lemma(token, toponym_filter: set[str]) -> bool:
    """Für lemmatisierten Corpus: Stoppwörter drin, Toponyme raus."""
    if token.is_punct or token.is_digit or token.is_space:
        return False
    if token.like_url or token.like_email:
        return False
    if len(token.text) <= 1:
        return False
    if token.lemma_.lower() in toponym_filter:
        return False
    if token.text.lower() in toponym_filter:
        return False
    return True


# ── Artikel verarbeiten ───────────────────────────────────────────────────────

def process_article_lemma(
    text: str,
    nlp: spacy.Language,
    toponym_filter: set[str],
) -> list[str]:
    """Gibt lemmatisierte Sätze zurück (mit Stoppwörtern, ohne Toponyme)."""
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        tokens = [
            token.lemma_.lower()
            for token in sent
            if is_relevant_token_lemma(token, toponym_filter)
        ]
        if len(tokens) >= MIN_TOKENS_PER_SENTENCE:
            sentences.append(" ".join(tokens))
    return sentences


def process_article_original(text: str, nlp: spacy.Language) -> list[str]:
    """Gibt Original-Sätze zurück (nur Satzsegmentierung)."""
    doc = nlp(text)
    return [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip()) > 10
    ]


# ── Datei-Verarbeitung ────────────────────────────────────────────────────────

def process_file(
    filepath: Path,
    out_lemma_dir: Path,
    out_orig_dir: Path,
    nlp: spacy.Language,
    toponym_filter: set[str],
) -> None:
    """
    Verarbeitet eine bereinigte TSV-Datei (ein Standort).
    Speichert alle Artikel als eine .txt Datei pro Standort.
    """
    site_name = filepath.stem.replace("_clean", "")
    print(f"\n  Standort: {site_name}")

    df = pd.read_csv(filepath, sep="\t", dtype=str, on_bad_lines="warn")
    df = df.dropna(subset=["content"])
    print(f"  Artikel: {len(df)}")

    all_lemma_sentences  = []
    all_orig_sentences   = []

    for _, row in df.iterrows():
        content = str(row.get("content", "")).strip()
        if not content:
            continue

        # Lemmatisiert
        sents_lemma = process_article_lemma(content, nlp, toponym_filter)
        all_lemma_sentences.extend(sents_lemma)

        # Original
        sents_orig = process_article_original(content, nlp)
        all_orig_sentences.extend(sents_orig)

    # Pro Standort eine Datei speichern
    lemma_path = out_lemma_dir / f"{site_name}.txt"
    orig_path  = out_orig_dir  / f"{site_name}.txt"

    lemma_path.write_text("\n".join(all_lemma_sentences), encoding="utf-8")
    orig_path.write_text( "\n".join(all_orig_sentences),  encoding="utf-8")

    print(f"  Sätze lemmatisiert: {len(all_lemma_sentences):,} → {lemma_path.name}")
    print(f"  Sätze original:     {len(all_orig_sentences):,}  → {orig_path.name}")


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AntConc Corpus-Vorbereitung (pro Standort eine Datei)"
    )
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--toponym-filter", default=None)
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    out_lemma  = output_dir / "lemmatized"
    out_orig   = output_dir / "original"

    out_lemma.mkdir(parents=True, exist_ok=True)
    out_orig.mkdir( parents=True, exist_ok=True)

    print("\nAntConc Corpus-Vorbereitung")
    print(f"  Input:        {input_dir}")
    print(f"  Lemmatisiert: {out_lemma}")
    print(f"  Original:     {out_orig}\n")

    nlp            = load_spacy_model()
    toponym_filter = load_toponym_filter(args.toponym_filter)

    files = sorted(input_dir.glob("*.tsv"))
    if not files:
        print(f"❌ Keine .tsv Dateien in {input_dir}")
        return

    print(f"\n  Dateien gefunden: {len(files)}\n")

    for f in files:
        process_file(f, out_lemma, out_orig, nlp, toponym_filter)

    print(f"\n{'─'*50}")
    print(f"✓ Fertig. Dateien in:")
    print(f"  {out_lemma}/  → lemmatisierte Korpora")
    print(f"  {out_orig}/   → Original-Korpora")
    print("\nIn AntConc: File → Open File(s) → Dateien auswählen")
    print("Done.")


if __name__ == "__main__":
    main()
