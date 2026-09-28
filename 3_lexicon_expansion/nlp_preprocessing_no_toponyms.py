"""
NLP Preprocessing für den Swissdox-Corpus
==========================================
Dieses Skript bereitet den gesamten bereinigten Corpus für zwei Zwecke vor:

    1. Embedding-Training: Eine Textdatei mit einem Satz pro Zeile,
       lemmatisiert und ohne Stoppwörter; das Standardformat für
       Word2Vec und fastText.

    2. Lexikon-Anwendung: Eine JSON-Datei pro Standort mit strukturierten
       Artikeln (Metadaten + Sätze), sodass später die Analyseeinheit
       (T-1, T, T+1) um das Toponym herum angewendet werden kann.
       --> Wurde in finalen Verlauf nicht verwendet. 

Verarbeitungsschritte pro Artikel:
    1. Satz-Segmentierung (spaCy)
    2. Tokenisierung
    3. Lemmatisierung
    4. Stoppwort-Entfernung (mit Ausnahmen für inhaltlich wichtige Wörter)
    5. Filterung von Satzzeichen und Zahlen

Benötigte Pakete:
    pip install spacy pandas
    python -m spacy download de_core_news_lg

Verwendung:
    python nlp_preprocessing.py --input cleaned/ --output preprocessed/
    python nlp_preprocessing.py --input cleaned/ --output preprocessed/ --toponym-filter filter/
"""

import argparse
import json
import re
import pandas as pd
import spacy
from pathlib import Path


# ── Konfiguration ─────────────────────────────────────────────────────────────

# Mindestzahl Tokens pro Satz nach Preprocessing
MIN_TOKENS_PER_SENTENCE = 2

# Zusätzliche Stoppwörter
CUSTOM_STOPWORDS = {}

# Wörter die spaCy als Stoppwörter markiert, die aber behalten werden sollen
KEEP_DESPITE_STOPWORD = {
    "nicht", "kein", "keine", "keinen", "keinem", "keiner",
    "sehr", "besonders", "äusserst", "überaus", "äußerst", "echt", 
    "echtes", "wirklich", "voll", "absolut", "sieben", "besser", "gut", "natürlich", 
    "endlich", "zusammen", "alle", "aller", "bekannt", "allein", "lange"
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
    for word in CUSTOM_STOPWORDS:
        nlp.vocab[word].is_stop = True
    for word in KEEP_DESPITE_STOPWORD:
        nlp.vocab[word].is_stop = False
    return nlp

# ── Toponym-Filter laden ──────────────────────────────────────────────────────
 
def load_toponym_filter(filepath: str | None) -> set[str]:
    """
    Lädt den Toponym-Filter aus einer JSON-Datei (erstellt von
    build_toponym_filter.py). Gibt ein leeres Set zurück falls
    kein Filter angegeben wurde.
    """
    if filepath is None:
        print("  Kein Toponym-Filter angegeben — Toponyme werden behalten.")
        return set()
    with open(filepath, encoding="utf-8") as f:
        toponyms = set(json.load(f))
    print(f"  Toponym-Filter geladen: {len(toponyms):,} Einträge")
    return toponyms


# ── Token-Filter ──────────────────────────────────────────────────────────────
 
def is_relevant_token(token, toponym_filter: set[str]) -> bool:
    """
    Gibt True zurück wenn der Token für das Embedding-Training
    behalten werden soll.
 
    Verworfen werden:
        - Stoppwörter (mit Ausnahmen)
        - Satzzeichen, Zahlen, Leerzeichen
        - URLs und E-Mail-Adressen
        - Tokens mit einem Zeichen
        - Toponyme (falls Filter geladen)
    """
    if token.is_stop:
        return False
    if token.is_punct or token.is_digit or token.is_space:
        return False
    if token.like_url or token.like_email:
        return False
    if len(token.text) <= 1:
        return False
 
    # Toponym-Check: lowercase Lemma und Originaltext prüfen
    if token.lemma_.lower() in toponym_filter:
        return False
    if token.text.lower() in toponym_filter:
        return False
 
    return True
 
 
def process_sentence(sent, toponym_filter: set[str]) -> list[str]:
    """Gibt die Lemmata aller relevanten Tokens eines Satzes zurück."""
    return [
        token.lemma_.lower()
        for token in sent
        if is_relevant_token(token, toponym_filter)
    ]

# ── Artikel-Verarbeitung ──────────────────────────────────────────────────────
 
def process_article(
    text: str,
    nlp: spacy.Language,
    toponym_filter: set[str]
) -> list[dict]:
    """
    Verarbeitet einen Artikel und gibt eine Liste von Satz-Dicts zurück.
    Jedes Dict enthält:
        - original:    Originaltext des Satzes
        - tokens:      Liste der Lemmata
        - token_count: Anzahl Tokens nach Preprocessing
    """
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        tokens = process_sentence(sent, toponym_filter)
        if len(tokens) < MIN_TOKENS_PER_SENTENCE:
            continue
        sentences.append({
            "original":    sent.text.strip(),
            "tokens":      tokens,
            "token_count": len(tokens)
        })
    return sentences


# ── Datei-Verarbeitung ────────────────────────────────────────────────────────
 
def process_file(
    filepath: Path,
    output_dir: Path,
    nlp: spacy.Language,
    toponym_filter: set[str]
) -> list[list[str]]:
    """
    Verarbeitet eine bereinigte TSV-Datei (ein Standort).
    Speichert eine JSON-Datei und gibt alle Token-Listen zurück.
    """
    site_name = filepath.stem.replace("_clean", "")
    print(f"\n  Standort: {site_name}")
 
    df = pd.read_csv(filepath, sep="\t", dtype=str, on_bad_lines="warn")
    df = df.dropna(subset=["content"])
    print(f"  Artikel: {len(df)}")
 
    all_sentences_for_training = []
    articles_structured = []
 
    for _, row in df.iterrows():
        content = str(row.get("content", "")).strip()
        if not content:
            continue
 
        sentences = process_article(content, nlp, toponym_filter)
        if not sentences:
            continue
 
        articles_structured.append({
            "id":          str(row.get("id", "")),
            "pubtime":     str(row.get("pubtime", "")),
            "medium_name": str(row.get("medium_name", "")),
            "head":        str(row.get("head", "")),
            "site":        site_name,
            "sentences":   sentences
        })
 
        for sent in sentences:
            all_sentences_for_training.append(sent["tokens"])
 
    # JSON pro Standort speichern
    json_path = output_dir / f"{site_name}_preprocessed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles_structured, f, ensure_ascii=False, indent=2)
 
    total_sentences = sum(len(a["sentences"]) for a in articles_structured)
    print(f"  Sätze nach Preprocessing: {total_sentences}")
    print(f"  Gespeichert: {json_path.name}")
 
    return all_sentences_for_training


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NLP Preprocessing des Swissdox-Corpus"
    )
    parser.add_argument(
        "--input", required=True,
        help="Ordner mit den bereinigten *_clean.tsv Dateien"
    )
    parser.add_argument(
        "--output", required=True,
        help="Ausgabeordner für preprocesste Dateien"
    )
    parser.add_argument(
        "--toponym-filter", default=None,
        help="Pfad zur toponym_filter.json (erstellt von build_toponym_filter.py)"
    )
    args = parser.parse_args()
 
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print("\nNLP Preprocessing")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}\n")
 
    # Modell und Filter laden
    nlp             = load_spacy_model()
    toponym_filter  = load_toponym_filter(args.toponym_filter)
 
    # Dateien verarbeiten
    files = sorted(input_dir.glob("*.tsv"))
    if not files:
        print(f"❌ Keine .tsv Dateien in {input_dir}")
        return
 
    print(f"\n  Dateien gefunden: {len(files)}\n")
 
    all_sentences = []
    for f in files:
        sentences = process_file(f, output_dir, nlp, toponym_filter)
        all_sentences.extend(sentences)
 
    # Embedding-Trainings-Datei speichern
    training_path = output_dir / "corpus_for_embedding_training.txt"
    with open(training_path, "w", encoding="utf-8") as f:
        for tokens in all_sentences:
            f.write(" ".join(tokens) + "\n")
 
    print(f"\n{'─'*50}")
    print(f"✓ Embedding-Training-Corpus: {training_path.name}")
    print(f"  Sätze total:  {len(all_sentences):,}")
    print(f"  Tokens total: {sum(len(s) for s in all_sentences):,}")
    print("\nDone.")


if __name__ == "__main__":
    main()


