"""
Frequency Analysis Corpus Builder
===================================
Erstellt einen Analyse-Corpus für die Frequenzanalyse.

Vorgehen pro Standort:
    1. Alle Artikel einlesen
    2. Sätze lemmatisieren (OHNE Stoppwortentfernung, OHNE Toponym-Entfernung)
    3. Analyseeinheit T-1/T/T+1 anwenden: nur Sätze behalten die ein
       Standort-Toponym enthalten, plus je ein Satz davor und danach
    4. Duplikate entfernen (jeder Satz maximal einmal)
    5. Pro Standort: JSON (strukturiert) + TXT (flach) speichern

Output-Struktur:
    frequency_corpus/
        oeschinensee.json   ← strukturiert mit Artikelmetadaten
        oeschinensee.txt    ← flach, eine Zeile pro Satz
        seealpsee.json
        ...

Verwendung:
    python build_frequency_corpus.py
        --input  cleaned/
        --output frequency_corpus/

Benötigte Pakete:
    pip install spacy pandas
    python -m spacy download de_core_news_lg
"""

import argparse
import json
import re
import pandas as pd
import spacy
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# TOPONYME
# ══════════════════════════════════════════════════════════════════════════════
# Format: "dateiname_ohne_clean": ["toponym1", "toponym2", ...]
# Gross-/Kleinschreibung egal — Suche läuft lowercase
# Ein Satz wird ausgewählt wenn er EINES der Toponyme enthält
#
# Dateinamen müssen dem Stamm der TSV-Dateien entsprechen
# (ohne "_clean.tsv"), z.B. "oeschinensee_clean.tsv" → "oeschinensee"

SITE_TOPONYMS = {
    "oeschinensee": [
        "oeschinensee",
        "oschinensee",
        "öschinensee",
    ],
    "seealpsee": [
        "seealpsee",
    ],
    "reuss_bremgarten": [
        "reuss",
        "bremgarten",
    ],
    "thur_thurauen": [
        "thurauen",
        "thur",          
    ],
    "robenhuserriet": [
        "robenhuserriet",
        "robenhauseriet",
        "robenhauser ried",
        "robenhauserried",
    ],
    "aegeriried": [
        "ägeriried",
        "aegeriried",
    ],
    "ufschoetti": [
        "ufschötti"
    ],
    "zuerichhorn": [
        "zürichhorn",
    ],
    "hochwacht_laegern": [
        "hochwacht",     # beide sind gültige Anker
        "lägern",
        "lägernhochwacht",
    ],
    "hochwacht_pfannenstiel": [
        "hochwacht",     # beide sind gültige Anker
        "pfannenstiel",
    ],
}

# ── Wortgrenzen-Matching ──────────────────────────────────────────────────────
# Toponyme in dieser Liste werden NUR als eigenständige Wörter gefunden (\b).
# Begründung: kurze Flussnamen wie "thur" und "reuss" kommen als Präfix in
# Komposita vor die nichts mit dem Studienstandort zu tun haben
# (z.B. "winterthur", "reusstal").
# Da beide Fälle Flussnamen sind, wird die Regel konsistent auf beide angewendet.
 
EXACT_MATCH_TOPONYMS = {
    "thur",
    "reuss",
}

# ══════════════════════════════════════════════════════════════════════════════


def load_spacy():
    try:
        nlp = spacy.load("de_core_news_lg")
        print("  spaCy geladen: de_core_news_lg")
        return nlp
    except OSError:
        raise SystemExit(
            "\n❌ spaCy-Modell fehlt.\n"
            "   python -m spacy download de_core_news_lg\n"
        )


def lemmatize_text(text: str, nlp) -> list[dict]:
    """
    Segmentiert den Text in Sätze und lemmatisiert jeden Token.
    Keine Stoppwortentfernung, keine Toponym-Entfernung.
    Alle Sätze werden behalten (keine Mindestlänge).

    Gibt eine Liste von Satz-Dicts zurück:
        {
            "original": "Originaltext des Satzes",
            "lemmas":   ["lemma1", "lemma2", ...],
            "lemma_str": "lemma1 lemma2 ..."  ← für Toponym-Suche
        }
    """
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            # Satzzeichen und reine Leerzeichen überspringen
            if token.is_punct or token.is_space:
                continue
            tokens.append(token.lemma_.lower())

        sentences.append({
            "original":  sent.text.strip(),
            "lemmas":    tokens,
            "lemma_str": " ".join(tokens),
        })
    return sentences


def contains_toponym(sent: dict, toponyms: list[str]) -> bool:
    """
    Prüft ob ein Satz eines der Standort-Toponyme enthält.
 
    Toponyme in EXACT_MATCH_TOPONYMS werden nur als eigenständige Wörter
    gefunden (Wortgrenzen via \b). Alle anderen Toponyme werden als
    Substring gesucht — Komposita wie "lägerngrat" werden also auch erfasst.
    """
    lemma_str = sent["lemma_str"]
    for topo in toponyms:
        t = topo.lower()
        if t in EXACT_MATCH_TOPONYMS:
            # Wortgrenzen erzwingen
            if re.search(r'\b' + re.escape(t) + r'\b', lemma_str):
                return True
        else:
            # Substring-Matching (Komposita erlaubt)
            if t in lemma_str:
                return True
    return False


def select_sentences(sentences: list[dict], toponyms: list[str]) -> list[dict]:
    """
    Wendet die T-1/T/T+1 Analyseeinheit an.

    Für jeden Satz der ein Toponym enthält werden die Indices
    i-1, i, i+1 in ein Set aufgenommen. Das Set wird sortiert
    und die Sätze in Reihenfolge zurückgegeben.

    Jeder Satz kommt dadurch maximal einmal vor — auch wenn
    Toponyme in aufeinanderfolgenden Sätzen erscheinen.
    """
    n = len(sentences)
    selected_indices = set()

    for i, sent in enumerate(sentences):
        if contains_toponym(sent, toponyms):
            if i > 0:
                selected_indices.add(i - 1)   # T-1
            selected_indices.add(i)             # T
            if i < n - 1:
                selected_indices.add(i + 1)     # T+1

    # Sortiert zurückgeben — Reihenfolge im Artikel bleibt erhalten
    return [sentences[i] for i in sorted(selected_indices)]


def process_file(
    filepath: Path,
    site_name: str,
    toponyms: list[str],
    nlp,
) -> list[dict]:
    """
    Verarbeitet eine TSV-Datei (ein Standort).
    Gibt eine Liste von Artikel-Dicts zurück.
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str, on_bad_lines="warn")
    df = df.dropna(subset=["content"])
    print(f"  Artikel: {len(df)}")

    articles = []
    total_selected = 0
    total_sentences = 0

    for _, row in df.iterrows():
        content = str(row.get("content", "")).strip()
        if not content:
            continue

        # Alle Sätze lemmatisieren
        sentences = lemmatize_text(content, nlp)
        total_sentences += len(sentences)

        # Analyseeinheit anwenden
        selected = select_sentences(sentences, toponyms)
        total_selected += len(selected)

        if not selected:
            continue

        articles.append({
            "id":          str(row.get("id", "")),
            "pubtime":     str(row.get("pubtime", ""))[:10],
            "medium_name": str(row.get("medium_name", "")),
            "site":        site_name,
            "sentences_total":    len(sentences),
            "sentences_selected": len(selected),
            "sentences": [
                {
                    "original":  s["original"],
                    "lemmas":    s["lemmas"],
                }
                for s in selected
            ],
        })

    print(f"  Sätze gesamt:    {total_sentences:,}")
    print(f"  Sätze ausgewählt (T-1/T/T+1): {total_selected:,}")
    print(f"  Artikel mit Treffern: {len(articles)}")

    return articles


def save_outputs(
    articles: list[dict],
    site_name: str,
    output_dir: Path,
) -> None:
    """
    Speichert JSON (strukturiert) und TXT (flach) pro Standort.
    """
    # JSON
    json_path = output_dir / f"{site_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    # TXT — eine Zeile pro Satz (lemmatisiert), Artikel durch Leerzeile getrennt
    txt_path = output_dir / f"{site_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for article in articles:
            for sent in article["sentences"]:
                f.write(" ".join(sent["lemmas"]) + "\n")
            f.write("\n")  # Leerzeile zwischen Artikeln

    total_sents = sum(len(a["sentences"]) for a in articles)
    print(f"  → {json_path.name} ({len(articles)} Artikel, {total_sents} Sätze)")
    print(f"  → {txt_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Frequency Analysis Corpus Builder (T-1/T/T+1)"
    )
    parser.add_argument(
        "--input", required=True,
        help="Ordner mit den bereinigten *_clean.tsv Dateien"
    )
    parser.add_argument(
        "--output", required=True,
        help="Ausgabeordner für den Frequency-Corpus"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFrequency Analysis Corpus Builder")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}\n")

    nlp = load_spacy()

    # Alle TSV-Dateien finden und mit SITE_TOPONYMS abgleichen
    files = sorted(input_dir.glob("*.tsv"))
    if not files:
        print(f"❌ Keine .tsv Dateien in {input_dir}")
        return

    matched = []
    for f in files:
        site_name = f.stem.replace("_clean", "")
        if site_name in SITE_TOPONYMS:
            matched.append((f, site_name, SITE_TOPONYMS[site_name]))
        else:
            print(f"  ⚠ Keine Toponyme definiert für: {site_name} — übersprungen")

    print(f"\n  Standorte gefunden: {len(matched)}\n")

    for filepath, site_name, toponyms in matched:
        print(f"\n{'─'*50}")
        print(f"  Standort: {site_name}")
        print(f"  Toponyme: {toponyms}")

        articles = process_file(filepath, site_name, toponyms, nlp)
        save_outputs(articles, site_name, output_dir)

    print(f"\n{'═'*50}")
    print("✓ Fertig.")
    print(f"  Corpus gespeichert in: {output_dir}")
    print("\nNächster Schritt: frequency_analysis.py auf die JSON-Dateien anwenden.")


if __name__ == "__main__":
    main()
