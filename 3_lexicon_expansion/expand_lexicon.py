"""
Lexikon-Erweiterung via Word2Vec (v2 — mit Seed-Wörtern)
=========================================================
Liest Lexikon 1a v3 (Excel) ein, das für jeden Begriff ein manuell
definiertes Seed-Wort enthält. Für jeden Seed wird:
    - spaCy-Lemmatisierung angewendet
    - geprüft ob der Seed ein Stopwort ist
    - geprüft ob der Seed im Word2Vec-Vokabular ist
    - eine Kandidatenliste erstellt (oder ein Statushinweis ausgegeben)

Die Output-Excel zeigt immer den Originalbegriff und den Seed nebeneinander,
auch wenn kein Kandidat gefunden werden konnte.

Struktur der Input-Excel:
    Spaltenpaare: [Kategorie-Begriffe | Kategorie-Seeds]
    Zeile 0: Kategorienamen und Seed-Spaltennamen
    Zeilen 1+: Begriffe und ihre Seeds (NaN = kein Seed definiert)

Verwendung:
    python expand_lexicon.py
        --lexicon  seed_lexicon.xlsx
        --model    models/swissdox_word2vec.model
        --output   lexicon_expansion_review.xlsx
        --topn     20
        --threshold 0.35
"""

import argparse
import pandas as pd
import spacy
from pathlib import Path
from gensim.models import Word2Vec


# ── Lexikon einlesen ──────────────────────────────────────────────────────────

def load_lexicon_with_seeds(filepath: Path) -> list[dict]:
    """
    Liest die Excel-Datei ein und gibt eine Liste von Einträgen zurück.
    Jeder Eintrag ist ein Dict mit:
        - kategorie:     Name der Kategorie
        - originalbegriff: der vollständige Originalterm
        - seed:          das manuell definierte Seed-Wort (oder None)
    """
    df = pd.read_excel(filepath, header=None)

    entries = []

    # Spaltenpaare durchgehen: 0+1, 2+3, 4+5, ...
    for col_idx in range(0, df.shape[1] - 1, 2):
        term_col  = col_idx
        seed_col  = col_idx + 1

        # Kategoriename aus Zeile 0
        cat_name = str(df.iloc[0, term_col]).strip()
        if cat_name in ("nan", "NaN", ""):
            continue

        # Zeilen 1+ = Begriffe und Seeds
        for row_idx in range(1, df.shape[0]):
            term = str(df.iloc[row_idx, term_col]).strip()
            seed = str(df.iloc[row_idx, seed_col]).strip()

            # Leere Zellen überspringen
            if term in ("nan", "NaN", ""):
                continue

            entries.append({
                "kategorie":      cat_name,
                "originalbegriff": term,
                "seed":           None if seed in ("nan", "NaN", "") else seed,
            })

    return entries


# ── Seed verarbeiten ──────────────────────────────────────────────────────────

def process_seed(seed: str, nlp, model: Word2Vec) -> dict:
    """
    Verarbeitet ein einzelnes Seed-Wort:
        1. spaCy-Lemmatisierung
        2. Stopwort-Prüfung
        3. Vokabular-Prüfung
    Gibt ein Dict mit Status und Lemma zurück.
    """
    doc = nlp(seed)

    # Nur das erste inhaltliche Token verwenden
    # (Seeds sollten Einzelwörter sein)
    token = doc[0] if len(doc) > 0 else None

    if token is None:
        return {
            "lemma":      seed.lower(),
            "is_stop":    False,
            "in_vocab":   False,
            "status":     "Kein Token nach spaCy",
        }

    lemma = token.lemma_.lower()
    is_stop = token.is_stop

    return {
        "lemma":    lemma,
        "is_stop":  is_stop,
        "in_vocab": lemma in model.wv,
        "status":   (
            "Stopwort" if is_stop
            else "Nicht im Vokabular" if lemma not in model.wv
            else "OK"
        ),
    }


# ── Kandidaten suchen ─────────────────────────────────────────────────────────

def find_candidates(
    lemma: str,
    model: Word2Vec,
    topn: int,
    threshold: float,
    exclude_lemmas: set[str],
) -> list[tuple[str, float]]:
    """
    Findet die ähnlichsten Wörter zum Seed-Lemma.
    Schliesst Lemmata die bereits im Lexikon sind aus.
    """
    raw = model.wv.most_similar(lemma, topn=topn + len(exclude_lemmas) + 50)
    results = []
    for word, score in raw:
        if score < threshold:
            break
        if word in exclude_lemmas:
            continue
        results.append((word, round(score, 3)))
        if len(results) >= topn:
            break
    return results


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lexikon-Erweiterung via Word2Vec mit Seed-Wörtern"
    )
    parser.add_argument("--lexicon",    required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--topn",       type=int,   default=20)
    parser.add_argument("--threshold",  type=float, default=0.35)
    args = parser.parse_args()

    print("\nLexikon-Erweiterung (mit Seed-Wörtern)")
    print(f"  Lexikon:       {args.lexicon}")
    print(f"  Modell:        {args.model}")
    print(f"  Kandidaten:    {args.topn} pro Seed")
    print(f"  Schwellenwert: {args.threshold}\n")

    # Modell und spaCy laden
    print("Lade Modell...")
    model = Word2Vec.load(args.model)
    print(f"  Vokabular: {len(model.wv):,} Wörter")

    print("Lade spaCy...")
    nlp = spacy.load("de_core_news_lg")
    KEEP_DESPITE_STOPWORD = {
         "nicht", "kein", "keine", "keinen", "keinem", "keiner",
        "sehr", "besonders", "äusserst", "überaus", "äußerst", "echt", 
        "echtes", "wirklich", "voll", "absolut", "sieben", "besser", "gut", "natürlich", 
        "endlich", "zusammen", "alle", "aller", "bekannt", "allein", "lange"
    }
    for word in KEEP_DESPITE_STOPWORD:
        nlp.vocab[word].is_stop = False

    # Lexikon einlesen
    print("Lese Lexikon...")
    entries = load_lexicon_with_seeds(Path(args.lexicon))
    print(f"  {len(entries)} Einträge gelesen\n")

    # Alle Seed-Lemmata als Ausschlussliste aufbauen
    all_seed_lemmas = set()
    for entry in entries:
        if entry["seed"]:
            lemma = nlp(entry["seed"])[0].lemma_.lower()
            all_seed_lemmas.add(lemma)

    # Ergebnisse sammeln
    kandidaten_rows = []  # Einträge MIT Kandidaten
    status_rows     = []  # Einträge OHNE Kandidaten (Stopwort / nicht im Vokabular / kein Seed)

    for entry in entries:
        kategorie      = entry["kategorie"]
        originalbegriff = entry["originalbegriff"]
        seed           = entry["seed"]

        # ── Kein Seed definiert ──────────────────────────────────────────────
        if seed is None:
            status_rows.append({
                "Kategorie":       kategorie,
                "Originalbegriff": originalbegriff,
                "Seed":            "",
                "Lemma":           "",
                "Status":          "Kein Seed definiert",
            })
            continue

        # ── Seed verarbeiten ─────────────────────────────────────────────────
        result = process_seed(seed, nlp, model)
        lemma  = result["lemma"]
        status = result["status"]

        # ── Stopwort oder nicht im Vokabular → Status-Sheet ──────────────────
        if status != "OK":
            status_rows.append({
                "Kategorie":       kategorie,
                "Originalbegriff": originalbegriff,
                "Seed":            seed,
                "Lemma":           lemma,
                "Status":          status,
            })
            continue

        # ── Kandidaten suchen ────────────────────────────────────────────────
        candidates = find_candidates(
            lemma, model, args.topn, args.threshold, all_seed_lemmas
        )

        if not candidates:
            status_rows.append({
                "Kategorie":       kategorie,
                "Originalbegriff": originalbegriff,
                "Seed":            seed,
                "Lemma":           lemma,
                "Status":          f"Keine Kandidaten über Schwellenwert {args.threshold}",
            })
            continue

        # ── Kandidaten in Output-Tabelle schreiben ───────────────────────────
        for kandidat, score in candidates:
            kandidaten_rows.append({
                "Kategorie":       kategorie,
                "Originalbegriff": originalbegriff,
                "Seed":            seed,
                "Lemma (spaCy)":   lemma,
                "Kandidat":        kandidat,
                "Ähnlichkeit":     score,
                "Aufnehmen":       "",   # ja / nein
                "Neue Kategorie":  "",   # falls Kandidat woanders passt
                "Notiz":           "",
            })

    # ── Excel speichern ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    kandidaten_df = pd.DataFrame(kandidaten_rows)
    status_df     = pd.DataFrame(status_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        kandidaten_df.to_excel(writer, sheet_name="Kandidaten",   index=False)
        status_df.to_excel(    writer, sheet_name="Status",        index=False)

    print(f"{'─'*50}")
    print(f"✓ Kandidaten:  {len(kandidaten_rows):,} Einträge  → Sheet 'Kandidaten'")
    print(f"✓ Status:      {len(status_rows):,} Einträge  → Sheet 'Status'")
    print(f"✓ Gespeichert: {output_path}")
    print("\nNächster Schritt: Excel öffnen, Spalte 'Aufnehmen' mit ja/nein füllen.")


if __name__ == "__main__":
    main()
