"""
Word2Vec Embedding-Training für den Swissdox-Corpus
=====================================================
Trainiert ein Word2Vec-Modell auf dem preprocessten Corpus und speichert
es für die spätere Lexikon-Erweiterung.

Parameter:
    vector_size = 200   Vektorgrösse
    window      = 8     Kontextfenster links und rechts
    min_count   = 1     
    sg          = 1     Skip-Gram 
    workers     = 4     Parallele Threads (anpassen an CPU-Kerne)
    epochs      = 10    Trainingsdurchläufe über den Corpus

Benötigte Pakete:
    pip install gensim

Verwendung:
    python train_embeddings.py
        --input  preprocessed/corpus_for_embedding_training.txt
        --output models/

    # Mit eigenen Parametern:
    python train_embeddings.py
        --input  preprocessed/corpus_for_embedding_training.txt
        --output models/
        --vector-size 200
        --window 8
        --min-count 1
        --epochs 10
"""

import argparse
import time
import logging
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# ── Logging einrichten ────────────────────────────────────────────────────────
# Zeigt Trainingsfortschritt in der Konsole
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO
)


# ── Fortschritts-Callback ─────────────────────────────────────────────────────

class EpochLogger(CallbackAny2Vec):
    """
    Zeigt nach jeder Trainings-Epoche den Fortschritt an.
    Hilft einzuschätzen wie lange das Training noch dauert.
    """
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        elapsed = time.time() - self.start_time
        print(f"  Epoche {self.epoch} abgeschlossen  "
              f"({elapsed:.0f}s seit Start)")


# ── Corpus-Iterator ───────────────────────────────────────────────────────────

class SentenceCorpus:
    """
    Iterator über die Trainings-Datei.
    Liest die Datei Zeile für Zeile statt alles in den Speicher zu laden —
    wichtig bei grossen Corpora.
    Jede Zeile ist ein Satz, Tokens durch Leerzeichen getrennt.
    """
    def __init__(self, filepath: Path):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:  # Leere Zeilen überspringen
                    yield tokens

    def __len__(self):
        """Zählt die Sätze (wird für Fortschrittsanzeige gebraucht)."""
        count = 0
        with open(self.filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Word2Vec Embedding-Training auf dem Swissdox-Corpus"
    )
    parser.add_argument(
        "--input", required=True,
        help="Pfad zur corpus_for_embedding_training.txt"
    )
    parser.add_argument(
        "--output", required=True,
        help="Ausgabeordner für das trainierte Modell"
    )
    parser.add_argument(
        "--vector-size", type=int, default=200,
        help="Grösse der Wortvektoren (Standard: 200)"
    )
    parser.add_argument(
        "--window", type=int, default=8,
        help="Kontextfenster-Grösse (Standard: 8)"
    )
    parser.add_argument(
        "--min-count", type=int, default=1,
        help="Minimale Worthäufigkeit (Standard: 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Anzahl Trainingsepochen (Standard: 10)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Anzahl parallele Threads (Standard: 4)"
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nWord2Vec Embedding-Training")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_dir}")
    print(f"  vector_size: {args.vector_size}")
    print(f"  window:      {args.window}")
    print(f"  min_count:   {args.min_count}")
    print(f"  epochs:      {args.epochs}")
    print(f"  workers:     {args.workers}")

    # Corpus laden
    print("\nLese Corpus...")
    corpus = SentenceCorpus(input_path)
    n_sentences = len(corpus)
    print(f"  Sätze: {n_sentences:,}")

    # Modell initialisieren
    # sg=1 = Skip-Gram 
    print("\nInitialisiere Modell...")
    model = Word2Vec(
        vector_size = args.vector_size,
        window      = args.window,
        min_count   = args.min_count,
        sg          = 1,           # Skip-Gram
        workers     = args.workers,
        epochs      = args.epochs,
        seed        = 42           # Reproduzierbarkeit
    )

    # Vokabular aufbauen
    print("Baue Vokabular auf...")
    model.build_vocab(corpus)
    print(f"  Vokabular: {len(model.wv):,} Wörter")

    # Training
    print("\nStarte Training...")
    start = time.time()
    model.train(
        corpus,
        total_examples = model.corpus_count,
        epochs         = model.epochs,
        callbacks      = [EpochLogger()]
    )
    duration = time.time() - start
    print(f"\n  Training abgeschlossen in {duration:.0f}s "
          f"({duration/60:.1f} Minuten)")

    # Modell speichern
    # .model: vollständiges Modell (für weiteres Training oder Laden)
    # .kv:    nur die Vektoren (kleiner, für reine Abfragen)
    model_path = output_dir / "swissdox_word2vec.model"
    model.save(str(model_path))
    print(f"\n  Modell gespeichert: {model_path}")

    # ── Schnelltest: Ähnliche Wörter für einige SoP-Begriffe ─────────────────
    # Gibt einen ersten Eindruck ob das Modell sinnvolle Assoziationen
    # gelernt hat. Ausgabe ist rein zur visuellen Kontrolle.
    print("\n" + "─"*50)
    print("Schnelltest — ähnliche Wörter für ausgewählte SoP-Begriffe:")
    test_terms = ["schön", "stille", "heimelig", "wanderung", "aussicht", "idyllisch"]

    for term in test_terms:
        try:
            similar = model.wv.most_similar(term, topn=5)
            similar_str = ", ".join(
                f"{w} ({s:.2f})" for w, s in similar
            )
            print(f"  {term:15} →  {similar_str}")
        except KeyError:
            print(f"  {term:15} →  nicht im Vokabular")

    print("\nDone.")
    print(
        "\nNächster Schritt: Lexikon-Erweiterung mit expand_lexicon.py"
        "\n  --model  " + str(model_path)
    )


if __name__ == "__main__":
    main()
