# Code zur Masterarbeit

Python-Skripte der Masterarbeit: Analyse von Sense-of-Place-Sprache in Schweizer
Zeitungsartikeln (Swissdox) an zehn Standorten. Die Skripte bilden eine
durchgehende Pipeline von der Korpus-Bereinigung bis zu Statistik und
Diagrammen; die Ordnernummerierung (1–7) entspricht der Reihenfolge, in der
die Schritte ausgeführt werden.


## Voraussetzungen

- Windows mit PowerShell
- Python 3.11 (getestet mit 3.11.9, läuft voraussichtlich ab 3.10)
- ca. 1 GB freier Speicher (Pakete und spaCy-Modell)

## Setup

Im Hauptordner des Repos:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` legt die virtuelle Umgebung `.venv` an, installiert die Pakete aus
`requirements.txt`, lädt das spaCy-Modell `de_core_news_lg` und trägt
`6_visualisations` in den Suchpfad der Umgebung ein (die Skripte in
`7_statistical_analysis` importieren `plot_style` und `visualize_categories_v2`
von dort).

Manuell, falls gewünscht:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download de_core_news_lg
$env:PYTHONPATH = "$PWD\6_visualisations"   # gilt nur für das aktuelle Fenster
```

## Daten (nicht im Repo)

Die folgenden Ordner müssen selbst angelegt werden;
sie enthalten Drittdaten, die nicht weitergegeben werden dürfen (Swissdox-
Lizenz bzw. swisstopo):

| Ordner | Inhalt |
|---|---|
| `data/articles_raw/` | Swissdox-Rohdaten, eine `.tsv` pro Standort |
| `swissnames3d/` | `swissNAMES3D_PKT.csv`, `swissNAMES3D_LIN.csv`, `swissNAMES3D_PLY.csv` (swisstopo) |

Im Repo enthalten sind die Lexika in `lexica/`:

| Datei | Verwendet von |
|---|---|
| `lexica/seed_lexicon.xlsx` | `3_lexicon_expansion/expand_lexicon.py` |
| `lexica/search_term_catalogue.xlsx` | `4_frequency_analysis/build_search_catalog.py` |

**Dateinamen:** Der Code erwartet die internen Standortnamen `oeschinensee`,
`seealpsee`, `thur_thurauen`, `reuss_bremgarten`, `robenhuserriet`, `aegeriried`,
`ufschoetti`, `zuerichhorn`, `hochwacht_laegern`, `hochwacht_pfannenstiel`
(ohne Umlaute, ohne `_raw`). 

## Pipeline

Alle Befehle mit korrekten relativen Pfaden stehen in
[`python_befehle.txt`](python_befehle.txt). Ergebnisse landen in `output/`.

| Schritt | Ordner | Zweck |
|---|---|---|
| 1 | `1_corpus_cleaning/` | HTML entfernen, exakte und Near-Duplicates entfernen |
| 2 | `2_sample_open_coding/` | Stratifiziertes Sample für das Open Coding (Taguette) |
| 3 | `3_lexicon_expansion/` | Toponym-Filter, Word2Vec, Lexikon-Erweiterung, AntConc-Korpus |
| 4 | `4_frequency_analysis/` | Frequenz-Korpus (T-1/T/T+1), Suchkatalog, Treffertabelle |
| 5 | `5_queries/` | Abfragen auf der Treffertabelle (u.a. Grundlage der Wordclouds) |
| 6 | `6_visualisations/` | Diagramme und Wordclouds |
| 7 | `7_statistical_analysis/` | Cosine Similarity, Partial-Mantel-Test, Kruskal-Wallis, Post-hoc-Tests |
