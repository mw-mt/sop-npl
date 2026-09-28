"""
Toponym Anchor Inspector
==========================
Zeigt für jeden Standort welche Toponyme als Ankerpunkte verwendet werden
und wie oft sie im Frequency-Corpus vorkommen — inklusive ob sie als
Teil eines längeren Wortes gefunden werden.

So kannst du entscheiden ob du Wortgrenzen erzwingen willst oder nicht.

Output:
    Für jedes Toponym:
        - Gesamtanzahl Treffer (substring-Matching, wie aktuell)
        - Davon: exakte Wortgrenzen-Treffer
        - Davon: nur als Teil eines längeren Wortes
        - Beispiele für Kompositum-Treffer

Verwendung:
    python inspect_toponym_anchors.py --corpus frequency_corpus/
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


# ── Toponyme (identisch mit build_frequency_corpus.py) ───────────────────────

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
        "ufschötti",
    ],
    "zuerichhorn": [
        "zürichhorn",
    ],
    "hochwacht_laegern": [
        "hochwacht",
        "lägern",
        "lägernhochwacht",
    ],
    "hochwacht_pfannenstiel": [
        "hochwacht",
        "pfannenstiel",
    ],
}

MAX_EXAMPLES = 5  # Wie viele Kompositum-Beispiele pro Toponym anzeigen


def inspect_corpus_file(
    filepath: Path,
    site: str,
    toponyms: list[str],
) -> dict:
    """
    Liest den Frequency-Corpus (JSON) und zählt für jedes Toponym:
        - total:     alle Sätze die es enthalten (substring)
        - exact:     Sätze mit Wortgrenze (\b)
        - compound:  Sätze nur als Kompositumteil (total - exact)
        - examples:  Beispiel-Token die das Toponym als Teil enthalten
    """
    with open(filepath, encoding="utf-8") as f:
        articles = json.load(f)

    results = {}

    for topo in toponyms:
        topo_lower  = topo.lower()
        pattern_sub = topo_lower                           # substring
        pattern_wb  = r'\b' + re.escape(topo_lower) + r'\b'  # Wortgrenze

        total_count   = 0
        exact_count   = 0
        compound_examples = []

        for article in articles:
            for sent in article.get("sentences", []):
                lemma_str = " ".join(sent.get("lemmas", []))

                has_sub   = topo_lower in lemma_str
                has_exact = bool(re.search(pattern_wb, lemma_str))

                if has_sub:
                    total_count += 1
                    if has_exact:
                        exact_count += 1
                    else:
                        # Kompositum-Treffer: welche Tokens enthalten das Toponym?
                        if len(compound_examples) < MAX_EXAMPLES:
                            for token in sent.get("lemmas", []):
                                if topo_lower in token and token != topo_lower:
                                    if token not in compound_examples:
                                        compound_examples.append(token)

        results[topo] = {
            "total":    total_count,
            "exact":    exact_count,
            "compound": total_count - exact_count,
            "examples": compound_examples,
        }

    return results


def print_report(all_results: dict) -> None:
    """Gibt einen lesbaren Bericht aus."""
    print("\n" + "═" * 70)
    print("  TOPONYM ANCHOR INSPECTION")
    print("═" * 70)

    for site, topo_results in sorted(all_results.items()):
        print(f"\n{'─'*70}")
        print(f"  STANDORT: {site}")
        print(f"{'─'*70}")
        print(f"  {'Toponym':<30} {'Total':>7} {'Exact':>7} {'Compound':>9}  Beispiele")
        print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*9}  {'─'*20}")

        for topo, stats in topo_results.items():
            examples_str = ", ".join(stats["examples"]) if stats["examples"] else "—"
            flag = " ⚠" if stats["compound"] > 0 else ""
            print(
                f"  {topo:<30} "
                f"{stats['total']:>7} "
                f"{stats['exact']:>7} "
                f"{stats['compound']:>9}  "
                f"{examples_str}{flag}"
            )

    print(f"\n{'═'*70}")
    print("  ⚠  = Toponym kommt als Teil längerer Wörter vor")
    print("  Total    = alle Sätze die das Toponym enthalten (aktuelles Verhalten)")
    print("  Exact    = nur Sätze mit Wortgrenzen (\\b)")
    print("  Compound = Differenz (Kompositum-Treffer)")
    print("═" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Toponym Anchor Inspector"
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Ordner mit den Frequency-Corpus JSON-Dateien"
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    all_results = {}

    for site, toponyms in SITE_TOPONYMS.items():
        json_file = corpus_dir / f"{site}.json"
        if not json_file.exists():
            print(f"  ⚠ Datei nicht gefunden: {json_file.name} — übersprungen")
            continue

        print(f"  Lese: {json_file.name} ...")
        all_results[site] = inspect_corpus_file(json_file, site, toponyms)

    print_report(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
