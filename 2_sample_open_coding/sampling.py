"""
Stratified Sampling for Open Coding 
===============================================
Draws a stratified random sample from cleaned TSV datasets (one per study site)
and exports each article as a plain-text file ready for import into Taguette.

Output structure:
    output/
        Oeschinensee/
            001234_2018-06-12_NZZ.txt
            ...
        sampling_log.csv

Stratification: Five fixed 5-year strata covering 2000-2025.
    2000-2004 | 2005-2009 | 2010-2014 | 2015-2019 | 2020-2025

Articles are drawn proportionally per stratum.

Usage:
    python sampling.py --input cleaned/ --output sample/
    python sampling.py --input cleaned/ --output sample/ --n 50 --seed 42
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_N    = 50   # Target articles per study site
DEFAULT_SEED = 42   # Random seed for reproducibility

# Fixed 5-year strata covering the full study period 2000-2025
STRATA_EDGES  = [2000, 2005, 2010, 2015, 2020, 2026]  # right edge exclusive
STRATA_LABELS = ["2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2025"]


def load_cleaned(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", dtype=str, on_bad_lines="warn")
    if "pubtime" in df.columns:
        df["pubtime"] = pd.to_datetime(df["pubtime"], errors="coerce", utc=True)
    df = df.dropna(subset=["pubtime", "content"])
    df = df[df["content"].str.strip().str.len() > 0]
    return df


def clean_field(value) -> str:
    """Safely convert any field value to a clean string, handling NaN and None."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def assign_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each article to one of the fixed 5-year strata."""
    years = df["pubtime"].dt.year
    df = df.copy()
    df["stratum"] = pd.cut(
        years,
        bins=STRATA_EDGES,
        labels=STRATA_LABELS,
        right=False,
        include_lowest=True
    )
    return df


def stratified_sample(df: pd.DataFrame, n_total: int,
                      seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """
    Draw a proportional stratified sample across the fixed time strata.
    Returns the sampled dataframe and a per-stratum report.
    """
    df = assign_strata(df)

    stratum_counts = df["stratum"].value_counts()
    stratum_props  = stratum_counts / len(df)
    target_n       = (stratum_props * n_total).round().astype(int)

    # Correct for rounding so total always equals n_total exactly
    diff = n_total - target_n.sum()
    if diff != 0:
        target_n[target_n.idxmax()] += diff

    sampled = []
    report  = []

    for stratum in STRATA_LABELS:
        pool      = df[df["stratum"] == stratum]
        available = len(pool)
        target    = int(target_n.get(stratum, 0))
        actual    = min(target, available)

        if actual > 0:
            sampled.append(pool.sample(n=actual, random_state=seed))

        report.append({
            "stratum":   stratum,
            "available": available,
            "target":    target,
            "sampled":   actual,
            "shortfall": max(0, target - available),
        })

    return pd.concat(sampled).reset_index(drop=True), report


def safe_filename(text, max_len: int = 40) -> str:
    """Sanitise a value for use in a filename."""
    text = re.sub(r"[^\w\s-]", "", str(text), flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


def write_article(row: pd.Series, folder: Path):
    """
    Write one article as a plain-text file for Taguette import.
    Filename format: {id}_{date}_{medium}.txt
    """
    article_id = safe_filename(clean_field(row.get("id", "unknown")))
    date_str   = str(row["pubtime"])[:10]
    medium     = safe_filename(clean_field(row.get("medium_name", "unknown")), max_len=30)

    filename = f"{article_id}_{date_str}_{medium}.txt"
    filepath = folder / filename

    head    = clean_field(row.get("head",    ""))
    subhead = clean_field(row.get("subhead", ""))
    content = clean_field(row.get("content", ""))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"ID:      {clean_field(row.get('id', ''))}\n")
        f.write(f"Date:    {date_str}\n")
        f.write(f"Medium:  {clean_field(row.get('medium_name', ''))} "
                f"({clean_field(row.get('medium_code', ''))})\n")
        f.write(f"Rubric:  {clean_field(row.get('rubric', ''))}\n")
        f.write(f"Link:    {clean_field(row.get('article_link', ''))}\n")
        f.write(f"{'─' * 60}\n\n")
        if head:
            f.write(f"{head}\n")
        if subhead:
            f.write(f"{subhead}\n")
        f.write(f"\n{content}\n")


def process_site(filepath: Path, output_dir: Path,
                 n_total: int, seed: int) -> pd.DataFrame:
    """Process one study site: sample and export articles."""
    site_name = filepath.stem.replace("_clean", "").replace("_", " ").title()
    site_dir  = output_dir / safe_filename(site_name, max_len=60)
    site_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Study site: {site_name}")

    df = load_cleaned(filepath)
    print(f"  Articles after cleaning: {len(df)}")

    if len(df) < n_total:
        print(f"  WARNING: Only {len(df)} articles available "
              f"(target was {n_total}). Sampling all available articles.")
        sample = assign_strata(df).copy()
        report = [{"stratum": "all", "available": len(df),
                   "target": n_total, "sampled": len(df),
                   "shortfall": n_total - len(df)}]
    else:
        sample, report = stratified_sample(df, n_total, seed)

    # Print stratum report
    print(f"  {'Stratum':<14} {'Available':>10} {'Target':>8} "
          f"{'Sampled':>8} {'Shortfall':>10}")
    for r in report:
        flag = " ⚠" if r["shortfall"] > 0 else ""
        print(f"  {r['stratum']:<14} {r['available']:>10} {r['target']:>8} "
              f"{r['sampled']:>8} {r['shortfall']:>10}{flag}")
    print(f"  Total sampled: {len(sample)}")

    for _, row in sample.iterrows():
        write_article(row, site_dir)

    # Build log — stratum column is guaranteed present after assign_strata
    sample_log = sample.copy()
    sample_log["study_site"] = site_name
    log_cols = ["study_site", "id", "pubtime", "medium_name",
                "rubric", "language", "article_link"]
    if "stratum" in sample_log.columns:
        log_cols.insert(6, "stratum")
    return sample_log[log_cols]


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling for open coding in Taguette")
    parser.add_argument("--input",  required=True,
                        help="Folder containing cleaned *_clean.tsv files")
    parser.add_argument("--output", required=True,
                        help="Output folder (one sub-folder per study site)")
    parser.add_argument("--n",    type=int, default=DEFAULT_N,
                        help=f"Target articles per site (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.tsv"))
    if not files:
        print(f"No .tsv files found in {input_dir}.")
        return

    print(f"\nStratified Sampling for Open Coding")
    print(f"  Input:       {input_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Target n:    {args.n} articles per site")
    print(f"  Random seed: {args.seed}")
    print(f"  Strata:      {' | '.join(STRATA_LABELS)}")
    print(f"  Files found: {len(files)}")

    all_logs = []
    for f in files:
        all_logs.append(process_site(f, output_dir, args.n, args.seed))

    log_path = output_dir / "sampling_log.csv"
    pd.concat(all_logs).reset_index(drop=True).to_csv(
        log_path, index=False, encoding="utf-8-sig")
    print(f"\n  Sampling log saved: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
