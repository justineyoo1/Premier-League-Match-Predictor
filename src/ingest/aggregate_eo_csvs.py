"""Aggregate EO CSVs into a single raw matches file.

Loads all CSVs under the `EO/` directory, selects Date, HomeTeam, AwayTeam,
FTHG, FTAG, FTR, normalizes the Date to YYYY-MM-DD, concatenates them, and
writes to `data/raw/matches.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eo_dir", default="EO", help="Directory with EO CSVs")
    parser.add_argument(
        "--out_csv", default="data/raw/matches.csv", help="Output concatenated CSV path"
    )
    return parser.parse_args()


def load_and_select(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df[COLUMNS].copy()
    # football-data dates can be in different formats across seasons
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    df["Date"] = df["Date"].astype("string")
    return df


def main(eo_dir: str, out_csv: str) -> None:
    in_dir = Path(eo_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = sorted(in_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {in_dir}")

    frames = [load_and_select(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    # Drop rows with invalid dates after coercion
    df = df.dropna(subset=["Date"]).copy()
    # Ensure consistent YYYY-MM-DD string
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.eo_dir, args.out_csv)


