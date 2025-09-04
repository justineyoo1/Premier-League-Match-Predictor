"""CLI to clean aggregated raw matches into a canonical dataset.

Reads the aggregated CSV (data/raw/matches.csv), enforces schema, normalizes
teams and result labels, and writes data/processed/matches_clean.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .schema import REQUIRED_COLUMNS, DTYPE_STRINGS, CANONICAL_DATE_FORMAT
from .cleaning import clean_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="data/raw/matches.csv")
    parser.add_argument("--out_csv", default="data/processed/matches_clean.csv")
    return parser.parse_args()


def main(in_csv: str, out_csv: str) -> None:
    inp = Path(in_csv)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, dtype={})
    # enforce required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    # enforce dtypes where possible
    for col, dtype in DTYPE_STRINGS.items():
        if col in df.columns:
            try:
                if col == "Date":
                    df[col] = pd.to_datetime(df[col], format=CANONICAL_DATE_FORMAT, errors="coerce")
                elif dtype == "int64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("int64")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception:
                pass

    cleaned = clean_matches(df, required_columns=REQUIRED_COLUMNS)
    cleaned["Date"] = pd.to_datetime(cleaned["Date"]).dt.strftime(CANONICAL_DATE_FORMAT)
    cleaned.to_csv(out, index=False)
    print(f"Wrote cleaned CSV with {len(cleaned)} rows to {out}")


if __name__ == "__main__":
    args = parse_args()
    main(args.in_csv, args.out_csv)


