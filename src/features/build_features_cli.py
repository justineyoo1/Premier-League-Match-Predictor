"""Build features and create time-based splits.

Reads cleaned matches, computes rolling and Elo features in a leakage-safe
fashion, and writes train/valid/test CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .rolling import compute_team_rolling_features
from .elo import compute_elo_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="data/processed/matches_clean.csv")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--windows", nargs="*", type=int, default=[3, 5, 10])
    parser.add_argument("--valid_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    return parser.parse_args()


def time_based_split(df: pd.DataFrame, valid_frac: float, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    n_train = n - n_valid - n_test
    train = df.iloc[:n_train].copy()
    valid = df.iloc[n_train : n_train + n_valid].copy()
    test = df.iloc[n_train + n_valid :].copy()
    return train, valid, test


def main(in_csv: str, out_dir: str, windows: List[int], valid_frac: float, test_frac: float) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime

    # Compute features
    df = compute_team_rolling_features(df, window_sizes=windows)
    df = compute_elo_features(df)

    # Define target as 3-class or binary later; for now, keep FTR and probabilities to be modeled
    train, valid, test = time_based_split(df, valid_frac, test_frac)

    train.to_csv(out / "train.csv", index=False)
    valid.to_csv(out / "valid.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print("Wrote:")
    print(f"  {len(train)} rows -> {out / 'train.csv'}")
    print(f"  {len(valid)} rows -> {out / 'valid.csv'}")
    print(f"  {len(test)} rows -> {out / 'test.csv'}")


if __name__ == "__main__":
    args = parse_args()
    main(args.in_csv, args.out_dir, args.windows, args.valid_frac, args.test_frac)


