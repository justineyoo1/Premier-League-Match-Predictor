"""Cleaning utilities for raw match datasets.

Implement functions to handle missing values, normalize team names, parse dates,
and produce a tidy, schema-consistent DataFrame for feature engineering.
"""

from __future__ import annotations

import logging
from typing import Iterable, Dict

import pandas as pd


logger = logging.getLogger(__name__)


TEAM_NORMALIZATION: Dict[str, str] = {
    # Add any known alternate names → canonical
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
}


def clean_matches(raw: pd.DataFrame, *, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Clean raw match data and return a tidy DataFrame.

    Args:
        raw: Raw input DataFrame.
        required_columns: Optional iterable of required columns to validate.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Cleaning raw match data with %d rows", len(raw))
    df = raw.copy()
    # Basic trimming of team names and normalization
    for col in ("HomeTeam", "AwayTeam"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(TEAM_NORMALIZATION)

    # Result normalization: ensure in {H, D, A}
    if "FTR" in df.columns:
        df["FTR"] = df["FTR"].astype("string").str.upper().str.strip()

    # Drop rows with missing critical fields
    critical = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] if c in df.columns]
    df = df.dropna(subset=critical)

    # Coerce goal columns to ints if present
    for goal_col in ("FTHG", "FTAG"):
        if goal_col in df.columns:
            df[goal_col] = pd.to_numeric(df[goal_col], errors="coerce")
    df = df.dropna(subset=[c for c in ("FTHG", "FTAG") if c in df.columns])
    for goal_col in ("FTHG", "FTAG"):
        if goal_col in df.columns:
            df[goal_col] = df[goal_col].astype(int)

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            logger.warning("Missing required columns: %s", missing)
    return df


