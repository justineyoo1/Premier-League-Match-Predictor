"""Cleaning utilities for raw match datasets.

Implement functions to handle missing values, normalize team names, parse dates,
and produce a tidy, schema-consistent DataFrame for feature engineering.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


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
    # Placeholder: implement schema checks, NA handling, date parsing, team normalization
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            logger.warning("Missing required columns: %s", missing)
    return df


