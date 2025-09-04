"""Basic tests for ingestion/cleaning.

Ensures critical columns exist and dtypes are sane after cleaning.
"""

from __future__ import annotations

import pandas as pd

from src.ingest.cleaning import clean_matches


def test_clean_matches_basic():
    df = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-08"],
            "HomeTeam": ["Man Utd", "Spurs"],
            "AwayTeam": ["Chelsea", "Arsenal"],
            "FTHG": [2, 1],
            "FTAG": [1, 1],
            "FTR": ["H", "D"],
        }
    )
    out = clean_matches(df)
    assert set(["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]).issubset(out.columns)
    assert out["FTHG"].dtype.kind in {"i", "u"}
    assert out["FTAG"].dtype.kind in {"i", "u"}

