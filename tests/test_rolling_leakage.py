"""Leakage guard for rolling features.

Checks that rolling features use shifted windows (no future info).
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling import compute_team_rolling_features


def test_rolling_features_are_shifted():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15"]),
            "HomeTeam": ["A", "A", "A"],
            "AwayTeam": ["B", "B", "B"],
            "FTHG": [1, 2, 3],
            "FTAG": [0, 0, 0],
            "FTR": ["H", "H", "H"],
        }
    )
    out = compute_team_rolling_features(df, window_sizes=[2])
    # For the second match, home_gf_avg_w2 should equal previous GF=1
    row2 = out.iloc[1]
    assert abs(row2["home_gf_avg_w2"] - 1.0) < 1e-9
    # For the first match, there is no history -> NaN
    assert pd.isna(out.iloc[0]["home_gf_avg_w2"]) or out.iloc[0]["home_gf_avg_w2"] == 0

