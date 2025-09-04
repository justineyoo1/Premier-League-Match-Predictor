"""Rolling statistics utilities.

Provide helpers to compute team-level rolling aggregates such as goals for/against,
win rates, and form over N previous matches. Implemented to be leakage-safe by
using shifted rolling windows (only prior matches are used).
"""

from __future__ import annotations

import logging
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _results_to_points(result: str, is_home_row: bool) -> int:
    if pd.isna(result):
        return 0
    r = str(result).upper()
    if r == "D":
        return 1
    if is_home_row:
        return 3 if r == "H" else 0
    return 3 if r == "A" else 0


def compute_team_rolling_features(matches: pd.DataFrame, *, window_sizes: List[int] | None = None) -> pd.DataFrame:
    """Compute leakage-safe rolling features for each match.

    For each team, compute rolling averages of goals for/against and points
    over the specified window sizes, using only past matches (shifted windows).

    Adds feature columns for home and away teams per window size:
      - home_gf_avg_w{n}, home_ga_avg_w{n}, home_pts_avg_w{n}
      - away_gf_avg_w{n}, away_ga_avg_w{n}, away_pts_avg_w{n}
    """
    if window_sizes is None:
        window_sizes = [3, 5, 10]

    df = matches.copy()
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
    df = df.sort_values("Date").reset_index(drop=True)

    # Build long team-match table (two rows per match: home and away)
    home = pd.DataFrame(
        {
            "Date": df["Date"],
            "Team": df["HomeTeam"],
            "GF": df["FTHG"],
            "GA": df["FTAG"],
            "Result": df["FTR"],
            "is_home": True,
        }
    )
    away = pd.DataFrame(
        {
            "Date": df["Date"],
            "Team": df["AwayTeam"],
            "GF": df["FTAG"],
            "GA": df["FTHG"],
            "Result": df["FTR"],
            "is_home": False,
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["points"] = [
        _results_to_points(r, is_home) for r, is_home in zip(long_df["Result"], long_df["is_home"])
    ]

    # Compute shifted rolling stats per team
    long_df = long_df.sort_values(["Team", "Date"]).reset_index(drop=True)
    feature_frames: Dict[int, pd.DataFrame] = {}
    for w in window_sizes:
        g = long_df.groupby("Team", group_keys=False)
        feats = g.apply(
            lambda x: pd.DataFrame(
                {
                    "Date": x["Date"],
                    "Team": x["Team"],
                    f"gf_avg_w{w}": x["GF"].shift(1).rolling(w, min_periods=1).mean(),
                    f"ga_avg_w{w}": x["GA"].shift(1).rolling(w, min_periods=1).mean(),
                    f"pts_avg_w{w}": x["points"].shift(1).rolling(w, min_periods=1).mean(),
                }
            )
        )
        feature_frames[w] = feats

    # Merge back to match rows for home and away teams
    result = df.copy()
    for w, feats in feature_frames.items():
        # home
        home_merge = feats.rename(
            columns={
                f"gf_avg_w{w}": f"home_gf_avg_w{w}",
                f"ga_avg_w{w}": f"home_ga_avg_w{w}",
                f"pts_avg_w{w}": f"home_pts_avg_w{w}",
            }
        )
        result = result.merge(
            home_merge,
            left_on=["Date", "HomeTeam"],
            right_on=["Date", "Team"],
            how="left",
        ).drop(columns=["Team"])  # drop merge helper

        # away
        away_merge = feats.rename(
            columns={
                f"gf_avg_w{w}": f"away_gf_avg_w{w}",
                f"ga_avg_w{w}": f"away_ga_avg_w{w}",
                f"pts_avg_w{w}": f"away_pts_avg_w{w}",
            }
        )
        away_merge = away_merge.rename(columns={"Team": "TeamAway"})
        result = result.merge(
            away_merge,
            left_on=["Date", "AwayTeam"],
            right_on=["Date", "TeamAway"],
            how="left",
        ).drop(columns=["TeamAway"])  # drop merge helper

    # Some earliest matches will have NaNs due to lack of history
    return result



