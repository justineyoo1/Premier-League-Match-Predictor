"""Elo rating feature utilities.

Provide functions to compute and update Elo ratings per team and derive Elo-based
features such as rating differential and expected result.

Implementation is sequential and leakage-safe: for each match, features are
derived from pre-match Elo values (based only on prior matches), then ratings
are updated after the result.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)

def compute_elo_features(matches: pd.DataFrame, *, k_factor: float = 20.0, base_rating: float = 1500.0) -> pd.DataFrame:
    """Compute Elo-based features per match using pre-match ratings.

    Adds columns: ``home_elo``, ``away_elo``, ``elo_diff``, ``home_exp``, ``away_exp``.
    """
    df = matches.copy()
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
    df = df.sort_values("Date").reset_index(drop=True)

    ratings: Dict[str, float] = {}
    home_elos = []
    away_elos = []
    home_exps = []
    away_exps = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        h_elo = ratings.get(home, base_rating)
        a_elo = ratings.get(away, base_rating)

        # expected score for home
        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400.0))
        exp_away = 1.0 - exp_home

        home_elos.append(h_elo)
        away_elos.append(a_elo)
        home_exps.append(exp_home)
        away_exps.append(exp_away)

        # After recording features, update ratings based on result
        r = str(row.get("FTR", "")).upper()
        if r == "H":
            s_home, s_away = 1.0, 0.0
        elif r == "A":
            s_home, s_away = 0.0, 1.0
        else:  # draw or unknown
            s_home, s_away = 0.5, 0.5

        ratings[home] = h_elo + k_factor * (s_home - exp_home)
        ratings[away] = a_elo + k_factor * (s_away - exp_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_exp"] = home_exps
    df["away_exp"] = away_exps
    return df


