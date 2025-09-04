"""Elo rating feature utilities.

Provide functions to compute and update Elo ratings per team and derive Elo-based
features such as rating differential and expected result.
"""

from __future__ import annotations

import logging
import pandas as pd


logger = logging.getLogger(__name__)


def compute_elo_features(matches: pd.DataFrame, *, k_factor: float = 20.0, base_rating: float = 1500.0) -> pd.DataFrame:
    """Stub to compute Elo ratings and features.

    Args:
        matches: Cleaned matches DataFrame.
        k_factor: Update magnitude for Elo updates.
        base_rating: Initial rating for all teams.

    Returns:
        DataFrame with Elo features appended.
    """
    logger.info("Computing Elo features (k=%s, base=%s)", k_factor, base_rating)
    # Placeholder implementation to be filled later
    return matches.copy()


