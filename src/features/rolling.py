"""Rolling statistics utilities.

Provide helpers to compute team-level rolling aggregates such as goals for/against,
win rates, and form over N previous matches.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


def compute_team_rolling_features(matches: pd.DataFrame, *, window_sizes: List[int] | None = None) -> pd.DataFrame:
    """Stub to compute rolling features for each team.

    Args:
        matches: Cleaned matches DataFrame.
        window_sizes: List of window lengths to compute.

    Returns:
        DataFrame with additional rolling feature columns.
    """
    logger.info("Computing rolling features")
    # Placeholder implementation to be filled later
    return matches.copy()


