"""Data loaders for Premier League match data.

Provides functions to load data from CSV files and online APIs. Functions are
stubs and should be implemented with actual logic later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


logger = logging.getLogger(__name__)


def load_csv(csv_path: str | Path, *, dtype: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load match data from a CSV file.

    Args:
        csv_path: Path to the input CSV.
        dtype: Optional dtype mapping for pandas.

    Returns:
        A pandas DataFrame containing raw match data.
    """
    path = Path(csv_path)
    logger.info("Loading CSV data from %s", path)
    # Placeholder: implement robust reading, date parsing, and schema validation
    return pd.read_csv(path, dtype=dtype) if path.exists() else pd.DataFrame()


def load_from_api(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Stub for loading match data from an online API.

    Args:
        url: API endpoint URL.
        params: Optional query parameters.
        headers: Optional request headers (e.g., auth tokens).

    Returns:
        A pandas DataFrame with the API response normalized to tabular form.
    """
    logger.info("Fetching data from API: %s", url)
    # Placeholder: implement requests.get + json normalization
    return pd.DataFrame()


