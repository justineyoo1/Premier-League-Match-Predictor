"""Canonical schema definitions for match data.

Defines required columns, expected dtypes, and date formats for ingestion and
cleaning. Use this to centralize assumptions and keep pipelines consistent.
"""

from __future__ import annotations

from typing import Dict, List


REQUIRED_COLUMNS: List[str] = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]


DTYPE_STRINGS: Dict[str, str] = {
    "Date": "string",
    "HomeTeam": "string",
    "AwayTeam": "string",
    "FTHG": "int64",
    "FTAG": "int64",
    "FTR": "string",
}


# Normalized date format used after aggregation
CANONICAL_DATE_FORMAT = "%Y-%m-%d"


