"""Baseline models.

Provide a RandomForest baseline and stubs for additional algorithms like
XGBoost, Poisson regression, and ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RandomForestParams:
    """Hyperparameters for RandomForest baseline."""

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42


def build_random_forest(params: Dict[str, Any] | None = None) -> Any:
    """Return a placeholder for a RandomForest model.

    Replace with sklearn.ensemble.RandomForestClassifier during implementation.
    """
    config = params or {}
    return {"model": "RandomForestClassifier", "params": config}


def build_xgboost(params: Dict[str, Any] | None = None) -> Any:
    """Stub for XGBoost model."""
    return {"model": "XGBClassifier", "params": params or {}}


def build_poisson(params: Dict[str, Any] | None = None) -> Any:
    """Stub for Poisson regression model."""
    return {"model": "PoissonRegressor", "params": params or {}}


def build_ensemble(params: Dict[str, Any] | None = None) -> Any:
    """Stub for an ensemble of models."""
    return {"model": "Ensemble", "params": params or {}}


