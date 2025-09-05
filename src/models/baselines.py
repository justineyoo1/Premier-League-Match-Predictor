"""Baseline models.

Provide a RandomForest baseline and XGBoost model builders. Additional
algorithms like Poisson regression and ensembles can be added later.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - xgboost optional at import time
    XGBClassifier = None  # type: ignore


def build_random_forest(params: Dict[str, Any] | None = None) -> Any:
    cfg = {"n_estimators": 200, "max_depth": None, "random_state": 42}
    if params:
        cfg.update(params)
    return RandomForestClassifier(**cfg)


def build_xgboost(params: Dict[str, Any] | None = None) -> Any:
    if XGBClassifier is None:
        raise ImportError("xgboost is not available. Install xgboost to use this model.")
    cfg = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": 42,
        "n_jobs": 4,
    }
    if params:
        cfg.update(params)
    return XGBClassifier(**cfg)



