"""Training entrypoint for baseline models.

Trains RandomForest or XGBoost on engineered features. Expects time-based
train/valid CSVs already created. Persists model and metadata artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump

from ..config import load_config
from .baselines import build_random_forest, build_xgboost


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TARGET = "FTR"
CLASSES = ["H", "D", "A"]


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[TARGET].astype(str).values
    # Select numeric feature columns (exclude non-features)
    non_features = {"Date", "HomeTeam", "AwayTeam", "FTR"}
    X = df.drop(columns=[c for c in df.columns if c in non_features]).copy()
    return X, y


def main(config_path: str) -> None:
    config = load_config(config_path)
    train_csv = config["data"]["train_csv"]
    valid_csv = config["data"]["valid_csv"]
    model_type = config["model"]["type"].lower()
    model_params = config["model"].get("params", {})
    output_dir = Path(config["training"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    valid = pd.read_csv(valid_csv)

    X_train, y_train = _prepare_xy(train)
    X_valid, y_valid = _prepare_xy(valid)

    # Split feature types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    if model_type == "random_forest":
        estimator = build_random_forest(model_params)
    elif model_type == "xgboost":
        estimator = build_xgboost(model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    clf.fit(X_train, y_train)

    # Simple validation log-loss with model-defined class order
    proba = clf.predict_proba(X_valid)
    try:
        model_classes = list(clf.named_steps["model"].classes_)
    except Exception:
        model_classes = CLASSES
    val_logloss = log_loss(y_valid, proba, labels=model_classes)
    logger.info("Validation log-loss: %.4f", val_logloss)

    # Persist artifacts
    model_path = output_dir / f"{model_type}.joblib"
    dump(clf, model_path)

    # Persist columns metadata
    columns_meta = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "classes": model_classes,
    }
    (output_dir / "columns.json").write_text(json.dumps(columns_meta, indent=2))

    # Save run metadata
    run_meta = {"model": model_type, "params": model_params, "val_logloss": float(val_logloss)}
    (output_dir / f"{model_type}_run.json").write_text(json.dumps(run_meta, indent=2))
    logger.info("Saved model to %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config")
    args = parser.parse_args()
    main(args.config)


