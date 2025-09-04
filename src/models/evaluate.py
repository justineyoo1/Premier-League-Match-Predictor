"""Evaluation entrypoint for baseline models.

Computes log-loss, multi-class Brier score, and saves a calibration curve image
for home-win probability.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from ..config import load_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TARGET = "FTR"
CLASSES = ["H", "D", "A"]


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_onehot = np.zeros_like(proba)
    for i, y in enumerate(y_true):
        idx = class_to_idx[y]
        y_onehot[i, idx] = 1.0
    return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))


def main(config_path: str) -> None:
    config = load_config(config_path)
    model_type = config["model"]["type"].lower()
    output_dir = Path(config["training"]["output_dir"]).resolve()
    test_csv = config["data"].get("test_csv", "data/processed/test.csv")

    model_path = output_dir / f"{model_type}.joblib"
    clf = load(model_path)

    test = pd.read_csv(test_csv)
    y_true = test[TARGET].astype(str).values
    X_test = test.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR"])  # same selection as train

    proba = clf.predict_proba(X_test)
    try:
        model_classes = list(clf.named_steps["model"].classes_)
    except Exception:
        model_classes = CLASSES
    mlogloss = log_loss(y_true, proba, labels=model_classes)
    brier = multiclass_brier(y_true, proba, model_classes)

    # Calibration curve for home-win (class 'H')
    home_idx = model_classes.index("H") if "H" in model_classes else 0
    prob_home = proba[:, home_idx]
    y_home = (y_true == "H").astype(int)
    frac_pos, mean_pred = calibration_curve(y_home, prob_home, n_bins=10, strategy="uniform")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"log_loss": float(mlogloss), "brier": float(brier)}
    (output_dir / f"{model_type}_metrics.json").write_text(json.dumps(metrics, indent=2))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(mean_pred, frac_pos, marker="o", label="home-win")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (home win)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_calibration.png", dpi=150)
    plt.close()
    logger.info("Saved metrics and calibration plot to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation config")
    args = parser.parse_args()
    main(args.config)


