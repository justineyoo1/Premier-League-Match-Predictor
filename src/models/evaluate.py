"""Evaluation entrypoint for baseline models.

Loads a trained model artifact and computes evaluation metrics on a validation
or test set as configured.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(config_path: str) -> None:
    """Evaluate model as specified in the configuration file."""
    config = load_config(config_path)
    logger.info("Loaded evaluation config: %s", config_path)
    # Placeholder: load data, build features, load model, compute metrics
    metrics_path = Path("artifacts/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{\n  \"accuracy\": 0.0\n}\n", encoding="utf-8")
    logger.info("Evaluation complete. Saved placeholder metrics to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation config")
    args = parser.parse_args()
    main(args.config)


