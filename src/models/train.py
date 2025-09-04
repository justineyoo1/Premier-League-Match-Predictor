"""Training entrypoint for baseline models.

This module provides a CLI-like interface to train models using configuration
files for hyperparameters and data paths. Actual training logic should be
implemented later.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(config_path: str) -> None:
    """Train model as specified in the configuration file.

    Args:
        config_path: Path to YAML/JSON config defining data paths and params.
    """
    config = load_config(config_path)
    logger.info("Loaded training config: %s", config_path)
    # Placeholder: load data, build features, train model, save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "model.pkl").write_bytes(b"placeholder-model")
    logger.info("Training complete. Saved placeholder model to artifacts/model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config")
    args = parser.parse_args()
    main(args.config)


