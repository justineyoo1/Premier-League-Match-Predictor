"""Simple CLI stub for offline predictions.

Loads model and config, then runs a placeholder prediction. Replace with
actual preprocessing and model inference code later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config


def main(config_path: str) -> None:
    _ = load_config(config_path)
    result = {
        "home_win_probability": 0.33,
        "draw_probability": 0.34,
        "away_win_probability": 0.33,
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/prediction.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)


