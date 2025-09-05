"""Config utilities for loading YAML/JSON configuration files.

Use these helpers to keep code config-driven and reproducible.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a configuration file from YAML or JSON.

    Args:
        config_path: Path to a YAML or JSON config file.

    Returns:
        Parsed configuration as a dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    logger.info("Loading config from %s", path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {path.suffix}")


