"""Configuration loading.

All settings live in ``config.yaml`` at the project root. Paths in that file are
relative to the project root, so the pipeline works from any working directory.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and turn every entry under ``paths`` into an absolute Path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    for key in ("processed_dir", "models_dir", "figures_dir", "metrics_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg


def set_seed(seed: int) -> None:
    """Fix every random seed we rely on, so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
