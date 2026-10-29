"""Every filesystem location the project uses, anchored to the project root.

palt2 used cwd-relative paths ("data/raw", "runs/"), so running a command from
anywhere but the repo root read or wrote the wrong place. Here every path is
derived from the package location, and each can be redirected with an
environment variable -- useful on a shared GPU node (H100) where data and runs
should live on fast scratch storage:

    IMBLLM_DATA   default <root>/data
    IMBLLM_RUNS   default <root>/runs
    IMBLLM_CACHE  default <runs>/_models      fine-tuned model cache
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../palt3

DATA = Path(os.environ.get("IMBLLM_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = DATA / "figures"

RUNS = Path(os.environ.get("IMBLLM_RUNS", ROOT / "runs"))
MODEL_CACHE = Path(os.environ.get("IMBLLM_CACHE", RUNS / "_models"))

CONFIGS = ROOT / "configs"
HARDWARE_CONFIGS = CONFIGS / "hardware"
EXPERIMENT_CONFIGS = CONFIGS / "experiments"
