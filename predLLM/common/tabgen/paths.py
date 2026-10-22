# Filesystem layout of the monorepo. Override the root with the PALT_ROOT env var.
import os
from pathlib import Path

ROOT = Path(os.environ.get("PALT_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"


def version_root(version: str) -> Path:
    """`v1` -> <root>/v1"""
    return ROOT / version


def runs_root(version: str) -> Path:
    return version_root(version) / "runs"
