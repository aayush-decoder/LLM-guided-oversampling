"""Fixtures: the built-in `synthetic` table run through the REAL protocol code.
Bimodal mixed-type minority -- the geometry where random-partner interpolation
fails. No download, no network."""

from __future__ import annotations

import importlib.util

import pytest

from imbllm.core.data import registry
from imbllm.core.data.protocol import prepare_frame


def has(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


needs_llm = pytest.mark.skipif(not has("torch", "transformers"),
                               reason="torch/transformers not installed")
needs_xgb = pytest.mark.skipif(not has("xgboost", "imblearn"),
                               reason="xgboost/imbalanced-learn not installed")


@pytest.fixture(scope="session")
def frame():
    return registry.synthetic_frame()


@pytest.fixture(scope="session")
def spec():
    return registry.get("synthetic")


@pytest.fixture
def splits(frame, spec):
    return prepare_frame(frame, spec, seed=0, q=0.2, protocol="clean")


@pytest.fixture
def paper_splits(frame, spec):
    return prepare_frame(frame, spec, seed=0, q=0.2, protocol="paper")


@pytest.fixture
def isolated_runs(tmp_path, monkeypatch):
    """Point runs/ and the model cache at a temp dir."""
    from imbllm import paths
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "MODEL_CACHE", tmp_path / "runs" / "_models")
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(paths, "RAW", tmp_path / "data" / "raw")
    return tmp_path
