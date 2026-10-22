"""The protocol: splits, q-reduction, disjointness, regression/multiclass handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tiltgen.core.data import registry
from tiltgen.core.data.protocol import prepare_frame, sanitize
from tiltgen.core.data.registry import DatasetSpec


def test_q_reduction_and_sizes(splits):
    s = splits
    full = len(s.train_minor) + len(s.minor_heldout)
    assert len(s.train_minor) == max(2, round(0.2 * full))
    assert s.n_target == len(s.train_major)            # binary: |largest class| = |D_major|
    assert s.imbalance_ratio > 1


def test_heldout_val_test_are_disjoint_from_train(splits):
    key = lambda df: set(map(tuple, df.astype(str).to_numpy().tolist()))  # noqa: E731
    tr = key(splits.imbalanced_train)
    assert not (key(splits.minor_heldout) & key(splits.train_minor))
    assert len(splits.val) > 0
    # val keeps the ORIGINAL prevalence (it is used for threshold tuning)
    p_val = (splits.val["target"] == "yes").mean()
    p_all = 200 / 1200
    assert abs(p_val - p_all) < 0.05
    assert len(tr) > 0


def test_paper_protocol_has_no_validation_fold(paper_splits):
    assert len(paper_splits.val) == 0


def test_deterministic(frame, spec):
    a = prepare_frame(frame, spec, seed=3)
    b = prepare_frame(frame, spec, seed=3)
    pd.testing.assert_frame_equal(a.train_minor, b.train_minor)
    c = prepare_frame(frame, spec, seed=4)
    assert not a.test.equals(c.test)


def test_regression_threshold_comes_from_train_only():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    spec = DatasetSpec("reg", {"type": "synthetic"}, "y", "regression", numerical=("x",),
                       categorical=())
    s = prepare_frame(df, spec, seed=0, protocol="paper")
    assert s.schema.minority_label == "high" and s.schema.task == "binary"
    # threshold == 80th percentile of the TRAIN split (replicated), not of the whole table
    from sklearn.model_selection import train_test_split
    strat = pd.qcut(df["y"], 10, labels=False, duplicates="drop")
    train, _ = train_test_split(df, test_size=0.2, random_state=0, stratify=strat)
    assert s.schema.threshold == pytest.approx(float(train["y"].quantile(0.8)), abs=1e-12)
    frac = (s.imbalanced_train["y"] == "high").mean()
    assert 0.03 < frac < 0.2


def test_multiclass_generates_largest_class_size():
    rng = np.random.default_rng(1)
    y = np.array(["a"] * 300 + ["b"] * 150 + ["c"] * 40)
    df = pd.DataFrame({"x": rng.normal(size=len(y)), "t": y})
    spec = DatasetSpec("mc", {"type": "synthetic"}, "t", "multiclass", numerical=("x",),
                       categorical=())
    s = prepare_frame(df, spec, seed=0, protocol="paper")
    assert s.schema.minority_label == "c" and s.schema.majority_label == "a"
    counts = s.train_major["t"].value_counts()
    assert s.n_target == counts["a"] < len(s.train_major)


def test_balanced_multiclass_does_not_crash():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=150), "t": np.repeat(["0", "1", "2"], 50)})
    spec = DatasetSpec("iris_like", {"type": "synthetic"}, "t", "multiclass",
                       numerical=("x",), categorical=())
    s = prepare_frame(df, spec, seed=0)
    assert s.schema.minority_label != s.schema.majority_label


def test_sanitize_makes_serialisation_unambiguous():
    assert "," not in sanitize("a, b")
    assert " is " not in sanitize("this is it")


def test_registry_groups_cover_both_tables():
    assert len(registry.GROUPS["paper"]) == 10
    assert len(registry.GROUPS["predllm"]) == 20
    assert len(registry.GROUPS["all"]) == 28      # adult (M=13) and adult11 (M=11) both listed
    assert registry.resolve("fast,iris")[-1] == "iris"
    with pytest.raises(KeyError):
        registry.get("nope")
