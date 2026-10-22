"""The audit's mathematical claims, checked against the SHIPPED code."""

from __future__ import annotations

import numpy as np
import pytest

from tiltgen.audits import math_checks, paper_tables
from tiltgen.methods.v1_paper import interpolate as v1i
from tiltgen.methods.v2_corrected import interpolate as v2i


@pytest.mark.parametrize("check", math_checks.ALL, ids=lambda f: f.__name__)
def test_math_check(check):
    name, ok, details = check()
    assert ok, (name, details)


def test_paper_table_transcription_is_exact():
    ok, res = paper_tables.check_transcription()
    assert ok, res
    assert paper_tables.TABLE_II["GReaT"][1] == .2059       # palt2 had .2760
    assert paper_tables.TABLE_II["TapTap"][1] == .2069      # palt2 had .2059


def test_paper_stats_corrected_values():
    t = paper_tables.omitted_tests()
    assert t["friedman_chi2"] == pytest.approx(24.882, abs=1e-3)   # palt2 reported 22.578
    assert set(t["not_separable_from_best"]) == {"ImbLLM", "SMOTE-NC", "CTGAN", "Pred-LLM"}


def test_ablation_decomposition():
    d = paper_tables.ablation_decomposition()
    assert d["C1_alone"] < 0 and d["C2_marginal_in_full"] == pytest.approx(0.0004)


def test_knn_uses_standardised_neighbours(splits):
    """With income in the tens of thousands, raw-space kNN = nearest income only.
    In z-scored space the partner is close on EVERY feature."""
    rng = np.random.default_rng(0)
    s = splits.schema
    minor = splits.train_minor_full
    Z, i, j = v2i.interpolate(minor, s, 400, rng, mode="knn")
    X = minor[s.continuous].astype(float)
    zs = ((X - X.mean()) / X.std()).to_numpy()
    d_partner = np.linalg.norm(zs[i] - zs[j], axis=1).mean()
    d_random = np.linalg.norm(zs[i] - zs[rng.permutation(i)], axis=1).mean()
    assert d_partner < 0.5 * d_random


def test_anchor_categoricals_keep_the_joint(splits):
    """v2 inherits categoricals from the ANCHOR: the bimodal fixture's
    age <-> edu association survives. palt2's random donors destroyed it."""
    rng = np.random.default_rng(1)
    s = splits.schema
    minor = splits.train_minor_full
    Z, i, _ = v2i.interpolate(minor, s, 3000, rng, mode="knn")
    Z = v2i.with_anchor_categoricals(Z, minor, i, s)
    young = Z["age"] < 45
    master_young = (Z.loc[young, "edu"] == "master").mean()
    master_old = (Z.loc[~young, "edu"] == "master").mean()
    assert master_old - master_young > 0.5


def test_random_mode_is_v1_exactly(splits):
    s = splits.schema
    a = v1i.interpolate(splits.train_minor, s, 200, np.random.default_rng(9))[0]
    b = v2i.interpolate(splits.train_minor, s, 200, np.random.default_rng(9), mode="random")[0]
    assert a.equals(b)
