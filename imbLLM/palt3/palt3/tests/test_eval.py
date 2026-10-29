"""Evaluation wiring: decision rules, both training sets, fixed metrics, stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_xgb
from imbllm.core.evaluate import quality, stats


@needs_xgb
def test_threshold_only_is_not_imbalance(splits):
    from imbllm.core.evaluate.downstream import evaluate_reference
    imb = evaluate_reference("imbalance", splits)
    thr = evaluate_reference("threshold_only", splits)
    pri = evaluate_reference("prior_shift", splits)
    assert imb["rule"] == "argmax" and thr["rule"] == "threshold" and pri["rule"] == "prior"
    # same classifier, different decision rule: AUC identical, the reported f1 is its own
    assert thr["auc"] == pytest.approx(imb["auc"])
    assert thr["f1"] == pytest.approx(imb["f1_tuned"])
    assert not np.isnan(thr["threshold"])


@needs_xgb
def test_threshold_rule_needs_a_real_validation_fold(paper_splits):
    from imbllm.core.evaluate.downstream import evaluate_reference
    assert "error" in evaluate_reference("threshold_only", paper_splits)
    assert np.isnan(evaluate_reference("imbalance", paper_splits)["f1_tuned"])


@needs_xgb
def test_synthetic_is_scored_both_ways(splits):
    from imbllm.core.evaluate.downstream import evaluate_synthetic
    syn = splits.minor_heldout.sample(n=splits.n_target, replace=True, random_state=0)
    m = evaluate_synthetic(syn[splits.schema.features], splits)
    assert m["real_n_train"] == m["n_train"] + len(splits.train_minor)
    assert {"f1", "auc", "auprc", "real_f1", "real_auc"} <= set(m)


@needs_xgb
def test_multiclass_downstream():
    from imbllm.core.data.protocol import prepare_frame
    from imbllm.core.data.registry import DatasetSpec
    from imbllm.core.evaluate.downstream import evaluate_reference
    rng = np.random.default_rng(0)
    y = np.array(["a"] * 300 + ["b"] * 200 + ["c"] * 60)
    X = rng.normal(size=len(y)) + (y == "c") * 2 + (y == "b") * 1
    df = pd.DataFrame({"x": X, "t": y})
    s = prepare_frame(df, DatasetSpec("mc", {"type": "synthetic"}, "t", "multiclass",
                                      numerical=("x",), categorical=()), seed=0)
    m = evaluate_reference("scale_pos_weight", s)
    assert 0 <= m["f1"] <= 1 and 0 <= m["macro_f1"] <= 1 and not np.isnan(m["auc"])


def test_quality_metric_space_ignores_the_synthetic_set(splits):
    """A synthetic outlier must not change the metric space (palt2 fitted its
    scaler on real + synthetic rows)."""
    s = splits.schema
    syn = splits.minor_heldout.copy()
    a = quality.quality_report(syn, splits)
    syn2 = pd.concat([syn, syn.iloc[[0]].assign(income=1e9)], ignore_index=True)
    b = quality.quality_report(syn2, splits)
    assert b["heldout_close_probability"] == pytest.approx(a["heldout_close_probability"])
    assert a["heldout_close_probability"] == 1.0          # identical rows are "close"
    assert a["dup_minor"] == 0.0                          # held-out rows are not training rows


def test_memorisation_is_detected(splits):
    q = quality.quality_report(splits.train_minor.sample(50, replace=True, random_state=0),
                               splits)
    assert q["dup_minor"] == 1.0 and q["nn_ratio"] == pytest.approx(0.0)
    assert q["paper_close_probability"] >= q["heldout_close_probability"]  # D5 contamination


def test_q_alpha_matches_demsar():
    for k, q in stats.DEMSAR_Q05.items():
        assert stats.q_alpha(k) == pytest.approx(q, abs=2e-3)
    assert stats.critical_difference(25, 10) > 0              # no k <= 15 limit


def test_failure_is_ranked_last_not_dropped():
    t = pd.DataFrame({"good": [.9, .9, .9, .9], "ok": [.8, .8, .8, .8],
                      "fails": [np.nan, .95, .95, .95]}, index=list("abcd"))
    r = stats.friedman_nemenyi(t)
    assert r["n_datasets"] == 4                               # palt2 dropped dataset "a"
    assert r["mean_ranks"]["fails"] == pytest.approx((3 + 1 + 1 + 1) / 4)
    assert r["failures"] == {"fails": 1}


def test_holm_is_monotone():
    adj = stats.holm({"a": .01, "b": .04, "c": .03})
    assert adj["a"] == pytest.approx(.03) and adj["b"] >= adj["c"] >= adj["a"]


def test_paired_reports_min_attainable_p():
    r = stats.paired({"d1": 1, "d2": 1, "d3": 1}, {"d1": 0, "d2": 0, "d3": 0})
    assert r["min_attainable_p"] == pytest.approx(0.25)       # N=3 can never be significant
