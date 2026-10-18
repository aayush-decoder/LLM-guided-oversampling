"""v3 unit tests: outlier detection, negative unlikelihood loss, schema partitions, and anomaly gating."""
import numpy as np
import pandas as pd
import pytest
import torch

from predllm_v3.codec import Schema, build_schema
from predllm_v3.curation import filter_by_anomaly_score
from predllm_v3.finetune import v3_unlikelihood_loss
from predllm_v3.outlier import AnomalyScorer, compute_outlier_ranges


def test_outlier_ranges_computation():
    # Synthetic dataset with known outlier values
    data = {"num1": [10.0, 11.0, 10.5, 12.0, 11.5, 10.2, 50.0], "label": [0, 0, 0, 0, 0, 0, 1]}
    df = pd.DataFrame(data)

    # 1. IQR mode
    iqr_ranges = compute_outlier_ranges(df, ["num1"], mode="iqr", factor=1.5)
    lo, hi = iqr_ranges["num1"]
    assert lo >= 10.0
    assert hi < 50.0  # 50.0 is an outlier, so upper IQR bound cuts off before 50.0

    # 2. Empirical mode
    emp_ranges = compute_outlier_ranges(df, ["num1"], mode="empirical")
    assert emp_ranges["num1"] == (10.0, 50.0)

    # 3. Minority mode
    min_ranges = compute_outlier_ranges(df, ["num1"], mode="minority", target_col="label", minority_classes=[1])
    # Minority is 50.0
    assert min_ranges["num1"][1] == 50.0


def test_anomaly_scorer_and_gate():
    rng = np.random.default_rng(42)
    # 200 normal points + 10 extreme outliers
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    outliers = rng.normal(loc=10.0, scale=1.0, size=(10, 2))
    X = np.vstack([normal, outliers])
    df = pd.DataFrame(X, columns=["feat1", "feat2"])

    scorer = AnomalyScorer(contamination=0.05, seed=42).fit(df)
    assert scorer.fitted

    # Decision function should rank extreme outliers as much lower (more anomalous)
    normal_score = scorer.score(normal[:10]).mean()
    outlier_score = scorer.score(outliers).mean()
    assert outlier_score < normal_score

    # Filter pool candidates
    keep, stats = filter_by_anomaly_score(
        df,
        ["feat1", "feat2"],
        scorer,
        target_quantile_low=0.0,
        target_quantile_high=0.85,
    )
    assert stats["anomaly_kept_ratio"] > 0.0

    # Test with mixed features where scorer was trained only on numeric subset
    df_mixed = df.copy()
    df_mixed["cat_col"] = [0.0] * len(df_mixed)
    keep_mixed, _ = filter_by_anomaly_score(
        df_mixed,
        ["feat1", "feat2", "cat_col"],
        scorer,
    )
    assert len(keep_mixed) == len(df_mixed)


def test_v3_negative_unlikelihood_loss():
    # Setup logits: 2 batch elements, length 3, vocab 5
    B, L, V = 2, 3, 5
    logits = torch.zeros((B, L, V), requires_grad=True)
    ids = torch.tensor([[1, 2, 3], [1, 4, 3]], dtype=torch.long)
    weights = torch.ones((B, L), dtype=torch.float)

    # First item is positive (minority), second is negative (majority)
    is_neg = torch.tensor([0.0, 1.0])

    # 1. Negative weight = 0.0 (reduces to positive loss only)
    loss_pos_only, stats0 = v3_unlikelihood_loss(logits, ids, weights, is_neg, neg_weight=0.0)
    assert stats0["loss_neg"] == 0.0
    assert loss_pos_only.item() > 0.0

    # 2. Negative weight > 0 (includes majority repulsion penalty)
    loss_with_neg, stats1 = v3_unlikelihood_loss(logits, ids, weights, is_neg, neg_weight=0.8)
    assert stats1["loss_neg"] > 0.0
    assert loss_with_neg.item() > loss_pos_only.item()

    # Backprop test
    loss_with_neg.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_v3_schema_active_ranges():
    sch = Schema(
        features=["f1", "f2"],
        target="y",
        task="classification",
        kinds={"f1": "num", "f2": "cat", "y": "cat"},
        decimals={"f1": 2},
        ranges={"f1": (0.0, 100.0)},
        outlier_ranges={"f1": (5.0, 45.0)},
        cats={"f2": ["a", "b"], "y": ["0", "1"]},
        minority_classes=["1"],
        majority_classes=["0"],
    )
    assert sch.active_range("f1", range_mode="outlier") == (5.0, 45.0)
    assert sch.active_range("f1", range_mode="empirical") == (0.0, 100.0)
    spec_outlier = sch.numeric_spec("f1", range_mode="outlier")
    assert spec_outlier[0] == 2  # max int digits for 45 is 2
