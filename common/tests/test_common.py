"""Offline unit tests for the shared library (iris ships with scikit-learn, no download needed)."""
import numpy as np
import pandas as pd
import pytest

from tabgen.data.preprocess import TARGET, TablePreprocessor, make_split
from tabgen.eval.fidelity import fidelity_report
from tabgen.eval.tstr import tstr
from tabgen.llm.registry import resolve_llm
from tabgen.utils import apply_overrides, format_number


@pytest.mark.parametrize("v,d,s", [(0.5, 2, "0.5"), (3.0, 2, "3"), (-0.0, 2, "0"), (0.125, 2, "0.12"),
                                   (16884.924, 4, "16884.924"), (39.0, 0, "39")])
def test_format_number(v, d, s):
    assert format_number(v, d) == s


def test_iris_split_protocol():
    sp = make_split("iris", seed=0)
    assert len(sp.train) == 120 and len(sp.test) == 30 and sp.n_classes == 3
    feats = sp.train[sp.features]
    assert feats.min().min() >= 0 and feats.max().max() <= 1
    assert (feats.round(2) == feats).all().all()                      # 2-decimal rounding (paper)
    assert sorted(sp.train[TARGET].unique()) == [0, 1, 2]
    assert make_split("iris", seed=0).train.equals(sp.train)          # deterministic per seed
    assert not make_split("iris", seed=1).train.equals(sp.train)


def test_preprocessor_roundtrip_categorical():
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0], "job": ["a", "b", "c", "a"], "y": ["no", "yes", "no", "yes"]})
    pre = TablePreprocessor(["age"], ["job"], ["age", "job"], "y", "classification").fit(df)
    enc = pre.transform(df)
    assert enc["job"].tolist() == [0.0, 0.5, 1.0, 0.0] and enc[TARGET].tolist() == [0, 1, 0, 1]
    back = pre.inverse_transform(enc)
    assert back["job"].tolist() == df["job"].tolist() and back["y"].tolist() == df["y"].tolist()
    # the paper rounds MinMax codes to 2 decimals, so numerics come back within half a
    # step of the range (30 -> 0.33 -> 29.9); categoricals and labels are exact
    assert np.allclose(back["age"], df["age"], atol=0.005 * (50 - 20) + 1e-9)


def test_tstr_and_fidelity_on_real_data():
    sp = make_split("iris", seed=0)
    res = tstr(sp.train, sp.test, sp.features, sp.task, sp.n_classes, 0)
    assert res["valid"] and res["accuracy"] > 0.8
    bad = sp.train.assign(**{TARGET: 0})
    assert tstr(bad, sp.test, sp.features, sp.task, sp.n_classes, 0)["accuracy"] == 0.0   # paper rule
    fid = fidelity_report(sp.train, sp.train, sp.features, set(), seed=0)
    assert fid["inverse_kl"] > 0.99 and fid["exact_copy_rate"] == 1.0 and fid["coverage"] == 1.0


def test_llm_switch():
    assert resolve_llm("1.5b").hf_id == "Qwen/Qwen2.5-1.5B"
    assert resolve_llm("3b").mode == "qlora"
    assert resolve_llm("distilgpt2", "lora").mode == "lora"
    assert resolve_llm("some-org/unlisted-model").mode == "lora"


def test_overrides():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=5", "a.c=true", "d=[1, 2]"])
    assert cfg == {"a": {"b": 5, "c": True}, "d": [1, 2]}
