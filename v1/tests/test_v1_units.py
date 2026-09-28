"""Offline unit tests for v1 (no LLM download)."""
import numpy as np

from predllm_v1.finetune import RowTextDataset
from predllm_v1.methods import PRESETS, resolve_methods
from predllm_v1.serialize import column_order, encode, parse, row_fields
from tabgen.data.preprocess import TARGET

FEATS = ["Age", "Edu", "Job"]


def test_permute_x_keeps_target_last_and_permute_xy_moves_it():
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = column_order(FEATS, "x", rng)
        assert o[-1] == TARGET and sorted(o[:-1]) == sorted(FEATS)
    assert any(column_order(FEATS, "xy", rng)[-1] != TARGET for _ in range(50))
    assert column_order(FEATS, "x", rng, original=True) == FEATS + [TARGET]


def test_encode_parse_roundtrip():
    f = row_fields({"Age": 0.25, "Edu": 1.0, "Job": 0.5, TARGET: 1}, FEATS, "classification")
    s = encode(f, ["Job", "Age", "Edu", TARGET])
    assert s == "Job is 0.5, Age is 0.25, Edu is 1, target is 1"
    assert parse(s, FEATS + [TARGET]) == {"Job": 0.5, "Age": 0.25, "Edu": 1.0, TARGET: 1.0}


def test_parse_rejects_garbage():
    r = parse("Age is 0.3, Age is 0.9, Foo is 1, Edu is abc, Job is 0.1 extra", FEATS)
    assert r == {"Age": 0.3, "Job": 0.1}


class CharTok:
    eos_token_id = 0

    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}


def test_dataset_is_permuted_plus_original():
    fields = [row_fields({"Age": 0.1 * i, "Edu": 0.2, "Job": 0.3, TARGET: i % 2}, FEATS, "classification") for i in range(5)]
    ds = RowTextDataset(fields, FEATS, CharTok(), "x", augment_original=True, seed=0)
    assert len(ds) == 10
    assert ds.text(7).startswith("Age is 0.2, Edu is 0.2, Job is 0.3")   # original order in 2nd half
    assert ds[0]["input_ids"][-1] == 0                                   # EOS appended
    assert len(RowTextDataset(fields, FEATS, CharTok(), "xy", augment_original=False)) == 5


def test_presets_match_table3():
    assert PRESETS["pred_llm"].permutation == "x" and PRESETS["pred_llm"].sampling == "feature"
    assert PRESETS["great"].query == "none" and PRESETS["taptap"].query == "classifier"
    assert len(resolve_methods("ablation")) == 8
