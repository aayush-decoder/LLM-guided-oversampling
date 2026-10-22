"""Config precedence, run bookkeeping, and the v1 == v2(v1_exact) guarantee."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tiltgen import config as C
from tiltgen.methods.v1_paper import method as v1m
from tiltgen.methods.v2_corrected import method as v2m


def test_precedence_preset_then_yaml_then_cli():
    cfg = C.resolve(v2m.V2Config, presets=v2m.PRESETS,
                    yaml_cfg={"preset": "full", "tau": 0.5, "gibbs_sweeps": 2},
                    cli={"tau": 0.25})
    assert cfg.preset == "full"
    assert cfg.gibbs_sweeps == 2          # YAML beats preset (palt2: preset won)
    assert cfg.tau == 0.25                # CLI beats YAML
    assert cfg.interpolation_mode == "knn"


def test_unknown_keys_and_values_are_rejected():
    with pytest.raises(ValueError):
        C.resolve(v2m.V2Config, cli={"not_a_field": 1})
    with pytest.raises(ValueError):
        v1m.V1Config(finetune_on="nonsense")
    with pytest.raises(ValueError):
        v2m.V2Config(tau=1.5)


def test_hash_is_stable_and_sensitive():
    a = C.stable_hash({"x": 1, "y": [1, 2]})
    assert a == C.stable_hash({"y": [1, 2], "x": 1})
    assert a != C.stable_hash({"x": 2, "y": [1, 2]})


def _v1_exact():
    return C.resolve(v2m.V2Config, presets=v2m.PRESETS, preset="v1_exact")


def test_v1_exact_corpus_is_identical_to_v1(splits):
    cfg1 = v1m.V1Config()
    cfg2 = _v1_exact()
    t1, _ = v1m.build_corpus(splits, np.random.default_rng(5), cfg=cfg1)
    t2, _ = v2m.build_corpus(splits, np.random.default_rng(5), cfg=cfg2)
    assert t1 == t2


def test_v1_exact_prompts_are_identical_to_v1(splits):
    p1 = v1m.strategies(v1m.V1Config()).prompts(300, splits, np.random.default_rng(2))
    p2 = v2m.strategies(_v1_exact()).prompts(300, splits, np.random.default_rng(2))
    assert p1 == p2


def test_recommended_differs_only_by_named_flags():
    rec = C.resolve(v2m.V2Config, presets=v2m.PRESETS, preset="recommended")
    ex = _v1_exact()
    diff = {k for k in v2m.V1_VALUES if getattr(rec, k) != getattr(ex, k)}
    assert diff == set(v2m.V1_VALUES) - {"gibbs_sweeps"}


def test_v2_corpus_carries_anchor_categoricals(splits):
    t, d = v2m.build_corpus(splits, np.random.default_rng(0),
                            cfg=C.resolve(v2m.V2Config, presets=v2m.PRESETS,
                                          preset="recommended"))
    assert d["composition"]["share_with_categoricals"] == 1.0
    assert all(" edu is " in x or x.startswith("edu is ") for x in t)


def test_runcontext_never_reports_stale_metrics(isolated_runs):
    from tiltgen.core.runs import RunContext, collect
    with RunContext("e", "d", "m", 0, {"a": 1}) as ctx:
        ctx.write("metrics.json", {"f1": 0.9})
    assert collect("e")["f1"].tolist() == [0.9]
    with pytest.raises(RuntimeError):
        with RunContext("e", "d", "m", 0, {"a": 1}):
            raise RuntimeError("boom")       # rerun fails before writing metrics
    df = collect("e")
    assert df["status"].tolist() == ["error"] and "f1" not in df.columns
    man = json.loads((isolated_runs / "runs/e/d/m/seed0/manifest.json").read_text())
    assert man["status"] == "error" and man["config_hash"]


def test_shard_partitions_items():
    from tiltgen.core.runs import shard
    items = list(range(10))
    parts = [shard(items, f"{i}/3") for i in range(3)]
    assert sorted(sum(parts, [])) == items
