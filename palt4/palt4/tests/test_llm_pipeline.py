"""End-to-end LLM paths on the OFFLINE tiny-random backbone (no download).

Marked `llm`: needs torch + transformers. Runs on CPU in ~1-2 min, faster on GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import needs_llm, needs_xgb

pytestmark = [pytest.mark.llm, needs_llm, needs_xgb]

SMOKE = {"model": "tiny-random", "epochs": 60, "lr": 2e-3, "effective_batch": 16,
         "interpolation_ratio": 0.3, "max_rounds": 2, "oversample_attempts": 2.0}


def _item(method, kind, cfg):
    from tiltgen.experiments.runner import WorkItem
    return WorkItem("t", "synthetic", 0, method, kind, cfg)


def test_engine_trains_fp32_params_and_learns_eos():
    from tiltgen.core.llm import hardware, zoo
    from tiltgen.core.llm.engine import LLMEngine, TrainCfg
    spec = zoo.get("tiny-random")
    eng = LLMEngine(spec, hardware.plan(spec, hardware.detect(), effective_batch=16),
                    seed=0).load()
    texts = [f"y is a, u is {i % 5}, v is w{i % 3}" for i in range(64)]
    hist = eng.finetune(texts, TrainCfg(epochs=40, lr=3e-3))
    assert all(p.dtype.is_floating_point and p.dtype.itemsize == 4
               for p in eng.model.parameters() if p.requires_grad)
    assert hist[-1]["loss"] < hist[0]["loss"]
    outs = eng.complete(["y is a,"] * 16, max_new_tokens=60)
    assert np.mean([o[1] for o in outs]) > 0.5            # rows END (EOS learned)
    assert outs[0][0].startswith(" ")                     # prompt ends at "," -> " col is"
    # score = exact sum of token log-probs of the suffix
    s = eng.score(["y is a, u is"], [[" 3", " 3, v is w1"]], add_eos=False)[0]
    assert s[1] < s[0] <= 0
    eng.release()


def test_v1_and_v2_end_to_end_with_model_cache(isolated_runs):
    from tiltgen.experiments.runner import run_item
    m1 = run_item(_item("v1", "v1", dict(SMOKE)), verbose=False)
    m2 = run_item(_item("v2", "v2", {**SMOKE, "preset": "recommended"}), verbose=False)
    for m in (m1, m2):
        assert 0 <= m["f1"] <= 1 and "real_f1" in m and "heldout_close_probability" in m
    run = isolated_runs / "runs/t/synthetic/v2/seed0"
    ps = json.loads((run / "parse_stats.json").read_text())
    assert ps["yield_rate"] > 0 and ps["completions"] >= ps["parsed"]
    d = json.loads((run / "diagnostics.json").read_text())
    assert d["model_cache"].startswith("miss")
    assert "order_sensitivity" in d and "tau_bridge" in d and "frontier" in d
    assert d["corpus"]["composition"]["share_with_categoricals"] == 1.0
    # tau is sampling-only -> the SAME fine-tuned model is reused
    run_item(_item("v2_tau05", "v2", {**SMOKE, "preset": "recommended", "tau": 0.5}),
             verbose=False)
    d2 = json.loads((isolated_runs / "runs/t/synthetic/v2_tau05/seed0/diagnostics.json")
                    .read_text())
    assert d2["model_cache"] == "hit"
    assert d2["model_fingerprint"] == d["model_fingerprint"]


def test_llm_baselines_and_gibbs(isolated_runs):
    from tiltgen.experiments.runner import run_item
    for b in ("taptap_style", "predllm_style"):
        m = run_item(_item(b, "llm_base", {**SMOKE, "baseline": b}), verbose=False)
        assert "f1" in m
    m = run_item(_item("full", "v2", {**SMOKE, "preset": "full", "gibbs_max_rows": 8}),
                 verbose=False)
    d = json.loads((isolated_runs / "runs/t/synthetic/full/seed0/diagnostics.json").read_text())
    assert d["refine"]["rows_refined"] == 8
    assert "f1" in m


def test_prior_2x2_arms_run(isolated_runs):
    from tiltgen.experiments.runner import run_item
    m = run_item(_item("anon", "v1", {**SMOKE, "anonymize_columns": True}), verbose=False)
    syn = (isolated_runs / "runs/t/synthetic/anon/seed0/synthetic.csv").read_text()
    assert syn.splitlines()[0].startswith("age")         # renamed back to real columns
    assert "f1" in m
