"""TILT end to end on the OFFLINE tiny-random backbone (no download).

Marked `llm`: needs torch + transformers. Checks that TILT-SIR and TILT-SMC run on a
real engine, write every artefact, reuse ONE cached fine-tune (all TILT knobs are
sampling-only), and log their λ / KL / SMC statistics.
"""

from __future__ import annotations

import json

import pytest

from conftest import needs_llm, needs_xgb

pytestmark = [pytest.mark.llm, needs_llm, needs_xgb]

SMOKE = {"model": "tiny-random", "epochs": 60, "lr": 2e-3, "effective_batch": 16,
         "interpolation_ratio": 0.3, "max_rounds": 2, "oversample_attempts": 2.0}


def _item(method, preset, **kw):
    from tiltgen.experiments.runner import WorkItem
    return WorkItem("t", "synthetic", 0, method, "tilt", {**SMOKE, "preset": preset, **kw})


def test_tilt_sir_and_smc_end_to_end_share_one_fine_tune(isolated_runs):
    from tiltgen.experiments.runner import run_item
    m_sir = run_item(_item("tilt_sir", "tilt_sir", pool_factor=2.0), verbose=False)
    m_smc = run_item(_item("tilt_smc", "tilt_smc", pilot=48, particles=4, keep=2),
                     verbose=False)
    for m in (m_sir, m_smc):
        assert 0 <= m["f1"] <= 1 and "mia_auc" in m and "cost_elapsed_sec" in m
    d_sir = json.loads((isolated_runs / "runs/t/synthetic/tilt_sir/seed0/diagnostics.json")
                       .read_text())
    d_smc = json.loads((isolated_runs / "runs/t/synthetic/tilt_smc/seed0/diagnostics.json")
                       .read_text())
    assert d_sir["model_cache"].startswith("miss")
    assert d_smc["model_cache"] == "hit"                    # same corpus -> same fine-tune
    assert d_sir["tilt"]["kl"] <= d_sir["tilt"]["kl_budget"] + 1e-3
    assert "smc" in d_smc["tilt"] and d_smc["tilt"]["smc"]["completions"] > 0
