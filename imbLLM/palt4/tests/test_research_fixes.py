"""Checks for the fixes listed in docs/RESEARCH_REVIEW.md (G1, G3, G4, G5, G8, 3.3).
No torch needed: LLM-facing code is exercised with fake engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imbLLM.palt4.tests.conftest import needs_xgb

from tiltgen.experiments import grids


# ----------------------------------------------------------------- G1 learners
@needs_xgb
def test_every_reference_is_scored_by_every_learner(splits, monkeypatch):
    from tiltgen.core.evaluate import downstream as ds
    monkeypatch.setenv("TILT_LEARNERS", "xgb,lr,rf,mlp")
    r = ds.evaluate_reference("threshold_only", splits, seed=0)
    for ln in ("lr", "rf", "mlp"):
        assert 0.0 <= r[f"{ln}_f1"] <= 1.0
        assert np.isfinite(r[f"{ln}_f1_tuned"])
    # the unprefixed keys are still the paper's XGBoost
    assert r["learner"] == "xgb"


@needs_xgb
def test_mlp_cannot_take_weights_and_reports_nan(splits, monkeypatch):
    from tiltgen.core.evaluate import downstream as ds
    monkeypatch.setenv("TILT_LEARNERS", "xgb,lr,mlp")
    r = ds.evaluate_reference("scale_pos_weight", splits, seed=0)
    assert np.isfinite(r["lr_f1"]) and np.isnan(r["mlp_f1"])


@needs_xgb
def test_learner_env_restricts_to_xgb(splits, monkeypatch):
    from tiltgen.core.evaluate import downstream as ds
    monkeypatch.setenv("TILT_LEARNERS", "xgb")
    r = ds.evaluate_reference("imbalance", splits, seed=0)
    assert not any(k.startswith(("lr_", "rf_", "mlp_")) for k in r)


# ----------------------------------------------------------------- G8 covariates
def test_covariates_see_the_bimodal_minority(splits):
    from tiltgen.core.evaluate import covariates
    c = covariates.compute(splits, seed=0)
    assert c["cov_minority_modes"] >= 2          # the synthetic minority is bimodal
    assert 0.0 <= c["cov_class_overlap"] <= 1.0
    assert c["cov_m_categorical"] == 2 and c["cov_imbalance_ratio"] > 1


def test_report_links_gain_to_covariates():
    from tiltgen.experiments.report import _covariate_links
    ds = [f"d{i}" for i in range(6)]
    mean = pd.DataFrame({"anchor": 0.5, "m": [0.5 + 0.01 * i for i in range(6)]}, index=ds)
    ok = pd.DataFrame({"dataset": ds, "cov_imbalance_ratio": range(6),
                       "cov_n_minor": [3, 1, 4, 1, 5, 9]})
    t = _covariate_links(ok, mean, "anchor")
    assert t.loc["m", "cov_imbalance_ratio"] == pytest.approx(1.0)


# ----------------------------------------------------------------- G5 curation
@needs_xgb
def test_cllm_keeps_real_minority_like_rows_and_drops_majority_rows(splits):
    from tiltgen.core.baselines.curation import cllm_filter
    s = splits.schema
    good = splits.minor_heldout[s.features].reset_index(drop=True)
    bad = splits.train_major[s.features].head(len(good)).reset_index(drop=True)
    kept_good, dg = cllm_filter(good, splits, seed=0)
    kept_bad, db = cllm_filter(bad, splits, seed=0)
    assert dg["kept_share"] > 0.6
    assert db["kept_share"] < 0.2
    assert list(kept_good.columns) == s.features


def test_new_grids_expand():
    order = grids.grid("order", ["synthetic"], [0])
    assert {i.config["permutation"] for i in order} == {"fix_y", "permute_xy", "identity"}
    assert all(i.config["measure_order_sensitivity"] for i in order)
    cur = grids.grid("curation", ["synthetic"], [0])
    assert sum(i.config.get("curate") == "cllm" for i in cur) == 3 == len(cur) // 2
    assert grids.ANCHORS["order"] in {i.method for i in order}
    assert grids.ANCHORS["curation"] in {i.method for i in cur}


def test_deep_baselines_listed_only_when_backend_installed():
    from tiltgen.core.baselines import deep
    base = {i.method for i in grids.grid("baselines", ["synthetic"], [0])}
    for name in deep.REGISTRY:
        assert (name in base) == deep.available(name)


def test_runner_hides_curate_from_method_config():
    from tiltgen.experiments.runner import WorkItem, build_llm
    it = WorkItem("x", "synthetic", 0, "v2_recommended+cllm", "v2",
                  {"preset": "recommended", "model": "tiny-random", "curate": "cllm"})
    gen = build_llm(it)                           # must not raise on the unknown key
    assert gen.cfg.model == "tiny-random"


# ----------------------------------------------------------------- G3 contamination
class Memoriser:
    """A 'model' that has memorised the raw CSV: continues any prompt from the file."""

    def __init__(self, text: str):
        self.text = text

    def complete(self, prompts, **kw):
        out = []
        for p in prompts:
            tail = p[-150:]                          # contiguous in the file (no header)
            i = self.text.find(tail)
            out.append((self.text[i + len(tail):][:200] if i >= 0 else "?", True))
        return out


class Amnesiac:
    def complete(self, prompts, **kw):
        return [("0,0,0", True) for _ in prompts]


def _raw(isolated_runs, frame):
    from tiltgen import paths
    paths.RAW.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.RAW / "synthetic.csv", index=False)
    return (paths.RAW / "synthetic.csv").read_text()


def test_contamination_flags_a_memoriser_and_not_an_amnesiac(isolated_runs, frame):
    from tiltgen.audits import contamination
    text = _raw(isolated_runs, frame)
    mem = contamination.probe("synthetic", Memoriser(text), trials=10)
    amn = contamination.probe("synthetic", Amnesiac(), trials=10)
    assert mem["row_exact_match"] > 0.8 and mem["memorisation_flag"]
    assert amn["row_exact_match"] == 0.0 and not amn["memorisation_flag"]
    assert mem["header_chars_matched"] == mem["header_chars_total"]
